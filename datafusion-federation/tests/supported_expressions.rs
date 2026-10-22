//! Expressions a remote cannot evaluate.
//!
//! A plan node with an expression the executor does not support
//! (`SQLExecutor::supports_expression`) remains in the local plan, and its inputs are
//! federated, as for an extension node it does not support. Here the remote is said to
//! lack `upper`.

#![cfg(feature = "sql")]

mod support;

use std::sync::Arc;

use async_trait::async_trait;
use datafusion::{
    arrow::{array::AsArray, datatypes::SchemaRef},
    common::tree_node::TreeNode,
    error::Result,
    physical_plan::{PhysicalExpr, SendableRecordBatchStream},
    prelude::{Expr, SessionContext},
    sql::unparser::dialect::Dialect,
};
use datafusion_federation::sql::SQLExecutor;

use support::{
    overwrite_default_schema, recorded_sql, remote_ctx, schema_provider, RecordingSQLExecutor,
};

/// A remote without `upper`.
struct WithoutUpper(RecordingSQLExecutor);

#[async_trait]
impl SQLExecutor for WithoutUpper {
    fn name(&self) -> &str {
        self.0.name()
    }

    fn compute_context(&self) -> Option<String> {
        self.0.compute_context()
    }

    fn dialect(&self) -> Arc<dyn Dialect> {
        self.0.dialect()
    }

    fn supports_expression(&self, expr: &Expr) -> bool {
        !expr
            .exists(|e| Ok(matches!(e, Expr::ScalarFunction(f) if f.name() == "upper")))
            .unwrap()
    }

    fn execute(
        &self,
        query: &str,
        schema: SchemaRef,
        filters: &[Arc<dyn PhysicalExpr>],
    ) -> Result<SendableRecordBatchStream> {
        self.0.execute(query, schema, filters)
    }

    async fn table_names(&self) -> Result<Vec<String>> {
        self.0.table_names().await
    }

    async fn get_table_schema(&self, table_name: &str) -> Result<SchemaRef> {
        self.0.get_table_schema(table_name).await
    }
}

/// `sql` run through `executor`'s remote: the optimized plan and the first column's
/// values, as text, in order.
async fn run(executor: Arc<dyn SQLExecutor>, sql: &str) -> (String, Vec<String>) {
    let schema = schema_provider(executor, &["test"]).await;
    let state = datafusion_federation::default_session_state();
    overwrite_default_schema(&state, schema);
    let ctx = SessionContext::new_with_state(state);
    let df = ctx.sql(sql).await.unwrap();
    let plan = df
        .clone()
        .into_optimized_plan()
        .unwrap()
        .display_indent()
        .to_string();
    let mut values = Vec::new();
    for batch in df.collect().await.unwrap() {
        let column = datafusion::arrow::compute::cast(
            batch.column(0),
            &datafusion::arrow::datatypes::DataType::Utf8,
        )
        .unwrap();
        values.extend(
            column
                .as_string::<i32>()
                .iter()
                .map(|v| v.unwrap().to_owned()),
        );
    }
    (plan, values)
}

const PROJECTED: &str = "SELECT upper(foo) AS u FROM test WHERE bar > 1 ORDER BY u";
const FILTERED: &str = "SELECT foo FROM test WHERE upper(foo) = 'B' AND bar > 1";

#[tokio::test]
async fn an_expression_the_remote_can_evaluate_is_federated() {
    let ctx = remote_ctx("test", "test.csv").await;
    let executor = RecordingSQLExecutor::new("sqlite", "sqlite_exec", ctx);
    let queries = executor.queries();

    let (plan, values) = run(Arc::new(executor), PROJECTED).await;

    assert_eq!(values, ["B", "C"]);
    assert!(plan.starts_with("Federated"), "all of it is sent:\n{plan}");
    assert!(recorded_sql(&queries).contains("upper("));
}

#[tokio::test]
async fn a_projection_the_remote_cannot_evaluate_is_computed_locally() {
    let ctx = remote_ctx("test", "test.csv").await;
    let executor = RecordingSQLExecutor::new("sqlite", "sqlite_exec", ctx);
    let queries = executor.queries();

    let (plan, values) = run(Arc::new(WithoutUpper(executor)), PROJECTED).await;

    assert_eq!(values, ["B", "C"]);
    let projection = plan
        .find("upper")
        .unwrap_or_else(|| panic!("upper is computed here:\n{plan}"));
    let federated = plan
        .find("Federated")
        .unwrap_or_else(|| panic!("its input is federated:\n{plan}"));
    assert!(projection < federated, "{plan}");
    let sql = recorded_sql(&queries);
    assert!(
        sql.contains("bar > 1"),
        "the filter below it is sent: {sql}"
    );
    assert!(!sql.contains("upper"), "upper is not: {sql}");
}

#[tokio::test]
async fn a_filter_the_remote_cannot_evaluate_is_computed_locally() {
    let ctx = remote_ctx("test", "test.csv").await;
    let executor = RecordingSQLExecutor::new("sqlite", "sqlite_exec", ctx);
    let queries = executor.queries();

    let (plan, values) = run(Arc::new(WithoutUpper(executor)), FILTERED).await;

    assert_eq!(values, ["b"]);
    assert!(
        plan.find("upper").unwrap() < plan.find("Federated").unwrap(),
        "the filter is computed here:\n{plan}"
    );
    let sql = recorded_sql(&queries);
    assert!(!sql.contains("upper"), "upper is not sent: {sql}");
}
