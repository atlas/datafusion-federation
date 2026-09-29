//! Extension nodes in a federated plan.
//!
//! A sub-plan containing an extension node is federated when the executor supports the
//! node (`SQLExecutor::supports_extension_node`), and is unparsed with the executor's
//! `SQLExecutor::extension_unparsers`. Otherwise the node remains in the local plan and
//! its inputs are federated.
//!
//! The extension node used here, [`Fence`], unparses as its input with `OFFSET 0`.

#![cfg(feature = "sql")]

mod support;

use std::{fmt, sync::Arc};

use async_trait::async_trait;
use datafusion::{
    arrow::datatypes::SchemaRef,
    common::{internal_err, DFSchemaRef},
    error::Result,
    execution::{
        context::{QueryPlanner, SessionState},
        SessionStateBuilder, TaskContext,
    },
    logical_expr::{
        col, lit, Extension, LogicalPlan, LogicalPlanBuilder, UserDefinedLogicalNode,
        UserDefinedLogicalNodeCore,
    },
    physical_plan::{
        DisplayAs, DisplayFormatType, ExecutionPlan, PhysicalExpr, PlanProperties,
        SendableRecordBatchStream,
    },
    physical_planner::{DefaultPhysicalPlanner, ExtensionPlanner, PhysicalPlanner},
    prelude::{Expr, SessionContext},
    sql::{
        sqlparser::ast::{self, Statement},
        unparser::{
            ast::{DerivedRelationBuilder, QueryBuilder, RelationBuilder, SelectBuilder},
            dialect::Dialect,
            extension_unparser::{UnparseWithinStatementResult, UserDefinedLogicalNodeUnparser},
            Unparser,
        },
    },
};
use datafusion_federation::{sql::SQLExecutor, FederatedPlanner};

use support::{
    overwrite_default_schema, recorded_sql, remote_ctx, row_count, schema_provider,
    RecordingSQLExecutor,
};

/// An extension node with one input and the same schema as it.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd)]
struct Fence {
    input: LogicalPlan,
}

impl Fence {
    fn wrap(input: LogicalPlan) -> LogicalPlan {
        LogicalPlan::Extension(Extension {
            node: Arc::new(Self { input }),
        })
    }
}

impl UserDefinedLogicalNodeCore for Fence {
    fn name(&self) -> &str {
        "Fence"
    }

    fn inputs(&self) -> Vec<&LogicalPlan> {
        vec![&self.input]
    }

    fn schema(&self) -> &DFSchemaRef {
        self.input.schema()
    }

    fn expressions(&self) -> Vec<Expr> {
        vec![]
    }

    fn fmt_for_explain(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Fence")
    }

    fn with_exprs_and_inputs(
        &self,
        _exprs: Vec<Expr>,
        mut inputs: Vec<LogicalPlan>,
    ) -> Result<Self> {
        Ok(Self {
            input: inputs.swap_remove(0),
        })
    }
}

/// Unparses a [`Fence`] as a derived table of its input with `OFFSET 0`. An enclosing
/// `SubqueryAlias` sets the alias.
#[derive(Debug)]
struct FenceUnparser;

impl UserDefinedLogicalNodeUnparser for FenceUnparser {
    fn unparse(
        &self,
        node: &dyn UserDefinedLogicalNode,
        unparser: &Unparser,
        _query: &mut Option<&mut QueryBuilder>,
        _select: &mut Option<&mut SelectBuilder>,
        relation: &mut Option<&mut RelationBuilder>,
    ) -> Result<UnparseWithinStatementResult> {
        let Some(fence) = node.as_any().downcast_ref::<Fence>() else {
            return Ok(UnparseWithinStatementResult::Unmodified);
        };
        let Some(relation) = relation else {
            return internal_err!("a fence is only unparsed as a relation");
        };
        let Statement::Query(mut inner) = unparser.plan_to_sql(&fence.input)? else {
            return internal_err!("a fence's input unparses to a query");
        };
        // The unparser emits an empty `LimitClause` for a query without LIMIT or OFFSET.
        let has_limit_or_offset = match &inner.limit_clause {
            None => false,
            Some(ast::LimitClause::LimitOffset { limit, offset, .. }) => {
                limit.is_some() || offset.is_some()
            }
            Some(ast::LimitClause::OffsetCommaLimit { .. }) => true,
        };
        if has_limit_or_offset {
            return internal_err!("a fence's input has no LIMIT or OFFSET of its own");
        }
        inner.limit_clause = Some(ast::LimitClause::LimitOffset {
            limit: None,
            offset: Some(ast::Offset {
                value: ast::Expr::value(ast::Value::Number("0".into(), false)),
                rows: ast::OffsetRows::None,
            }),
            limit_by: vec![],
        });

        let mut derived = DerivedRelationBuilder::default();
        derived.lateral(false).alias(None).subquery(inner);
        relation.derived(derived);
        Ok(UnparseWithinStatementResult::Modified)
    }
}

/// A [`RecordingSQLExecutor`] that supports [`Fence`].
struct FencingExecutor(RecordingSQLExecutor);

#[async_trait]
impl SQLExecutor for FencingExecutor {
    fn name(&self) -> &str {
        self.0.name()
    }

    fn compute_context(&self) -> Option<String> {
        self.0.compute_context()
    }

    fn dialect(&self) -> Arc<dyn Dialect> {
        self.0.dialect()
    }

    fn supports_extension_node(&self, node: &dyn UserDefinedLogicalNode) -> bool {
        node.as_any().is::<Fence>()
    }

    fn extension_unparsers(&self) -> Vec<Arc<dyn UserDefinedLogicalNodeUnparser>> {
        vec![Arc::new(FenceUnparser)]
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

/// Passes its input through. It keeps the default of
/// [`ExecutionPlan::gather_filters_for_pushdown`], so no filter is pushed through it.
#[derive(Debug)]
struct FenceExec {
    input: Arc<dyn ExecutionPlan>,
}

impl DisplayAs for FenceExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "FenceExec")
    }
}

impl ExecutionPlan for FenceExec {
    fn name(&self) -> &str {
        "FenceExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        self.input.properties()
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(FenceExec {
            input: children.swap_remove(0),
        }))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        self.input.execute(partition, context)
    }
}

/// Plans a [`Fence`] as a [`FenceExec`].
#[derive(Debug)]
struct FencePlanner;

#[async_trait]
impl ExtensionPlanner for FencePlanner {
    async fn plan_extension(
        &self,
        _planner: &dyn PhysicalPlanner,
        node: &dyn UserDefinedLogicalNode,
        _logical_inputs: &[&LogicalPlan],
        physical_inputs: &[Arc<dyn ExecutionPlan>],
        _session_state: &SessionState,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        Ok(node.as_any().is::<Fence>().then(|| {
            Arc::new(FenceExec {
                input: Arc::clone(&physical_inputs[0]),
            }) as Arc<dyn ExecutionPlan>
        }))
    }
}

#[derive(Debug)]
struct Planner;

#[async_trait]
impl QueryPlanner for Planner {
    async fn create_physical_plan(
        &self,
        logical_plan: &LogicalPlan,
        session_state: &SessionState,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        DefaultPhysicalPlanner::with_extension_planners(vec![
            Arc::new(FederatedPlanner::new()),
            Arc::new(FencePlanner),
        ])
        .create_physical_plan(logical_plan, session_state)
        .await
    }
}

/// Runs `SELECT * FROM (<fence over test WHERE bar > 1>) AS v WHERE v.foo <> 'c'`
/// against `executor`, returning the optimized plan and the number of rows.
async fn fenced_query(executor: Arc<dyn SQLExecutor>) -> (String, usize) {
    let schema = schema_provider(executor, &["test"]).await;
    let state =
        SessionStateBuilder::new_from_existing(datafusion_federation::default_session_state())
            .with_query_planner(Arc::new(Planner))
            .build();
    overwrite_default_schema(&state, schema);
    let ctx = SessionContext::new_with_state(state);

    let scan = ctx.table("test").await.unwrap().into_unoptimized_plan();
    let fenced = LogicalPlanBuilder::from(scan)
        .filter(col("bar").gt(lit(1)))
        .unwrap()
        .build()
        .unwrap();
    let plan = LogicalPlanBuilder::from(Fence::wrap(fenced))
        .alias("v")
        .unwrap()
        .filter(col("v.foo").not_eq(lit("c")))
        .unwrap()
        .build()
        .unwrap();

    let df = ctx.execute_logical_plan(plan).await.unwrap();
    let optimized = df
        .clone()
        .into_optimized_plan()
        .unwrap()
        .display_indent()
        .to_string();
    let batches = df.collect().await.unwrap();
    (optimized, row_count(&batches))
}

#[tokio::test]
async fn an_extension_the_remote_can_express_is_federated_with_its_input() {
    let ctx = remote_ctx("test", "test.csv").await;
    let executor = FencingExecutor(RecordingSQLExecutor::new("sqlite", "sqlite_exec", ctx));
    let queries = executor.0.queries();

    let (plan, rows) = fenced_query(Arc::new(executor)).await;

    assert_eq!(rows, 1, "only b has bar > 1 and foo <> 'c'");
    assert!(
        plan.starts_with("Federated"),
        "the whole plan is federated:\n{plan}"
    );

    let sql = recorded_sql(&queries);
    let fence = sql
        .find("offset 0")
        .unwrap_or_else(|| panic!("the fence is sent: {sql}"));
    let outer = sql
        .find("<> 'c'")
        .unwrap_or_else(|| panic!("the outer filter is sent: {sql}"));
    assert!(
        fence < outer,
        "the outer filter stays outside the fence: {sql}"
    );
}

#[tokio::test]
async fn an_extension_the_remote_cannot_express_is_a_federation_boundary() {
    let ctx = remote_ctx("test", "test.csv").await;
    let executor = RecordingSQLExecutor::new("sqlite", "sqlite_exec", ctx);
    let queries = executor.queries();

    let (plan, rows) = fenced_query(Arc::new(executor)).await;

    assert_eq!(rows, 1, "only b has bar > 1 and foo <> 'c'");
    let fence = plan
        .find("Fence")
        .unwrap_or_else(|| panic!("the fence is local:\n{plan}"));
    let federated = plan
        .find("Federated")
        .unwrap_or_else(|| panic!("its input is federated:\n{plan}"));
    assert!(
        fence < federated,
        "the fence sits above the federated sub-plan:\n{plan}"
    );

    let sql = recorded_sql(&queries);
    assert!(
        sql.contains("bar > 1"),
        "what is below the fence is still pushed down: {sql}"
    );
    assert!(
        !sql.contains("offset") && !sql.contains("<> 'c'"),
        "nothing above it is: {sql}"
    );
}
