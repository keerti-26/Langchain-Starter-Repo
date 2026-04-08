"""
Postgres → Airflow DAG Generator Agent (Lab 6)
================================================
An agent that connects to a PostgreSQL database, inspects its schema,
and generates Apache Airflow DAGs based on discovered tables and relationships.

Prerequisites:
    pip install psycopg2-binary sqlalchemy
    export DATABASE_URL="postgresql://user:password@host:port/dbname"
"""

import os
import json
import textwrap
from datetime import datetime

from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from sqlalchemy import create_engine, inspect, text

from app.config import get_llm


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

## TODO MAKE SURE YOU UPDATE THIS WITH THE URL ZACH SENDS YOU
DATABASE_URL = os.environ.get(
    "DATABASE_URL"
)

if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is not set. Remember to set it!")

engine = create_engine(DATABASE_URL)


# ---------------------------------------------------------------------------
# Tools — Database Introspection
# ---------------------------------------------------------------------------

@tool
def list_tables(schemas=[]) -> str:
    """List all user tables in the connected PostgreSQL database.

    Returns a JSON array of objects with schema, table name, and
    approximate row count for each table.
    """
    inspector = inspect(engine)
    tables = []
    for schema in inspector.get_schema_names():
        if schema not in ("information_schema", "pg_catalog", "pg_toast") and len(schemas) == 0:
            continue
        elif len(schemas) > 0 and schema not in schemas:
            continue
        for table_name in inspector.get_table_names(schema=schema):
            with engine.connect() as conn:
                result = conn.execute(
                    text(
                        "SELECT n_live_tup FROM pg_stat_user_tables "
                        "WHERE schemaname = :schema AND relname = :table"
                    ),
                    {"schema": schema, "table": table_name},
                )
                row = result.fetchone()
                row_count = row[0] if row else "unknown"

            tables.append({
                "schema": schema,
                "table": table_name,
                "approx_rows": row_count,
            })

    if not tables:
        return "No tables found in the database. Please try other schemas"
    return json.dumps(tables, indent=2)


@tool
def get_table_schema(table_name: str) -> str:
    """Get the detailed schema (columns, types, nullable, primary keys,
    foreign keys) for a specific table.

    Args:
        table_name: Name of the table to inspect. Use 'schema.table' format
                    if not in the public schema.
    """
    schema = "public"
    if "." in table_name:
        schema, table_name = table_name.split(".", 1)

    inspector = inspect(engine)

    columns = inspector.get_columns(table_name, schema=schema)
    col_info = [
        {
            "name": c["name"],
            "type": str(c["type"]),
            "nullable": c.get("nullable", True),
            "default": str(c.get("default", "")) or None,
        }
        for c in columns
    ]

    pk = inspector.get_pk_constraint(table_name, schema=schema)
    pk_columns = pk.get("constrained_columns", []) if pk else []

    fks = inspector.get_foreign_keys(table_name, schema=schema)
    fk_info = [
        {
            "columns": fk["constrained_columns"],
            "references": f"{fk.get('referred_schema', 'public')}.{fk['referred_table']}({', '.join(fk['referred_columns'])})",
        }
        for fk in fks
    ]

    indexes = inspector.get_indexes(table_name, schema=schema)
    idx_info = [
        {"name": idx["name"], "columns": idx["column_names"], "unique": idx.get("unique", False)}
        for idx in indexes
    ]

    result = {
        "table": f"{schema}.{table_name}",
        "columns": col_info,
        "primary_key": pk_columns,
        "foreign_keys": fk_info,
        "indexes": idx_info,
    }
    return json.dumps(result, indent=2)


@tool
def get_table_relationships(schemas = []) -> str:
    """Discover all foreign key relationships across the database.

    Returns a JSON array describing each foreign-key relationship,
    useful for determining table dependencies and DAG ordering.
    """
    inspector = inspect(engine)
    relationships = []

    for schema in inspector.get_schema_names():
        if schema not in schemas:
            continue
        for table_name in inspector.get_table_names(schema=schema):
            fks = inspector.get_foreign_keys(table_name, schema=schema)
            for fk in fks:
                relationships.append({
                    "from_table": f"{schema}.{table_name}",
                    "from_columns": fk["constrained_columns"],
                    "to_table": f"{fk.get('referred_schema', 'public')}.{fk['referred_table']}",
                    "to_columns": fk["referred_columns"],
                })

    if not relationships:
        return "No foreign key relationships found."
    return json.dumps(relationships, indent=2)


@tool
def sample_table_data(table_name: str, limit: int = 5) -> str:
    """Retrieve a small sample of rows from a table to understand data patterns.

    Args:
        table_name: Name of the table (use 'schema.table' for non-public schemas).
        limit: Number of rows to sample (default 5, max 20).
    """
    schema = "public"
    if "." in table_name:
        schema, table_name = table_name.split(".", 1)

    limit = min(limit, 20)

    with engine.connect() as conn:
        result = conn.execute(
            text(f'SELECT * FROM "{schema}"."{table_name}" LIMIT :limit'),
            {"limit": limit},
        )
        columns = list(result.keys())
        rows = [dict(zip(columns, row)) for row in result.fetchall()]

    if not rows:
        return f"Table {schema}.{table_name} is empty."

    for row in rows:
        for k, v in row.items():
            if isinstance(v, (datetime,)):
                row[k] = v.isoformat()
            elif not isinstance(v, (str, int, float, bool, type(None))):
                row[k] = str(v)

    return json.dumps({"columns": columns, "sample_rows": rows}, indent=2)


@tool
def generate_airflow_dag(dag_spec: str) -> str:
    """Generate and save an Airflow DAG Python file based on a specification.

    Args:
        dag_spec: The complete Python code for the Airflow DAG file.
                  Must be valid Python that Airflow can parse.
    """
    output_path = "generated_dag.py"
    with open(output_path, "w") as f:
        f.write(dag_spec)
    return f"DAG file saved to: {output_path}"


# ---------------------------------------------------------------------------
# Agent builder
# ---------------------------------------------------------------------------

POSTGRES_AGENT_SYSTEM = textwrap.dedent("""\
    You are a data engineering assistant that specializes in building
    Apache Airflow DAGs from database schemas. Only look at tables in the bootcamp schema

    Your workflow:
    1. First, use `list_tables` to discover all tables in the database.
    2. Use `get_table_relationships` to understand foreign key dependencies.
    3. Use `get_table_schema` on key tables to understand their structure.
    4. Optionally use `sample_table_data` to peek at data patterns.
    5. Finally, use `generate_airflow_dag` to produce a production-ready
       Airflow DAG that:
       - Respects foreign key dependencies for task ordering
       - Uses appropriate Airflow operators (e.g., PostgresOperator or
         PythonOperator with SQLAlchemy)
       - Includes proper DAG metadata (description, schedule, tags)
       - Generates new DDLs based on the JOINs and aggregation of the tables
       - Follows Airflow best practices (idempotent tasks, catchup=False)
       
    Always inspect the database thoroughly before generating the DAG.
    Explain your reasoning as you go.
""")


def build_postgres_airflow_agent():
    """Create a ReAct agent for Postgres introspection and Airflow DAG generation."""
    llm = get_llm(model="gpt-4o", temperature=0)
    tools = [
        list_tables,
        get_table_schema,
        get_table_relationships,
        sample_table_data,
        generate_airflow_dag,
    ]
    agent = create_react_agent(
        llm,
        tools,
        prompt=SystemMessage(content=POSTGRES_AGENT_SYSTEM),
    )
    return agent
