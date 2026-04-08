"""Lab agents migrated from langchain-day-1-lab."""

from app.lab_agents.simple_agent import build_simple_agent
from app.lab_agents.tool_agent import build_tool_agent
from app.lab_agents.middleware_agent import build_middleware_agent, ProfanityFilterMiddleware, TopicFilterMiddleware
from app.lab_agents.memory_agent import build_memory_agent
from app.lab_agents.postgres_airflow_agent import build_postgres_airflow_agent
from app.lab_agents.deep_research_agent import build_deep_research_agent

__all__ = [
    "build_simple_agent",
    "build_tool_agent",
    "build_middleware_agent",
    "ProfanityFilterMiddleware",
    "build_memory_agent",
    "build_postgres_airflow_agent",
    "build_deep_research_agent",
]
