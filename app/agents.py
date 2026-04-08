"""Agent definitions — tool agent and memory agent."""

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from app.config import get_llm
from app.tools import calculator, get_current_time, search_knowledge_base

# ---------------------------------------------------------------------------
# 1) Tool Agent — stateless, has access to tools
# ---------------------------------------------------------------------------

TOOL_AGENT_SYSTEM = """You are a helpful assistant with access to tools.
Use them whenever they can help answer the user's question.
Always prefer using a tool over guessing."""


def build_tool_agent():
    """Create a ReAct agent with tools (no persistent memory)."""
    llm = get_llm()
    tools = [calculator, get_current_time, search_knowledge_base]
    agent = create_react_agent(
        llm,
        tools,
        prompt=SystemMessage(content=TOOL_AGENT_SYSTEM)
    )
    return agent


# ---------------------------------------------------------------------------
# 2) Memory Agent — conversational memory across turns
# ---------------------------------------------------------------------------

MEMORY_AGENT_SYSTEM = """You are a friendly conversational assistant.
You remember everything the user has told you in this session.
Refer back to earlier parts of the conversation when relevant."""


def build_memory_agent():
    """Create a ReAct agent backed by an in-memory checkpointer."""
    llm = get_llm()
    checkpointer = MemorySaver()
    agent = create_react_agent(
        llm,
        tools=[],  # pure conversation, no tools
        prompt=SystemMessage(content=MEMORY_AGENT_SYSTEM),
        checkpointer=checkpointer
    )
    return agent
