"""
Simple Agent (Lab 1)
====================
The most basic agent — just an LLM with a system prompt.
No tools, no memory, no middleware. Just a straight conversation.
"""

from langchain_core.messages import SystemMessage, HumanMessage

from app.config import get_llm


SIMPLE_AGENT_SYSTEM = "You are an helpful assistant. Be very mean"


def build_simple_agent(system_prompt: str = SIMPLE_AGENT_SYSTEM):
    """Create the simplest possible agent — just an LLM with a persona."""
    llm = get_llm()
    return llm, system_prompt


def chat(agent_tuple, user_message: str, system_prompt_override: str | None = None):
    """Send a single message to the simple agent and get a response."""
    llm, default_prompt = agent_tuple
    prompt = system_prompt_override or default_prompt
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=user_message),
    ]
    response = llm.invoke(messages)
    return response.content
