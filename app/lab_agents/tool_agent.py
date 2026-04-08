"""
Tool Agent (Lab 2)
===================
A ReAct agent with custom tools — calculator, weather, and string reversal.
"""

from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from app.config import get_llm


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression and return the result. Use Python syntax.
    Examples: '2 + 2', '(10 * 5) / 3', '2 ** 10'
    """
    try:
        allowed_names = {"__builtins__": {}}
        result = eval(expression, allowed_names)
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"


@tool
def get_current_weather(city: str) -> str:
    """Get the current weather for a given city. Returns temperature and conditions."""
    fake_weather = {
        "new york": "72°F, Partly Cloudy",
        "london": "58°F, Rainy",
        "tokyo": "68°F, Sunny",
        "paris": "63°F, Overcast",
    }
    return fake_weather.get(
        city.lower(),
        f"Weather data not available for '{city}'. Try: New York, London, Tokyo, or Paris.",
    )


@tool
def reverse_string(text: str) -> str:
    """Reverse the characters in a string. Useful for puzzles or encoding."""
    return text[::-1]


# ---------------------------------------------------------------------------
# Agent builder
# ---------------------------------------------------------------------------

TOOL_AGENT_SYSTEM = """You are a helpful assistant with access to tools.
Use them whenever they can help answer the user's question.
Always prefer using a tool over guessing."""


def build_tool_agent():
    """Create a ReAct agent with calculator, weather, and string tools."""
    llm = get_llm(temperature=0)
    tools = [calculator, get_current_weather, reverse_string]
    agent = create_react_agent(
        llm,
        tools,
        prompt=SystemMessage(content=TOOL_AGENT_SYSTEM),
    )
    return agent
