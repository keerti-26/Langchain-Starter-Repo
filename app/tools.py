"""Custom tools for the agent."""

import math
from datetime import datetime

from langchain_core.tools import tool


@tool
def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression. Supports basic arithmetic and math functions.

    Args:
        expression: A math expression like '2 + 2', 'sqrt(16)', 'sin(3.14)'
    """
    allowed_names = {
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "pi": math.pi,
        "e": math.e,
        "abs": abs,
        "round": round,
    }
    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"


@tool
def search_knowledge_base(query: str) -> str:
    """Search a mock knowledge base for information.

    Args:
        query: The search query string
    """
    # Mock knowledge base — swap this for a real vector store / retriever
    knowledge = {
        "refund policy": "Refunds are available within 30 days of purchase with a valid receipt.",
        "business hours": "We are open Monday through Friday, 9 AM to 5 PM EST.",
        "shipping": "Standard shipping takes 5-7 business days. Express shipping takes 1-2 business days.",
        "contact": "You can reach us at support@example.com or call 1-800-555-0199.",
    }
    query_lower = query.lower()
    for key, value in knowledge.items():
        if key in query_lower:
            return value
    return "No relevant information found. Please try a different query."
