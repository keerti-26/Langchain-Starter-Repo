"""
MCP Agents (Lab 3)
==================
Agents that connect to external MCP servers (GitHub & Linear) to
discover tools dynamically at runtime via Model Context Protocol.

This module now supports a split multi-agent architecture:
- build_github_mcp_agent(): GitHub-only MCP tools (+ Linear bridge tool)
- build_linear_mcp_agent(): Linear-only MCP tools

Prerequisites:
- GITHUB_TOKEN and LINEAR_API_KEY environment variables
- npm packages: @modelcontextprotocol/server-github, @anthropic/linear-mcp-server
"""

import os

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from app.config import get_llm


# ---------------------------------------------------------------------------
# MCP Server Configuration
# ---------------------------------------------------------------------------

MCP_SERVER_GITHUB = {
    "transport": "stdio",
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-github"],
    "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": os.environ.get("GITHUB_TOKEN", ""),
    },
}

LINEAR_MCP_HTTP_URL = os.environ.get("LINEAR_MCP_HTTP_URL", "")

MCP_SERVER_LINEAR = {
    "transport": "streamable_http",
    "url": LINEAR_MCP_HTTP_URL,
    "headers": {
        "Authorization": f"Bearer {os.environ.get('LINEAR_API_KEY', '')}",
    },
}

# ---------------------------------------------------------------------------
# Agent builders
# ---------------------------------------------------------------------------

def _extract_final_text_from_messages(messages: list) -> str:
    """Extract the last AI text response from a LangGraph message list."""
    for msg in reversed(messages):
        if getattr(msg, "type", None) == "ai" and getattr(msg, "content", None):
            return str(msg.content)
    return ""


async def _build_mcp_agent_for_servers(servers: dict, extra_tools: list | None = None, system_prompt="You are a helpful assistant"):
    """
    Connect to MCP servers, discover tools, and create a ReAct agent.

    Args:
        servers: MCP server config dict.
        extra_tools: Optional additional tools to expose to the agent.

    Returns:
        A configured ReAct agent.
    """
    if MultiServerMCPClient is None:
        raise ImportError(
            "MCP agent dependencies are unavailable due to an import/version mismatch. "
            "Install compatible versions of langchain-mcp-adapters and langchain-core."
        ) from _MCP_IMPORT_ERROR

    llm = get_llm(temperature=0)

    client = MultiServerMCPClient(servers)
    tools = await client.get_tools()

    for t in tools:
        print(t.name)
    # filtered_tools = filter(lambda x:x.name!="create_issue",tools)

    if extra_tools:
        tools = [*tools, *extra_tools]

    return create_react_agent(llm, tools, prompt=SystemMessage(content=system_prompt))


@tool
async def ask_linear_agent(query: str) -> str:
    """
    Delegate a task to the Linear MCP agent and return its final answer.

    Use this tool when the agent needs Linear context
    (to create issues, to view projects, view cycles, teams, users, etc).
    """
    linear_agent = await _build_mcp_agent_for_servers({"linear": MCP_SERVER_LINEAR})
    result = await linear_agent.ainvoke(
        {"messages": [query]}
    )
    messages = result.get("messages", [])
    final_text = _extract_final_text_from_messages(messages)
    return final_text or "Linear agent completed but did not return text content."


async def build_github_mcp_agent():
    """Build a GitHub-dedicated MCP agent with a bridge tool to call the Linear agent."""
    return await _build_mcp_agent_for_servers(
        {"github": MCP_SERVER_GITHUB},
        extra_tools=[ask_linear_agent]
        # system_prompt="You are a coding assistant you look at pull requests and issue. When creating issues only use linear"
    )


async def build_linear_mcp_agent():
    """Build a Linear-dedicated MCP agent."""
    return await _build_mcp_agent_for_servers({"linear": MCP_SERVER_LINEAR})
