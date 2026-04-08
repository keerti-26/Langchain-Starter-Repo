"""Request/response schemas for the API."""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., description="The user's message")
    session_id: str | None = Field(
        default=None,
        description="Session ID for memory agent (enables multi-turn conversations)",
    )


class AgentStep(BaseModel):
    """A single intermediate step from the agent's reasoning process."""
    type: str = Field(..., description="Step type: 'ai', 'tool_call', 'tool_result'")
    name: str | None = Field(default=None, description="Tool name (for tool_call/tool_result)")
    content: str = Field(..., description="Step content or tool output")
    tool_input: str | None = Field(default=None, description="Tool input arguments (for tool_call)")


class ChatResponse(BaseModel):
    response: str = Field(..., description="The agent's final reply")
    session_id: str | None = Field(
        default=None,
        description="Session ID (returned by memory endpoint)",
    )
    steps: list[AgentStep] = Field(
        default_factory=list,
        description="Intermediate reasoning steps (tool calls, results, thinking)",
    )
