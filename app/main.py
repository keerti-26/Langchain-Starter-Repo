"""FastAPI application with LangChain agent endpoints."""

import json
import uuid

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage

from app.agents import build_tool_agent, build_memory_agent
from app.schemas import ChatRequest, ChatResponse, AgentStep
from app.lab_agents import (
    build_simple_agent,
    build_middleware_agent,
    build_postgres_airflow_agent,
    build_deep_research_agent,
)
from app.lab_agents.simple_agent import chat as simple_chat
from app.lab_agents.tool_agent import build_tool_agent as build_lab_tool_agent
from app.lab_agents.memory_agent import build_memory_agent as build_lab_memory_agent

app = FastAPI(
    title="LangChain Starter Agent",
    description="A starter API with tool-based and memory-based agent endpoints.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Build agents at startup
# ---------------------------------------------------------------------------

# Original starter agents
tool_agent = build_tool_agent()
memory_agent = build_memory_agent()

# Lab agents
simple_agent = build_simple_agent()
lab_tool_agent = build_lab_tool_agent()
middleware_agent, profanity_filter, topic_filter = build_middleware_agent(mode="censor")
lab_memory_agent = build_lab_memory_agent()
postgres_airflow_agent = build_postgres_airflow_agent()
deep_research_agent = build_deep_research_agent()

# Map of agent mode → agent instance (for streaming endpoint)
AGENT_MAP = {
    "tools": tool_agent,
    "memory": memory_agent,
    "lab-tools": lab_tool_agent,
    "middleware": middleware_agent,
    "lab-memory": lab_memory_agent,
    "postgres-airflow": postgres_airflow_agent,
    "deep-research": deep_research_agent,
}

STATEFUL_AGENTS = {"memory", "lab-memory"}


# ---------------------------------------------------------------------------
# Helper: extract intermediate steps from agent messages (non-streaming)
# ---------------------------------------------------------------------------

def extract_steps(messages: list) -> tuple[str, list[AgentStep]]:
    """
    Parse LangGraph message list into a final response + intermediate steps.
    """
    steps = []
    final_response = ""

    for msg in messages:
        msg_type = getattr(msg, "type", None)

        if msg_type == "human":
            continue

        if msg_type == "ai":
            tool_calls = getattr(msg, "tool_calls", [])
            if tool_calls:
                for tc in tool_calls:
                    args_str = json.dumps(tc.get("args", {}), indent=2)
                    steps.append(AgentStep(
                        type="tool_call",
                        name=tc.get("name", "unknown"),
                        content=f"Calling tool: {tc.get('name', 'unknown')}",
                        tool_input=args_str,
                    ))
            if msg.content and tool_calls:
                steps.append(AgentStep(
                    type="ai",
                    content=msg.content,
                ))
            if msg.content and not tool_calls:
                final_response = msg.content

        elif msg_type == "tool":
            content = msg.content if isinstance(msg.content, str) else json.dumps(msg.content)
            steps.append(AgentStep(
                type="tool_result",
                name=getattr(msg, "name", None),
                content=content,
            ))

    return final_response, steps


# ---------------------------------------------------------------------------
# Helper: SSE formatting
# ---------------------------------------------------------------------------

def sse_event(event: str, data: dict) -> str:
    """Format a Server-Sent Event."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# ---------------------------------------------------------------------------
# Streaming generator using astream_events
# ---------------------------------------------------------------------------

async def stream_agent_events(agent, messages: dict, config: dict | None = None):
    """
    Stream LangGraph agent events as SSE.

    Yields SSE events:
      - event: step  → intermediate tool calls, results, thinking
      - event: done  → final response + all accumulated steps
    """
    steps = []
    final_response = ""
    session_id = None

    if config:
        session_id = config.get("configurable", {}).get("thread_id")

    try:
        async for event in agent.astream_events(
            messages,
            config=config or {},
            version="v2",
        ):
            kind = event.get("event", "")
            name = event.get("name", "")
            data = event.get("data", {})

            # Tool call started
            if kind == "on_tool_start":
                tool_input = data.get("input", {})
                if isinstance(tool_input, dict):
                    input_str = json.dumps(tool_input, indent=2)
                else:
                    input_str = str(tool_input)

                step = AgentStep(
                    type="tool_call",
                    name=name,
                    content=f"Calling tool: {name}",
                    tool_input=input_str,
                )
                steps.append(step)
                yield sse_event("step", step.model_dump())

            # Tool call completed
            elif kind == "on_tool_end":
                output = data.get("output", "")
                if hasattr(output, "content"):
                    output = output.content
                if not isinstance(output, str):
                    output = json.dumps(output)

                step = AgentStep(
                    type="tool_result",
                    name=name,
                    content=output,
                )
                steps.append(step)
                yield sse_event("step", step.model_dump())

            # Chat model streaming — capture intermediate AI thinking
            elif kind == "on_chat_model_end":
                output = data.get("output", None)
                if output:
                    msg = output
                    # If output is a message object
                    if hasattr(msg, "content") and hasattr(msg, "tool_calls"):
                        # Intermediate AI thinking (has content + tool calls)
                        if msg.content and getattr(msg, "tool_calls", []):
                            step = AgentStep(
                                type="ai",
                                content=msg.content,
                            )
                            steps.append(step)
                            yield sse_event("step", step.model_dump())
                        # Final response (has content, no tool calls)
                        elif msg.content and not getattr(msg, "tool_calls", []):
                            final_response = msg.content

    except Exception as e:
        yield sse_event("error", {"message": str(e)})
        return

    # Send the final done event with all accumulated data
    done_data = {
        "response": final_response,
        "steps": [s.model_dump() for s in steps],
    }
    if session_id:
        done_data["session_id"] = session_id
    yield sse_event("done", done_data)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    """Health check."""
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Streaming endpoint — handles all agents via mode parameter
# ---------------------------------------------------------------------------

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest, mode: str = "tools"):
    """
    Stream agent thinking steps as Server-Sent Events.

    Modes: tools, memory, simple, lab-tools, mcp-github, mcp-linear, middleware, lab-memory, postgres-airflow, deep-research
    """
    # --- Simple agent (no streaming — just return SSE with result) ---
    if mode == "simple":
        async def simple_stream():
            response = simple_chat(simple_agent, req.message)
            yield sse_event("done", {"response": response, "steps": []})

        return StreamingResponse(simple_stream(), media_type="text/event-stream")

    # --- MCP agents (special lifecycle) ---
    if mode in {"mcp-github", "mcp-linear"}:
        from app.lab_agents.mcp_agent import (
            build_github_mcp_agent,
            build_linear_mcp_agent,
        )

        builder_by_mode = {
            "mcp-github": build_github_mcp_agent,
            "mcp-linear": build_linear_mcp_agent,
        }

        async def mcp_stream():
            agent = await builder_by_mode[mode]()
            async for chunk in stream_agent_events(
                agent,
                {"messages": [HumanMessage(content=req.message)]},
            ):
                yield chunk

        return StreamingResponse(mcp_stream(), media_type="text/event-stream")

    # --- Middleware agent (profanity + topic filter + streaming) ---
    if mode == "middleware":
        async def middleware_stream():
            all_steps = []

            # Step 1: Profanity filter
            should_proceed, message_to_use, filter_result = profanity_filter.process(req.message)

            profanity_step = AgentStep(
                type="ai" if not filter_result["flagged"] else "tool_call",
                name="ProfanityFilter",
                content=(
                    f"🚫 Profanity detected: {', '.join(filter_result['matched_words'])}. "
                    f"Mode: {profanity_filter.mode}. "
                    + (f"Censored: \"{filter_result['censored_message']}\"" if should_proceed else "Message blocked.")
                ) if filter_result["flagged"] else "✅ Message passed profanity filter.",
            )
            all_steps.append(profanity_step)
            yield sse_event("step", profanity_step.model_dump())

            if not should_proceed:
                yield sse_event("done", {
                    "response": "Please be kinder 💛",
                    "steps": [s.model_dump() for s in all_steps],
                })
                return

            # Step 2: Topic filter
            on_topic, topic_result = topic_filter.process(message_to_use)

            topic_step = AgentStep(
                type="ai" if on_topic else "tool_call",
                name="TopicFilter",
                content="✅ Message is on-topic (AI engineering)." if on_topic
                    else "🍔 Off-topic message detected. This agent only answers AI engineering questions.",
            )
            all_steps.append(topic_step)
            yield sse_event("step", topic_step.model_dump())

            if not on_topic:
                from app.lab_agents.middleware_agent import OFF_TOPIC_RESPONSE
                yield sse_event("done", {
                    "response": OFF_TOPIC_RESPONSE,
                    "steps": [s.model_dump() for s in all_steps],
                })
                return

            # Step 3: Stream the agent response
            async for chunk in stream_agent_events(
                middleware_agent,
                {"messages": [HumanMessage(content=message_to_use)]},
            ):
                yield chunk

        return StreamingResponse(middleware_stream(), media_type="text/event-stream")

    # --- Standard agents (tool, memory, lab-tools, lab-memory, postgres-airflow) ---
    agent = AGENT_MAP.get(mode)
    if not agent:
        async def error_stream():
            yield sse_event("error", {"message": f"Unknown agent mode: {mode}"})
        return StreamingResponse(error_stream(), media_type="text/event-stream")

    config = None
    if mode in STATEFUL_AGENTS:
        session_id = req.session_id or str(uuid.uuid4())
        config = {"configurable": {"thread_id": session_id}}

    async def agent_stream():
        async for chunk in stream_agent_events(
            agent,
            {"messages": [HumanMessage(content=req.message)]},
            config=config,
        ):
            yield chunk

    return StreamingResponse(agent_stream(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Original non-streaming endpoints (kept for backwards compatibility)
# ---------------------------------------------------------------------------

@app.post("/chat/tools", response_model=ChatResponse)
async def chat_with_tools(req: ChatRequest):
    """Chat with the tool agent (stateless)."""
    result = tool_agent.invoke(
        {"messages": [HumanMessage(content=req.message)]}
    )
    final, steps = extract_steps(result["messages"])
    return ChatResponse(response=final, steps=steps)


@app.post("/chat/memory", response_model=ChatResponse)
async def chat_with_memory(req: ChatRequest):
    """Chat with the memory agent (stateful)."""
    session_id = req.session_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}
    result = memory_agent.invoke(
        {"messages": [HumanMessage(content=req.message)]},
        config=config,
    )
    final, steps = extract_steps(result["messages"])
    return ChatResponse(response=final, session_id=session_id, steps=steps)


@app.post("/chat/simple", response_model=ChatResponse)
async def chat_simple(req: ChatRequest):
    """Chat with the simple agent (Lab 1)."""
    response = simple_chat(simple_agent, req.message)
    return ChatResponse(response=response)


@app.post("/chat/lab-tools", response_model=ChatResponse)
async def chat_lab_tools(req: ChatRequest):
    """Chat with the lab tool agent (Lab 2)."""
    result = lab_tool_agent.invoke(
        {"messages": [HumanMessage(content=req.message)]}
    )
    final, steps = extract_steps(result["messages"])
    return ChatResponse(response=final, steps=steps)


@app.post("/chat/deep-research", response_model=ChatResponse)
async def chat_deep_research(req: ChatRequest):
    """Chat with the deep research agent (Lab 7)."""
    result = deep_research_agent.invoke(
        {"messages": [HumanMessage(content=req.message)]}
    )
    final, steps = extract_steps(result["messages"])
    return ChatResponse(response=final, steps=steps)


async def _chat_with_mcp_builder(req: ChatRequest, builder):
    """Shared MCP chat lifecycle helper."""
    agent = await builder()
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content=req.message)]}
    )
    final, steps = extract_steps(result["messages"])
    return ChatResponse(response=final, steps=steps)



@app.post("/chat/mcp-github", response_model=ChatResponse)
async def chat_mcp_github(req: ChatRequest):
    """Chat with the GitHub MCP agent (Lab 3 split architecture)."""
    from app.lab_agents.mcp_agent import build_github_mcp_agent
    return await _chat_with_mcp_builder(req, build_github_mcp_agent)


@app.post("/chat/mcp-linear", response_model=ChatResponse)
async def chat_mcp_linear(req: ChatRequest):
    """Chat with the Linear MCP agent (Lab 3 split architecture)."""
    from app.lab_agents.mcp_agent import build_linear_mcp_agent
    return await _chat_with_mcp_builder(req, build_linear_mcp_agent)


@app.post("/chat/middleware", response_model=ChatResponse)
async def chat_middleware(req: ChatRequest):
    """Chat with the middleware agent (Lab 4). Profanity + topic filter applied."""
    all_steps = []

    # Step 1: Profanity filter
    should_proceed, message_to_use, filter_result = profanity_filter.process(req.message)

    profanity_step = AgentStep(
        type="ai" if not filter_result["flagged"] else "tool_call",
        name="ProfanityFilter",
        content=(
            f"Profanity detected: {', '.join(filter_result['matched_words'])}. "
            f"Mode: {profanity_filter.mode}. "
            + (f"Censored: \"{filter_result['censored_message']}\"" if should_proceed else "Message blocked.")
        ) if filter_result["flagged"] else "Message passed profanity filter.",
    )
    all_steps.append(profanity_step)

    if not should_proceed:
        return ChatResponse(
            response="Please be kinder 💛",
            steps=all_steps,
        )

    # Step 2: Topic filter
    on_topic, topic_result = topic_filter.process(message_to_use)
    topic_step = AgentStep(
        type="ai" if on_topic else "tool_call",
        name="TopicFilter",
        content="Message is on-topic (AI engineering)." if on_topic
            else "Off-topic message detected. This agent only answers AI engineering questions.",
    )
    all_steps.append(topic_step)

    if not on_topic:
        from app.lab_agents.middleware_agent import OFF_TOPIC_RESPONSE
        return ChatResponse(response=OFF_TOPIC_RESPONSE, steps=all_steps)

    # Step 3: Agent
    result = middleware_agent.invoke(
        {"messages": [HumanMessage(content=message_to_use)]}
    )
    final, steps = extract_steps(result["messages"])
    return ChatResponse(response=final, steps=all_steps + steps)


@app.post("/chat/lab-memory", response_model=ChatResponse)
async def chat_lab_memory(req: ChatRequest):
    """Chat with the lab memory agent (Lab 5)."""
    session_id = req.session_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}
    result = lab_memory_agent.invoke(
        {"messages": [HumanMessage(content=req.message)]},
        config=config,
    )
    final, steps = extract_steps(result["messages"])
    return ChatResponse(response=final, session_id=session_id, steps=steps)


@app.post("/chat/postgres-airflow", response_model=ChatResponse)
async def chat_postgres_airflow(req: ChatRequest):
    """Chat with the Postgres Airflow agent (Lab 6)."""
    result = postgres_airflow_agent.invoke(
        {"messages": [HumanMessage(content=req.message)]}
    )
    final, steps = extract_steps(result["messages"])
    return ChatResponse(response=final, steps=steps)
