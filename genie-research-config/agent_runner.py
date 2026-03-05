"""
Full multi-agent pipeline extracted from Genie_deepresearch.ipynb.

Builds the LangGraph supervisor → Genie / ParallelExecutor → final_answer
graph and exposes a single `run_query(query)` function.
"""

import asyncio
import functools
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from zoneinfo import ZoneInfo

import yaml
from pydantic import BaseModel
from typing_extensions import Annotated, TypedDict

_APP_DIR = Path(__file__).resolve().parent
_local_config = _APP_DIR / "configs.yaml"
_parent_config = _APP_DIR.parent / "configs.yaml"
CONFIG_PATH = _local_config if _local_config.exists() else _parent_config

_local_env = _APP_DIR / ".env"
_parent_env = _APP_DIR.parent / ".env"
ENV_PATH = _local_env if _local_env.exists() else _parent_env


def _load_env():
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip()
                if k not in os.environ:
                    os.environ[k] = v


def _load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Lazy initialization: heavy imports and agent construction only on first use
# ---------------------------------------------------------------------------

_initialized = False
_init_error = None

# Module-level references populated by _ensure_initialized()
genie_agent = None
llm = None
multi_agent = None
genie_cache = None

LLM_ENDPOINT_NAME = None
LLM_TEMPERATURE = None
GENIE_SPACE_ID = None
GENIE_DESCRIPTION = None
MAX_ITERATIONS = None
SYSTEM_PROMPT = None
RESEARCH_PROMPT = None
FINAL_ANSWER_PROMPT = None
QUALITY_CHECK_PROMPT = None
POST_WORKER_ROUTING_PROMPT = None
MAX_CONVERSATION_MESSAGES = None


def _ensure_initialized():
    """Import heavy dependencies and build the graph on first call."""
    global _initialized, _init_error
    global genie_agent, llm, multi_agent, genie_cache
    global LLM_ENDPOINT_NAME, LLM_TEMPERATURE, GENIE_SPACE_ID, GENIE_DESCRIPTION
    global MAX_ITERATIONS, SYSTEM_PROMPT, RESEARCH_PROMPT, FINAL_ANSWER_PROMPT
    global QUALITY_CHECK_PROMPT, POST_WORKER_ROUTING_PROMPT, MAX_CONVERSATION_MESSAGES

    if _initialized:
        return
    if _init_error:
        raise RuntimeError(_init_error)

    try:
        _load_env()

        import mlflow
        from databricks.sdk import WorkspaceClient
        from databricks_langchain import ChatDatabricks
        from databricks_langchain.genie import GenieAgent
        from langchain_core.runnables import RunnableLambda
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.graph import END, StateGraph, add_messages

        mlflow.set_tracking_uri("databricks")

        cfg = _load_config()
        ac = cfg["agent_configs"]

        LLM_ENDPOINT_NAME = ac["llm"]["endpoint_name"]
        LLM_TEMPERATURE = ac["llm"]["temperature"]
        GENIE_SPACE_ID = ac["genie_agent"]["space_id"]
        GENIE_DESCRIPTION = ac["genie_agent"]["description"]
        MAX_ITERATIONS = ac["supervisor_agent"]["max_iterations"]
        SYSTEM_PROMPT = ac["supervisor_agent"]["system_prompt"]
        RESEARCH_PROMPT = ac["supervisor_agent"]["research_prompt"]
        FINAL_ANSWER_PROMPT = ac["supervisor_agent"]["final_answer_prompt"]
        QUALITY_CHECK_PROMPT = ac["supervisor_agent"].get("quality_check_prompt", "")
        POST_WORKER_ROUTING_PROMPT = ac["supervisor_agent"].get("post_worker_routing_prompt", "")
        MAX_CONVERSATION_MESSAGES = ac.get("conversation", {}).get("max_messages", 7)

        genie_cache = QueryCache(ttl_seconds=300)

        _db_host = os.environ.get("DATABRICKS_HOST", os.environ.get("DB_MODEL_SERVING_HOST_URL", ""))
        _db_token = os.environ.get("DATABRICKS_TOKEN", os.environ.get("DATABRICKS_GENIE_PAT", ""))
        ws_kwargs = {}
        if _db_host:
            ws_kwargs["host"] = _db_host
        if _db_token:
            ws_kwargs["token"] = _db_token

        genie_agent = GenieAgent(
            genie_space_id=GENIE_SPACE_ID,
            genie_agent_name="Genie",
            description=GENIE_DESCRIPTION,
            client=WorkspaceClient(**ws_kwargs),
        )
        llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME, temperature=LLM_TEMPERATURE)

        _genie_node = functools.partial(agent_node, agent=genie_agent, name="Genie")

        wf = StateGraph(AgentState)
        wf.add_node("Genie", _genie_node)
        wf.add_node("ParallelExecutor", research_planner_node)
        wf.add_node("supervisor", supervisor_agent)
        wf.add_node("final_answer", final_answer)
        wf.set_entry_point("supervisor")
        for worker in ["Genie", "ParallelExecutor"]:
            wf.add_edge(worker, "supervisor")
        wf.add_conditional_edges(
            "supervisor",
            lambda x: x["next_node"],
            {**{k: k for k in ["Genie", "ParallelExecutor"]}, "FINISH": "final_answer"},
        )
        wf.add_edge("final_answer", END)
        multi_agent = wf.compile(checkpointer=MemorySaver())

        _initialized = True
    except Exception as e:
        _init_error = str(e)
        raise


# ---------------------------------------------------------------------------
# Query cache
# ---------------------------------------------------------------------------

class QueryCache:
    def __init__(self, ttl_seconds: int = 300):
        self._cache: Dict[str, tuple] = {}
        self._ttl = ttl_seconds

    def get(self, query: str) -> Optional[str]:
        key = query.strip().lower()
        if key in self._cache:
            value, ts = self._cache[key]
            if time.time() - ts < self._ttl:
                return value
            del self._cache[key]
        return None

    def set(self, query: str, response: str):
        self._cache[query.strip().lower()] = (response, time.time())


# ---------------------------------------------------------------------------
# Temporal context
# ---------------------------------------------------------------------------

def get_temporal_context() -> Dict[str, str]:
    now = datetime.now(ZoneInfo("America/New_York"))
    today_iso = now.date().isoformat()
    fy_end_year = now.year + 1 if now.month >= 9 else now.year
    fy = f"FY{fy_end_year}"
    if now.month in (9, 10, 11):
        fq = "Q1"
    elif now.month in (12, 1, 2):
        fq = "Q2"
    elif now.month in (3, 4, 5):
        fq = "Q3"
    else:
        fq = "Q4"
    return {"today_iso": today_iso, "fy": fy, "fq": fq}


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

options = ["FINISH", "Genie", "ParallelExecutor"]
FINISH = {"next_node": "FINISH"}


class ResearchPlan(BaseModel):
    queries: List[str]
    rationale: str


class ResearchPlanOutput(BaseModel):
    should_plan_research: bool
    research_plan: Optional[ResearchPlan] = None
    next_node: Literal[tuple(options)]
    refined_query: Optional[str] = None
    confidence: float = 1.0


class QualityCheckOutput(BaseModel):
    passes_quality_check: bool
    reason: Optional[str] = None


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

from langgraph.graph import add_messages as _add_messages


class AgentState(TypedDict):
    messages: Annotated[list, _add_messages]
    next_node: str
    iteration_count: int
    research_plan: Optional[Dict[str, Any]]
    research_results: Optional[List[Dict[str, Any]]]
    refined_query: Optional[str]
    same_node_count: int


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

_log_lines: list = []


def _log(msg: str):
    print(msg)
    _log_lines.append(msg)


def supervisor_agent(state):
    from langchain_core.runnables import RunnableLambda

    try:
        count = state.get("iteration_count", 0) + 1
        if count > MAX_ITERATIONS:
            return FINISH

        temporal_ctx = get_temporal_context()
        temporal_prefix = (
            "Below is information on the current date and fiscal year/quarter information. "
            "You may or may not use this in your analysis.\n"
            f"- The current date is: {temporal_ctx['today_iso']}\n"
            f"- The current fiscal year is: {temporal_ctx['fy']}\n"
            f"- The current fiscal quarter is: {temporal_ctx['fq']}\n\n"
        )

        has_worker_response = any(
            (isinstance(m, dict) and m.get("name") in ("Genie", "ParallelExecutor"))
            or (hasattr(m, "name") and getattr(m, "name", None) in ("Genie", "ParallelExecutor"))
            for m in state.get("messages", [])
        )

        if has_worker_response and count > 1 and POST_WORKER_ROUTING_PROMPT:
            system_content = temporal_prefix + SYSTEM_PROMPT + "\n\n" + POST_WORKER_ROUTING_PROMPT
        else:
            system_content = temporal_prefix + SYSTEM_PROMPT + "\n\n" + RESEARCH_PROMPT

        def _build_supervisor_messages(state):
            msgs = [{"role": "system", "content": system_content}]
            for m in state["messages"]:
                if isinstance(m, dict):
                    msgs.append(m)
                else:
                    msg_dict = {"role": getattr(m, "role", "assistant"), "content": getattr(m, "content", str(m))}
                    if hasattr(m, "name") and m.name:
                        msg_dict["name"] = m.name
                    msgs.append(msg_dict)
            if msgs and msgs[-1].get("role") == "assistant":
                msgs.append({"role": "user", "content": "Based on the conversation above, decide the next routing step."})
            return msgs

        supervisor_chain = RunnableLambda(_build_supervisor_messages) | llm.with_structured_output(ResearchPlanOutput)
        decision = supervisor_chain.invoke(state)
        _log(f"[SUPERVISOR] Route={decision.next_node}, confidence={decision.confidence:.2f}, research={decision.should_plan_research}")

        prev_node = state.get("next_node")
        same_node_count = state.get("same_node_count", 0)
        if prev_node == decision.next_node:
            same_node_count += 1
            if same_node_count >= 2:
                return FINISH
        else:
            same_node_count = 0

        result = {
            "iteration_count": count,
            "next_node": decision.next_node,
            "same_node_count": same_node_count,
        }

        if decision.should_plan_research and decision.research_plan:
            result["research_plan"] = {
                "queries": decision.research_plan.queries,
                "rationale": decision.research_plan.rationale,
            }
            _log(f"[RESEARCH PLAN] {len(decision.research_plan.queries)} queries: {decision.research_plan.queries}")

        if decision.refined_query:
            result["refined_query"] = decision.refined_query

        return result
    except Exception as e:
        _log(f"[ERROR] Supervisor routing failed: {e}")
        return FINISH


async def research_planner_node(state):
    try:
        research_plan = state.get("research_plan")
        if not research_plan or not research_plan.get("queries"):
            return {"messages": [{"role": "assistant", "content": "No research plan found.", "name": "ParallelExecutor"}]}

        queries = research_plan["queries"]
        rationale = research_plan.get("rationale", "")
        _log(f"[PARALLEL] Executing {len(queries)} queries in parallel...")

        QUERY_TIMEOUT_SECONDS = 120

        async def execute_genie_query_async(query: str, max_retries: int = 2) -> Dict[str, Any]:
            last_error = None
            for attempt in range(1, max_retries + 1):
                try:
                    query_state = {"messages": [{"role": "user", "content": query}]}
                    result = await asyncio.wait_for(
                        asyncio.to_thread(genie_agent.invoke, query_state),
                        timeout=QUERY_TIMEOUT_SECONDS,
                    )
                    return {
                        "query": query,
                        "success": True,
                        "response": result["messages"][-1].content if result.get("messages") else "No response",
                        "error": None,
                    }
                except asyncio.TimeoutError:
                    last_error = f"Timed out after {QUERY_TIMEOUT_SECONDS}s (attempt {attempt}/{max_retries})"
                    _log(f"[WARN] {last_error}: {query[:80]}...")
                except Exception as e:
                    last_error = f"{e} (attempt {attempt}/{max_retries})"
                    _log(f"[WARN] Genie query failed: {last_error}")
                if attempt < max_retries:
                    await asyncio.sleep(2 ** attempt)
            return {"query": query, "success": False, "response": None, "error": last_error}

        tasks = [execute_genie_query_async(q) for q in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({"query": queries[i], "success": False, "response": None, "error": str(result)})
                _log(f"[ERROR] Parallel execution failed for query '{queries[i]}': {result}")
            else:
                processed_results.append(result)
                if result["success"]:
                    _log(f"[PARALLEL] Query {i+1}/{len(queries)} completed successfully")

        response_parts = [f"Research Plan: {rationale}\n", "Parallel Research Results:\n"]
        for i, result in enumerate(processed_results, 1):
            response_parts.append(f"\n{i}. Query: {result['query']}")
            if result["success"]:
                response_parts.append(f"   Result: {result['response']}")
            else:
                response_parts.append(f"   Error: {result['error']}")

        response_parts.append(
            f"\n\nSynthesis: The parallel research has gathered comprehensive data from {len(queries)} different angles."
        )

        return {
            "messages": [{"role": "assistant", "content": "\n\n".join(response_parts), "name": "ParallelExecutor"}],
            "research_results": processed_results,
        }
    except Exception as e:
        _log(f"[ERROR] Parallel research execution failed: {e}")
        return {"messages": [{"role": "assistant", "content": f"Parallel research failed: {e}", "name": "ParallelExecutor"}]}


_GENIE_EMPTY_PATTERNS = [
    "no data found", "no results", "could not find", "couldn't find",
    "unable to retrieve", "no matching", "i don't have",
    "i couldn't", "no records", "data is not available",
    "not available in", "does not exist", "do not have access",
]


def _validate_genie_response(content: str) -> tuple:
    if not content or not content.strip():
        return False, "Empty response from Genie"
    lower = content.lower()
    for pattern in _GENIE_EMPTY_PATTERNS:
        if pattern in lower:
            return False, f"Genie indicated data unavailability: '{pattern}'"
    return True, ""


def _extract_user_query(messages):
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content", "")
            if content and content != FINAL_ANSWER_PROMPT:
                return content
        elif hasattr(msg, "role") and msg.role == "user":
            if hasattr(msg, "content") and msg.content and msg.content != FINAL_ANSWER_PROMPT:
                return msg.content
    return None


def agent_node(state, agent, name):
    query = state.get("refined_query")
    if not query:
        query = _extract_user_query(state.get("messages", []))

    if not query:
        return {"messages": [{"role": "assistant", "content": "No user query found.", "name": name}]}

    _log(f"[{name}] Processing query: {query[:100]}...")

    try:
        cached_response = genie_cache.get(query)
        if cached_response is not None:
            _log(f"[{name}] Cache hit — skipping Genie call")
            response_content = cached_response
        else:
            clean_state = {"messages": [{"role": "user", "content": query}]}
            result = agent.invoke(clean_state)

            if not result or "messages" not in result or not result["messages"]:
                raise ValueError(f"Invalid result structure from {name} agent")

            all_contents = []
            for msg in result["messages"]:
                content = msg.content if hasattr(msg, "content") else msg.get("content", "")
                if content and content.strip():
                    all_contents.append(content)
            response_content = "\n\n".join(all_contents) if all_contents else ""
            genie_cache.set(query, response_content)

        is_valid, reason = _validate_genie_response(response_content)
        if not is_valid:
            response_content = f"[Data Gap] {reason}. Original query: {query}\n\nRaw response: {response_content}"

        return {"messages": [{"role": "assistant", "content": response_content, "name": name}], "refined_query": None}
    except Exception as e:
        _log(f"[ERROR] {name} agent: {e}")
        return {"messages": [{"role": "assistant", "content": f"Error in {name}: {e}", "name": name}], "refined_query": None}


def final_answer(state):
    from langchain_core.runnables import RunnableLambda

    try:
        preprocessor = RunnableLambda(lambda s: s["messages"] + [{"role": "user", "content": FINAL_ANSWER_PROMPT}])
        final_answer_chain = preprocessor | llm
        answer_msg = final_answer_chain.invoke(state)
        answer_content = answer_msg.content if hasattr(answer_msg, "content") else str(answer_msg)

        user_query = _extract_user_query(state.get("messages", []))
        if user_query and QUALITY_CHECK_PROMPT:
            try:
                check_messages = [
                    {"role": "system", "content": QUALITY_CHECK_PROMPT},
                    {"role": "user", "content": f"Original question: {user_query}\n\nFinal answer: {answer_content}"},
                ]
                quality_result = llm.with_structured_output(QualityCheckOutput).invoke(check_messages)
                if not quality_result.passes_quality_check:
                    _log(f"[QUALITY] Failed: {quality_result.reason}. Attempting self-correction.")
                    correction_messages = state["messages"] + [
                        {"role": "assistant", "content": answer_content},
                        {"role": "user", "content": (
                            f"Your answer did not pass quality review. Reason: {quality_result.reason}\n\n"
                            "Revise your answer to directly address the question with specific data points.\n\n"
                            f"{FINAL_ANSWER_PROMPT}"
                        )},
                    ]
                    corrected_msg = llm.invoke(correction_messages)
                    return {"messages": [corrected_msg]}
            except Exception as qe:
                _log(f"[WARN] Quality check skipped: {qe}")

        return {"messages": [answer_msg]}
    except Exception as e:
        _log(f"[ERROR] Final answer generation failed: {e}")
        return {"messages": [{"role": "assistant", "content": f"Error generating final answer: {e}"}]}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def _run_async(query: str, thread_id: str | None = None) -> tuple[str, list[str]]:
    """Run the full multi-agent pipeline and return (answer, log_lines)."""
    global _log_lines
    _log_lines = []

    _ensure_initialized()

    thread_id = thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    request = {"messages": [{"role": "user", "content": query}]}

    final_content = ""
    async for event in multi_agent.astream(request, config=config, stream_mode="updates"):
        for node_name, node_data in event.items():
            if node_name == "final_answer":
                for msg in node_data.get("messages", []):
                    content = msg.content if hasattr(msg, "content") else msg.get("content", "")
                    if content:
                        final_content = content

    return final_content, list(_log_lines)


def run_query(query: str, thread_id: str | None = None) -> tuple[str, list[str]]:
    """Synchronous wrapper around the async pipeline. Returns (answer, log_lines).

    Runs the async pipeline in a dedicated thread with a fresh event loop
    so it works inside uvloop-based servers (Gradio/uvicorn) where
    nest_asyncio cannot patch the loop.
    """
    import concurrent.futures

    def _run_in_thread():
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_run_async(query, thread_id))
        finally:
            loop.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_run_in_thread)
        return future.result()
