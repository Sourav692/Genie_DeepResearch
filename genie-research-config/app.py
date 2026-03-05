import json
import os
import tempfile
import time
import traceback
from pathlib import Path

import gradio as gr
import yaml

_APP_DIR = Path(__file__).resolve().parent
_local_config = _APP_DIR / "configs.yaml"
_parent_config = _APP_DIR.parent / "configs.yaml"
CONFIG_PATH = _local_config if _local_config.exists() else _parent_config

_local_env = _APP_DIR / ".env"
_parent_env = _APP_DIR.parent / ".env"
ENV_PATH = _local_env if _local_env.exists() else _parent_env


def _load_env():
    """Load .env file into a dict (does not override os.environ)."""
    env = {}
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


def _load_yaml(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {}


def _save_yaml(path: Path, data: dict):
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True, width=120)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config():
    cfg = _load_yaml(CONFIG_PATH)
    db = cfg.get("databricks_configs", {})
    ag = cfg.get("agent_configs", {})
    llm = ag.get("llm", {})
    genie = ag.get("genie_agent", {})
    parallel = ag.get("parallel_executor_agent", {})
    sup = ag.get("supervisor_agent", {})
    conv = ag.get("conversation", {})

    return (
        db.get("catalog", ""),
        db.get("schema", ""),
        db.get("model", ""),
        db.get("workspace_url", ""),
        db.get("sql_warehouse_id", ""),
        db.get("mlflow_experiment_name", ""),
        ag.get("agent_name", ""),
        conv.get("max_messages", 7),
        llm.get("endpoint_name", ""),
        llm.get("temperature", 0.1),
        genie.get("space_id", ""),
        genie.get("description", ""),
        parallel.get("description", ""),
        sup.get("max_iterations", 6),
        sup.get("system_prompt", ""),
        sup.get("research_prompt", ""),
        sup.get("final_answer_prompt", ""),
        sup.get("post_worker_routing_prompt", ""),
        sup.get("quality_check_prompt", ""),
    )


def save_config(
    catalog, schema, model, workspace_url, sql_warehouse_id, mlflow_experiment,
    agent_name, max_messages,
    endpoint_name, temperature,
    genie_space_id, genie_description,
    parallel_description,
    max_iterations, system_prompt, research_prompt, final_answer_prompt,
    post_worker_routing_prompt, quality_check_prompt,
):
    cfg = _load_yaml(CONFIG_PATH)

    cfg["databricks_configs"] = {
        "catalog": catalog,
        "schema": schema,
        "model": model,
        "workspace_url": workspace_url,
        "sql_warehouse_id": sql_warehouse_id,
        "databricks_pat": cfg.get("databricks_configs", {}).get("databricks_pat", {}),
        "mlflow_experiment_name": mlflow_experiment,
    }

    cfg["agent_configs"] = {
        "agent_name": agent_name,
        "conversation": {"max_messages": int(max_messages)},
        "llm": {
            "endpoint_name": endpoint_name,
            "temperature": float(temperature),
        },
        "genie_agent": {
            "space_id": genie_space_id,
            "description": genie_description,
        },
        "parallel_executor_agent": {
            "description": parallel_description,
        },
        "supervisor_agent": {
            "max_iterations": int(max_iterations),
            "system_prompt": system_prompt,
            "research_prompt": research_prompt,
            "final_answer_prompt": final_answer_prompt,
            "post_worker_routing_prompt": post_worker_routing_prompt,
            "quality_check_prompt": quality_check_prompt,
        },
    }

    _save_yaml(CONFIG_PATH, cfg)
    return gr.Info("Configuration saved successfully.")


def export_yaml(
    catalog, schema, model, workspace_url, sql_warehouse_id, mlflow_experiment,
    agent_name, max_messages,
    endpoint_name, temperature,
    genie_space_id, genie_description,
    parallel_description,
    max_iterations, system_prompt, research_prompt, final_answer_prompt,
    post_worker_routing_prompt, quality_check_prompt,
):
    cfg = {
        "databricks_configs": {
            "catalog": catalog,
            "schema": schema,
            "model": model,
            "workspace_url": workspace_url,
            "sql_warehouse_id": sql_warehouse_id,
            "mlflow_experiment_name": mlflow_experiment,
        },
        "agent_configs": {
            "agent_name": agent_name,
            "conversation": {"max_messages": int(max_messages)},
            "llm": {
                "endpoint_name": endpoint_name,
                "temperature": float(temperature),
            },
            "genie_agent": {
                "space_id": genie_space_id,
                "description": genie_description,
            },
            "parallel_executor_agent": {
                "description": parallel_description,
            },
            "supervisor_agent": {
                "max_iterations": int(max_iterations),
                "system_prompt": system_prompt,
                "research_prompt": research_prompt,
                "final_answer_prompt": final_answer_prompt,
                "post_worker_routing_prompt": post_worker_routing_prompt,
                "quality_check_prompt": quality_check_prompt,
            },
        },
    }

    tmp = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w")
    yaml.dump(cfg, tmp, default_flow_style=False, sort_keys=False, allow_unicode=True, width=120)
    tmp.close()
    return tmp.name


# ---------------------------------------------------------------------------
# Agent runner
# ---------------------------------------------------------------------------

_full_agent_module = None
_full_agent_error = None


def _get_workspace_client():
    """Get a WorkspaceClient that auto-discovers credentials in any environment.
    Works with PAT (local dev), service principal OAuth (Databricks Apps), and CLI profiles.
    """
    _load_env_vars()
    try:
        from databricks.sdk import WorkspaceClient
        return WorkspaceClient()
    except Exception as e:
        raise RuntimeError(
            f"Could not authenticate to Databricks. "
            f"Set DATABRICKS_HOST + DATABRICKS_TOKEN in .env for local dev, "
            f"or ensure the app's service principal has correct permissions when deployed.\n\n{e}"
        )


def _load_env_vars():
    """Push .env values into os.environ so the SDK can pick them up."""
    env = _load_env()
    for k, v in env.items():
        if k not in os.environ:
            os.environ[k] = v


def _get_full_agent():
    """Lazy-load the full agent pipeline on first use."""
    global _full_agent_module, _full_agent_error
    if _full_agent_module is not None:
        return _full_agent_module
    if _full_agent_error is not None:
        return None
    try:
        from agent_runner import run_query
        _full_agent_module = run_query
        return _full_agent_module
    except Exception as e:
        _full_agent_error = str(e)
        return None


def run_full_agent(query: str):
    """Run the full multi-agent pipeline (supervisor + parallel executor + quality check)."""
    if not query.strip():
        return "Please enter a query."

    run_fn = _get_full_agent()
    if run_fn is None:
        return (f"**Error**: Could not load the full agent pipeline.\n\n"
                f"```\n{_full_agent_error}\n```\n\n"
                "Make sure all dependencies are installed: `databricks-langchain`, `langgraph`, `mlflow`, etc.")

    t0 = time.time()
    try:
        answer, log_lines = run_fn(query)
        elapsed = time.time() - t0

        log_section = "\n".join(log_lines) if log_lines else "No routing logs captured."

        return (f"{answer}\n\n"
                f"---\n"
                f"*Completed in {elapsed:.1f}s via Full Agent Pipeline*\n\n"
                f"<details><summary>Agent Routing Log</summary>\n\n"
                f"```\n{log_section}\n```\n\n"
                f"</details>")
    except Exception as e:
        elapsed = time.time() - t0
        import traceback
        return f"**Error** after {elapsed:.1f}s:\n\n```\n{traceback.format_exc()}\n```"


def run_serving_endpoint(query: str, endpoint_name: str):
    """Call a deployed MLflow ChatAgent serving endpoint via the Databricks SDK."""
    if not query.strip():
        return "Please enter a query."
    if not endpoint_name.strip():
        return "Please enter the serving endpoint name."

    t0 = time.time()
    try:
        w = _get_workspace_client()
        resp = w.serving_endpoints.query(
            name=endpoint_name.strip(),
            messages=[{"role": "user", "content": query.strip()}],
        )
        elapsed = time.time() - t0

        if hasattr(resp, "choices") and resp.choices:
            content = resp.choices[0].message.content
        elif hasattr(resp, "messages") and resp.messages:
            content = resp.messages[-1].get("content", json.dumps(resp.as_dict(), indent=2))
        else:
            content = json.dumps(resp.as_dict(), indent=2)

        return f"{content}\n\n---\n*Completed in {elapsed:.1f}s via `{endpoint_name}`*"
    except Exception as e:
        elapsed = time.time() - t0
        return f"**Error** after {elapsed:.1f}s:\n\n```\n{e}\n```"


def run_genie_direct(query: str, space_id: str):
    """Call Databricks Genie API directly via the SDK for quick single-query testing."""
    if not query.strip():
        return "Please enter a query."
    if not space_id.strip():
        return "Please enter the Genie Space ID (from Genie Agent tab)."

    t0 = time.time()
    try:
        w = _get_workspace_client()
        genie_api = w.genie

        start = genie_api.start_conversation(space_id=space_id.strip(), content=query.strip())
        conversation_id = start.conversation_id
        message_id = start.message_id

        if not conversation_id or not message_id:
            return f"**Unexpected response**: no conversation_id or message_id returned."

        for _ in range(60):
            time.sleep(5)
            msg = genie_api.get_message(space_id=space_id.strip(),
                                        conversation_id=conversation_id,
                                        message_id=message_id)
            status = msg.status.value if hasattr(msg.status, "value") else str(msg.status)

            if status == "COMPLETED":
                elapsed = time.time() - t0
                parts = []
                for att in (msg.attachments or []):
                    if hasattr(att, "text") and att.text and hasattr(att.text, "content") and att.text.content:
                        parts.append(att.text.content)
                    if hasattr(att, "query") and att.query and hasattr(att.query, "query") and att.query.query:
                        parts.append(f"**SQL Generated:**\n```sql\n{att.query.query}\n```")
                content = "\n\n".join(parts) if parts else str(msg)
                return f"{content}\n\n---\n*Completed in {elapsed:.1f}s via Genie Space `{space_id}`*"

            if status in ("FAILED", "CANCELLED"):
                return f"**Genie query {status}**:\n\n```\n{msg}\n```"

        return "**Error**: Genie query timed out after 5 minutes of polling."

    except Exception:
        return f"**Error**:\n\n```\n{traceback.format_exc()}\n```"


SAMPLE_QUERIES = [
    "What are the lead counts for stage 4 for 6th December?",
    "Analyze all KPI failures in the last 7 days. Cluster failures by: time of day, product type, journey stage, platform.",
    "Compare performance metrics between November 2025 and December 2025 across all products.",
    "Analyze the impact of app version updates on user behavior. Compare Android versions P.26.0.0 vs P.13.3.0.",
]


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter")],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono")],
).set(
    body_background_fill="linear-gradient(135deg, #f0f4ff 0%, #e8edf5 50%, #f0f4ff 100%)",
    block_background_fill="white",
    block_border_width="0px",
    block_shadow="0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04)",
    block_radius="12px",
    input_background_fill="#f8fafc",
    input_border_width="1px",
    input_border_color="#e2e8f0",
    input_radius="8px",
    button_primary_background_fill="linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%)",
    button_primary_text_color="white",
    button_primary_border_color="transparent",
    button_secondary_background_fill="white",
    button_secondary_text_color="#374151",
    button_secondary_border_color="#d1d5db",
    shadow_spread="0px",
)

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Global ──────────────────────────────────────────────────────── */
.gradio-container {
    max-width: 1100px !important;
    margin: 0 auto !important;
    padding: 0 1.5rem !important;
}
footer { display: none !important; }

/* ── Header Banner ───────────────────────────────────────────────── */
.hero-banner {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 40%, #2563eb 100%);
    border-radius: 16px;
    padding: 2.5rem 2rem 2rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(37, 99, 235, 0.18);
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -60%;
    right: -20%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(255,255,255,0.06) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-banner::after {
    content: '';
    position: absolute;
    bottom: -40%;
    left: -10%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(59,130,246,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-content {
    position: relative;
    z-index: 1;
    text-align: center;
}
.hero-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 56px;
    height: 56px;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 14px;
    margin-bottom: 1rem;
    backdrop-filter: blur(8px);
}
.hero-icon svg {
    width: 28px;
    height: 28px;
    color: white;
}
.hero-content h1 {
    font-size: 1.75rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.02em;
}
.hero-content p {
    color: rgba(255,255,255,0.65);
    font-size: 0.9rem;
    font-weight: 400;
    margin: 0;
    letter-spacing: 0.01em;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 20px;
    padding: 4px 14px;
    margin-top: 1rem;
    font-size: 0.75rem;
    color: rgba(255,255,255,0.7);
    letter-spacing: 0.03em;
    text-transform: uppercase;
    font-weight: 500;
}
.hero-badge .dot {
    width: 6px;
    height: 6px;
    background: #34d399;
    border-radius: 50%;
    animation: pulse-dot 2s ease-in-out infinite;
}
@keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(0.85); }
}

/* ── Tabs ─────────────────────────────────────────────────────────── */
.tabs > .tab-nav {
    background: white !important;
    border-radius: 12px !important;
    padding: 6px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
    margin-bottom: 1rem !important;
    border: 1px solid #e5e7eb !important;
    gap: 4px !important;
}
.tabs > .tab-nav > button {
    border-radius: 8px !important;
    padding: 10px 20px !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    color: #64748b !important;
    border: none !important;
    transition: all 0.2s ease !important;
    background: transparent !important;
}
.tabs > .tab-nav > button:hover {
    color: #1e40af !important;
    background: #eff6ff !important;
}
.tabs > .tab-nav > button.selected {
    background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
    color: white !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 8px rgba(37, 99, 235, 0.25) !important;
}

/* ── Section cards ────────────────────────────────────────────────── */
.section-card {
    background: white;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    border: 1px solid #f1f5f9;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    transition: box-shadow 0.2s ease;
}
.section-card:hover {
    box-shadow: 0 4px 12px rgba(0,0,0,0.06);
}
.section-title {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 0 0 1rem 0;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid #f1f5f9;
}
.section-title .icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    border-radius: 8px;
    flex-shrink: 0;
}
.section-title .icon.blue   { background: #eff6ff; color: #2563eb; }
.section-title .icon.green  { background: #f0fdf4; color: #16a34a; }
.section-title .icon.purple { background: #faf5ff; color: #7c3aed; }
.section-title .icon.amber  { background: #fffbeb; color: #d97706; }
.section-title .icon.rose   { background: #fff1f2; color: #e11d48; }
.section-title .icon.cyan   { background: #ecfeff; color: #0891b2; }
.section-title h3 {
    font-size: 0.95rem;
    font-weight: 600;
    color: #1e293b;
    margin: 0;
    letter-spacing: -0.01em;
}
.section-title p {
    font-size: 0.8rem;
    color: #94a3b8;
    margin: 2px 0 0 0;
}

/* ── Prompt text areas ────────────────────────────────────────────── */
.prompt-box textarea {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
    line-height: 1.6 !important;
    background: #f8fafc !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
    padding: 12px !important;
}
.prompt-box textarea:focus {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.08) !important;
}

/* ── Execution Mode Radio ─────────────────────────────────────────── */
.execution-mode .wrap {
    gap: 8px !important;
}
.execution-mode label {
    border-radius: 8px !important;
    padding: 10px 18px !important;
    border: 1px solid #e2e8f0 !important;
    font-weight: 500 !important;
    transition: all 0.15s ease !important;
}
.execution-mode label.selected {
    background: #eff6ff !important;
    border-color: #2563eb !important;
    color: #1d4ed8 !important;
}

/* ── Run Button ───────────────────────────────────────────────────── */
.run-btn button {
    width: 100% !important;
    padding: 14px !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
    box-shadow: 0 4px 14px rgba(37, 99, 235, 0.3) !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.01em !important;
}
.run-btn button:hover {
    box-shadow: 0 6px 20px rgba(37, 99, 235, 0.4) !important;
    transform: translateY(-1px) !important;
}

/* ── Response output ──────────────────────────────────────────────── */
.response-output {
    background: white !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    padding: 1.5rem !important;
    min-height: 120px;
}
.response-output .prose {
    font-size: 0.9rem !important;
    line-height: 1.7 !important;
    color: #334155 !important;
}
.response-output h1, .response-output h2, .response-output h3 {
    color: #0f172a !important;
    margin-top: 1.2rem !important;
}
.response-output code {
    background: #f1f5f9 !important;
    padding: 2px 6px !important;
    border-radius: 4px !important;
    font-size: 0.82rem !important;
}
.response-output pre {
    background: #0f172a !important;
    border-radius: 8px !important;
    padding: 1rem !important;
}
.response-output pre code {
    background: transparent !important;
    color: #e2e8f0 !important;
}

/* ── Action bar ───────────────────────────────────────────────────── */
.action-bar {
    background: white;
    border-radius: 12px;
    padding: 1rem;
    margin-top: 0.5rem;
    border: 1px solid #e5e7eb;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.action-bar button {
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    padding: 10px 24px !important;
    transition: all 0.15s ease !important;
}

/* ── Query area ───────────────────────────────────────────────────── */
.query-input textarea {
    border-radius: 10px !important;
    border: 1.5px solid #e2e8f0 !important;
    padding: 12px 14px !important;
    font-size: 0.9rem !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}
.query-input textarea:focus {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.08) !important;
}

/* ── Form field polish ────────────────────────────────────────────── */
.gradio-container input[type="text"],
.gradio-container input[type="number"] {
    border-radius: 8px !important;
    font-size: 0.88rem !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}
.gradio-container input[type="text"]:focus,
.gradio-container input[type="number"]:focus {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.08) !important;
}
label > span {
    font-weight: 500 !important;
    font-size: 0.84rem !important;
    color: #475569 !important;
}
"""

HERO_HTML = """
<div class="hero-banner">
  <div class="hero-content">
    <div class="hero-icon">
      <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor">
        <path stroke-linecap="round" stroke-linejoin="round"
          d="M9.813 15.904 9 18.75l-.813-2.846a4.5 4.5 0 0 0-3.09-3.09L2.25 12l2.846-.813a4.5
             4.5 0 0 0 3.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 0 0 3.09 3.09L15.75 12l-2.846.813a4.5
             4.5 0 0 0-3.09 3.09ZM18.259 8.715 18 9.75l-.259-1.035a3.375 3.375 0 0
             0-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 0 0 2.455-2.456L18 2.25l.259
             1.035a3.375 3.375 0 0 0 2.455 2.456L21.75 6l-1.036.259a3.375 3.375 0 0
             0-2.455 2.456ZM16.894 20.567 16.5 21.75l-.394-1.183a2.25 2.25 0 0
             0-1.423-1.423L13.5 18.75l1.183-.394a2.25 2.25 0 0 0 1.423-1.423l.394-1.183.394
             1.183a2.25 2.25 0 0 0 1.423 1.423l1.183.394-1.183.394a2.25 2.25 0 0
             0-1.423 1.423Z" />
      </svg>
    </div>
    <h1>Genie Deep Research Agent</h1>
    <p>AI-powered multi-agent analytics platform</p>
    <div class="hero-badge">
      <span class="dot"></span>
      Configuration &amp; Execution Console
    </div>
  </div>
</div>
"""


def _section(icon_class, icon_svg, title, subtitle=None):
    sub = f"<p>{subtitle}</p>" if subtitle else ""
    return (
        f'<div class="section-title">'
        f'<div class="icon {icon_class}">{icon_svg}</div>'
        f"<div><h3>{title}</h3>{sub}</div>"
        f"</div>"
    )


_ICON_PLAY = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path d="m11.596 8.697-6.363 3.692c-.54.313-1.233-.066-1.233-.697V4.308c0-.63.692-1.01 1.233-.696l6.363 3.692a.802.802 0 0 1 0 1.393z"/></svg>'
_ICON_DB = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path d="M3.904 1.777C4.978 1.289 6.427 1 8 1s3.022.289 4.096.777C13.125 2.245 14 2.993 14 4s-.875 1.755-1.904 2.223C11.022 6.711 9.573 7 8 7s-3.022-.289-4.096-.777C2.875 5.755 2 5.007 2 4s.875-1.755 1.904-2.223Z"/><path d="M2 6.161V7c0 1.007.875 1.755 1.904 2.223C4.978 9.711 6.427 10 8 10s3.022-.289 4.096-.777C13.125 8.755 14 8.007 14 7v-.839c-.457.432-1.004.751-1.49.972C11.278 7.693 9.682 8 8 8s-3.278-.307-4.51-.867c-.486-.22-1.033-.54-1.49-.972Z"/><path d="M2 9.161V10c0 1.007.875 1.755 1.904 2.223C4.978 12.711 6.427 13 8 13s3.022-.289 4.096-.777C13.125 11.755 14 11.007 14 10v-.839c-.457.432-1.004.751-1.49.972-1.232.56-2.828.867-4.51.867s-3.278-.307-4.51-.867c-.486-.22-1.033-.54-1.49-.972Z"/><path d="M2 12.161V13c0 1.007.875 1.755 1.904 2.223C4.978 15.711 6.427 16 8 16s3.022-.289 4.096-.777C13.125 14.755 14 14.007 14 13v-.839c-.457.432-1.004.751-1.49.972-1.232.56-2.828.867-4.51.867s-3.278-.307-4.51-.867c-.486-.22-1.033-.54-1.49-.972Z"/></svg>'
_ICON_GEAR = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path d="M8 4.754a3.246 3.246 0 1 0 0 6.492 3.246 3.246 0 0 0 0-6.492zM5.754 8a2.246 2.246 0 1 1 4.492 0 2.246 2.246 0 0 1-4.492 0z"/><path d="M9.796 1.343c-.527-1.79-3.065-1.79-3.592 0l-.094.319a.873.873 0 0 1-1.255.52l-.292-.16c-1.64-.892-3.433.902-2.54 2.541l.159.292a.873.873 0 0 1-.52 1.255l-.319.094c-1.79.527-1.79 3.065 0 3.592l.319.094a.873.873 0 0 1 .52 1.255l-.16.292c-.892 1.64.901 3.434 2.541 2.54l.292-.159a.873.873 0 0 1 1.255.52l.094.319c.527 1.79 3.065 1.79 3.592 0l.094-.319a.873.873 0 0 1 1.255-.52l.292.16c1.64.893 3.434-.902 2.54-2.541l-.159-.292a.873.873 0 0 1 .52-1.255l.319-.094c1.79-.527 1.79-3.065 0-3.592l-.319-.094a.873.873 0 0 1-.52-1.255l.16-.292c.893-1.64-.902-3.433-2.541-2.54l-.292.159a.873.873 0 0 1-1.255-.52l-.094-.319zm-2.633.283c.246-.835 1.428-.835 1.674 0l.094.319a1.873 1.873 0 0 0 2.693 1.115l.291-.16c.764-.415 1.6.42 1.184 1.185l-.159.292a1.873 1.873 0 0 0 1.116 2.692l.318.094c.835.246.835 1.428 0 1.674l-.319.094a1.873 1.873 0 0 0-1.115 2.693l.16.291c.415.764-.42 1.6-1.185 1.184l-.291-.159a1.873 1.873 0 0 0-2.693 1.116l-.094.318c-.246.835-1.428.835-1.674 0l-.094-.319a1.873 1.873 0 0 0-2.692-1.115l-.292.16c-.764.415-1.6-.42-1.184-1.185l.159-.291A1.873 1.873 0 0 0 1.945 8.93l-.319-.094c-.835-.246-.835-1.428 0-1.674l.319-.094A1.873 1.873 0 0 0 3.06 4.377l-.16-.292c-.415-.764.42-1.6 1.185-1.184l.292.159a1.873 1.873 0 0 0 2.692-1.115l.094-.319z"/></svg>'
_ICON_SPARKLE = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path d="M7.657 6.247c.11-.33.576-.33.686 0l.645 1.937a2.89 2.89 0 0 0 1.829 1.828l1.936.645c.33.11.33.576 0 .686l-1.937.645a2.89 2.89 0 0 0-1.828 1.829l-.645 1.936a.361.361 0 0 1-.686 0l-.645-1.937a2.89 2.89 0 0 0-1.828-1.828l-1.937-.645a.361.361 0 0 1 0-.686l1.937-.645a2.89 2.89 0 0 0 1.828-1.829l.645-1.936zM13.794 1.12a.36.36 0 0 1 .686 0l.374 1.123a2.89 2.89 0 0 0 1.83 1.83l1.122.373a.36.36 0 0 1 0 .686l-1.123.374a2.89 2.89 0 0 0-1.829 1.83l-.374 1.122a.36.36 0 0 1-.686 0l-.374-1.123a2.89 2.89 0 0 0-1.83-1.829l-1.122-.374a.36.36 0 0 1 0-.686l1.123-.374a2.89 2.89 0 0 0 1.829-1.83l.374-1.122zM8.572.057a.36.36 0 0 1 .686 0l.213.64a2.89 2.89 0 0 0 1.83 1.83l.64.212a.36.36 0 0 1 0 .686l-.64.213a2.89 2.89 0 0 0-1.83 1.83l-.213.64a.36.36 0 0 1-.686 0l-.213-.64a2.89 2.89 0 0 0-1.83-1.83l-.64-.213a.36.36 0 0 1 0-.686l.64-.213a2.89 2.89 0 0 0 1.83-1.83L8.572.057z"/></svg>'
_ICON_BOLT = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path d="M5.52.359A.5.5 0 0 1 6 0h4a.5.5 0 0 1 .474.658L8.694 6H12.5a.5.5 0 0 1 .395.807l-7 9a.5.5 0 0 1-.873-.454L6.823 9.5H3.5a.5.5 0 0 1-.48-.641l2.5-8.5z"/></svg>'
_ICON_BRAIN = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path d="M12.096 13.634a1 1 0 0 1 .274-.504l.078-.078a.5.5 0 0 0-.354-.854h-.022a.5.5 0 0 0-.354.147l-.078.078a1 1 0 0 1-.504.274 1 1 0 0 1-.574-.026l-.006-.003a1 1 0 0 1-.482-.39l-.043-.07a.5.5 0 0 0-.432-.249h-.003a.5.5 0 0 0-.432.75l.043.07a1 1 0 0 1 .11.568l-.005.078a1 1 0 0 1-.32.636l-.038.035a.5.5 0 0 0 .34.863h.025a.5.5 0 0 0 .354-.147l.037-.035a1 1 0 0 1 .637-.32l.077-.005a1 1 0 0 1 .568.11l.07.043a.5.5 0 0 0 .75-.432v-.003a.5.5 0 0 0-.249-.432l-.07-.043a1 1 0 0 1-.39-.482l-.003-.006a1 1 0 0 1-.026-.574Z"/><path d="M8 1a5 5 0 0 0-5 5c0 1.863 1.02 3.47 2.512 4.34.17.098.28.275.28.47v.926a.5.5 0 0 0 .5.5h3.416a.5.5 0 0 0 .5-.5v-.926c0-.195.11-.372.28-.47A5 5 0 0 0 8 1Zm2.354 7.768A3.5 3.5 0 1 0 4.646 5.232a3.5 3.5 0 0 0 5.708 3.536Z"/></svg>'


with gr.Blocks(theme=THEME, css=CSS, title="Genie Deep Research Agent") as app:

    gr.HTML(HERO_HTML)

    all_fields = []

    with gr.Tabs(elem_classes=["tabs"]):

        # ── Tab 0: Run Agent ──────────────────────────────────────────
        with gr.Tab("Run Agent", id="run"):
            with gr.Column(elem_classes=["section-card"]):
                gr.HTML(_section("blue", _ICON_PLAY, "Query the Agent", "Select execution mode and enter your research query"))

                f_run_mode = gr.Radio(
                    choices=["Full Agent", "Direct Genie", "Serving Endpoint"],
                    value="Full Agent",
                    label="Execution Mode",
                    info="Full Agent = supervisor + parallel execution + quality check (same as notebook)",
                    elem_classes=["execution-mode"],
                )
                with gr.Row():
                    f_serving_ep = gr.Textbox(
                        label="Serving Endpoint Name",
                        placeholder="e.g. agents_journey-kpi-reasoning-agent",
                        visible=False,
                    )
                    f_run_space_id = gr.Textbox(
                        label="Genie Space ID",
                        placeholder="Loaded from config on startup",
                        visible=False,
                    )

                def _toggle_mode(mode):
                    return (
                        gr.update(visible=mode == "Serving Endpoint"),
                        gr.update(visible=mode == "Direct Genie"),
                    )

                f_run_mode.change(fn=_toggle_mode, inputs=[f_run_mode], outputs=[f_serving_ep, f_run_space_id])

                f_sample = gr.Dropdown(
                    choices=[""] + SAMPLE_QUERIES,
                    value="",
                    label="Sample Queries (pick one or type your own below)",
                )
                f_query = gr.Textbox(
                    label="Your Query",
                    lines=3,
                    placeholder="Ask a question about your data...",
                    elem_classes=["query-input"],
                )

                def _fill_sample(sample):
                    return sample if sample else gr.update()

                f_sample.change(fn=_fill_sample, inputs=[f_sample], outputs=[f_query])

                btn_run = gr.Button("Run Research Query", variant="primary", size="lg", elem_classes=["run-btn"])

            with gr.Column(elem_classes=["section-card"]):
                gr.HTML(_section("green", _ICON_SPARKLE, "Agent Response", "Results from the research pipeline"))
                f_output = gr.Markdown(
                    label="Agent Response",
                    value="*Waiting for a query...*",
                    elem_classes=["response-output"],
                )

            def _run(mode, query, endpoint, space_id):
                if mode == "Full Agent":
                    return run_full_agent(query)
                if mode == "Serving Endpoint":
                    return run_serving_endpoint(query, endpoint)
                return run_genie_direct(query, space_id)

            btn_run.click(
                fn=_run,
                inputs=[f_run_mode, f_query, f_serving_ep, f_run_space_id],
                outputs=[f_output],
            )

        # ── Tab 1: Connection & Infrastructure ────────────────────────
        with gr.Tab("Connection & Infrastructure"):
            with gr.Column(elem_classes=["section-card"]):
                gr.HTML(_section("blue", _ICON_DB, "Databricks Connection", "Unity Catalog and warehouse settings"))
                with gr.Row():
                    f_catalog = gr.Textbox(label="Catalog", placeholder="e.g. bfl_std_lake")
                    f_schema = gr.Textbox(label="Schema", placeholder="e.g. digital360_preprod")
                    f_model = gr.Textbox(label="Model", placeholder="e.g. bajaj_funnel_analysis")
                with gr.Row():
                    f_workspace_url = gr.Textbox(label="Workspace URL", placeholder="https://...")
                    f_sql_warehouse_id = gr.Textbox(label="SQL Warehouse ID", placeholder="e.g. 4b9b953939869799")
                f_mlflow_experiment = gr.Textbox(label="MLflow Experiment Name", placeholder="e.g. bajaj_research")

            with gr.Column(elem_classes=["section-card"]):
                gr.HTML(_section("green", _ICON_GEAR, "Agent Settings", "Agent identity and conversation limits"))
                with gr.Row():
                    f_agent_name = gr.Textbox(label="Agent Name", placeholder="e.g. journey-kpi-reasoning-agent")
                    f_max_messages = gr.Number(label="Max Conversation Messages", value=7, precision=0, minimum=1, maximum=50)

            with gr.Column(elem_classes=["section-card"]):
                gr.HTML(_section("purple", _ICON_BRAIN, "LLM Configuration", "Model endpoint and inference parameters"))
                with gr.Row():
                    f_endpoint = gr.Textbox(label="LLM Endpoint Name", placeholder="e.g. databricks-claude-3-7-sonnet")
                    f_temperature = gr.Slider(label="Temperature", minimum=0.0, maximum=1.0, step=0.05, value=0.1)

        # ── Tab 2: Genie Agent ────────────────────────────────────────
        with gr.Tab("Genie Agent"):
            with gr.Column(elem_classes=["section-card"]):
                gr.HTML(_section("cyan", _ICON_SPARKLE, "Genie Space Configuration", "Connect to a Databricks Genie space for data querying"))
                f_genie_space_id = gr.Textbox(label="Genie Space ID", placeholder="e.g. 01f0d6ff25da...")
                f_genie_description = gr.Textbox(
                    label="Genie Agent Description",
                    lines=10,
                    placeholder="Describe the data and capabilities available to the Genie agent...",
                    elem_classes=["prompt-box"],
                )

        # ── Tab 3: Parallel Executor ──────────────────────────────────
        with gr.Tab("Parallel Executor"):
            with gr.Column(elem_classes=["section-card"]):
                gr.HTML(_section("amber", _ICON_BOLT, "Parallel Executor Agent", "Configure how sub-queries are distributed and executed concurrently"))
                f_parallel_description = gr.Textbox(
                    label="Parallel Executor Description",
                    lines=8,
                    placeholder="Describe the parallel execution capabilities...",
                    elem_classes=["prompt-box"],
                )

        # ── Tab 4: Supervisor & Prompts ───────────────────────────────
        with gr.Tab("Supervisor & Prompts"):
            with gr.Column(elem_classes=["section-card"]):
                gr.HTML(_section("rose", _ICON_GEAR, "Supervisor Settings", "Control iteration limits for the orchestration loop"))
                f_max_iterations = gr.Number(label="Max Supervisor Iterations", value=6, precision=0, minimum=1, maximum=20)

            with gr.Column(elem_classes=["section-card"]):
                gr.HTML(_section("purple", _ICON_BRAIN, "System Prompt"))
                f_system_prompt = gr.Textbox(
                    label="System Prompt",
                    lines=20,
                    placeholder="The core supervisor system prompt with HITL workflow...",
                    elem_classes=["prompt-box"],
                )

            with gr.Column(elem_classes=["section-card"]):
                gr.HTML(_section("blue", _ICON_SPARKLE, "Research Prompt"))
                f_research_prompt = gr.Textbox(
                    label="Research Prompt",
                    lines=20,
                    placeholder="Routing decision criteria with HITL support...",
                    elem_classes=["prompt-box"],
                )

            with gr.Column(elem_classes=["section-card"]):
                gr.HTML(_section("green", _ICON_SPARKLE, "Final Answer Prompt"))
                f_final_answer_prompt = gr.Textbox(
                    label="Final Answer Prompt",
                    lines=15,
                    placeholder="Instructions for generating the final answer...",
                    elem_classes=["prompt-box"],
                )

            with gr.Column(elem_classes=["section-card"]):
                gr.HTML(_section("amber", _ICON_BOLT, "Post-Worker Routing Prompt"))
                f_post_worker_routing_prompt = gr.Textbox(
                    label="Post-Worker Routing Prompt",
                    lines=10,
                    placeholder="Lightweight routing prompt used after workers report back...",
                    elem_classes=["prompt-box"],
                )

            with gr.Column(elem_classes=["section-card"]):
                gr.HTML(_section("rose", _ICON_GEAR, "Quality Check Prompt"))
                f_quality_check_prompt = gr.Textbox(
                    label="Quality Check Prompt",
                    lines=10,
                    placeholder="Self-reflection prompt for answer quality validation...",
                    elem_classes=["prompt-box"],
                )

    all_fields = [
        f_catalog, f_schema, f_model, f_workspace_url, f_sql_warehouse_id, f_mlflow_experiment,
        f_agent_name, f_max_messages,
        f_endpoint, f_temperature,
        f_genie_space_id, f_genie_description,
        f_parallel_description,
        f_max_iterations, f_system_prompt, f_research_prompt, f_final_answer_prompt,
        f_post_worker_routing_prompt, f_quality_check_prompt,
    ]

    # ── Action Buttons ────────────────────────────────────────────────
    with gr.Row(elem_classes=["action-bar"]):
        btn_load = gr.Button("Load Config", variant="secondary", size="lg")
        btn_save = gr.Button("Save Config", variant="primary", size="lg")
        btn_export = gr.DownloadButton("Export as YAML", variant="secondary", size="lg")

    btn_load.click(fn=load_config, inputs=[], outputs=all_fields)
    btn_save.click(fn=save_config, inputs=all_fields, outputs=[])
    btn_export.click(fn=export_yaml, inputs=all_fields)

    def _load_all():
        """Load config and also populate the Run tab's Genie Space ID."""
        vals = load_config()
        genie_space = vals[10]
        return (*vals, genie_space)

    app.load(fn=_load_all, inputs=[], outputs=[*all_fields, f_run_space_id])


if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "8000")),
    )
