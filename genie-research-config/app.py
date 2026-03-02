import json
import os
import tempfile
import time
import traceback
from pathlib import Path

import gradio as gr
import requests
import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs.yaml"
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


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

_full_agent_loaded = False
_full_agent_error = None

try:
    from agent_runner import run_query as _run_full_agent
    _full_agent_loaded = True
except Exception as e:
    _full_agent_error = str(e)


def _get_databricks_creds():
    env = _load_env()
    host = os.environ.get("DATABRICKS_HOST") or env.get("DATABRICKS_HOST", "")
    token = os.environ.get("DATABRICKS_TOKEN") or env.get("DATABRICKS_TOKEN", "")
    return host.rstrip("/"), token


def run_full_agent(query: str):
    """Run the full multi-agent pipeline (supervisor + parallel executor + quality check)."""
    if not query.strip():
        return "Please enter a query."
    if not _full_agent_loaded:
        return (f"**Error**: Could not load the full agent pipeline.\n\n"
                f"```\n{_full_agent_error}\n```\n\n"
                "Make sure all dependencies are installed: `databricks-langchain`, `langgraph`, `mlflow`, etc.")

    t0 = time.time()
    try:
        answer, log_lines = _run_full_agent(query)
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
        return f"**Error** after {elapsed:.1f}s:\n\n```\n{e}\n```"


def run_serving_endpoint(query: str, endpoint_name: str):
    """Call a deployed MLflow ChatAgent serving endpoint."""
    if not query.strip():
        return "Please enter a query."
    if not endpoint_name.strip():
        return "Please enter the serving endpoint name."

    host, token = _get_databricks_creds()
    if not host or not token:
        return ("**Error**: `DATABRICKS_HOST` and `DATABRICKS_TOKEN` must be set "
                "in `.env` or environment variables.")

    url = f"{host}/serving-endpoints/{endpoint_name.strip()}/invocations"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {
        "messages": [{"role": "user", "content": query.strip()}]
    }

    t0 = time.time()
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=300)
        elapsed = time.time() - t0
    except requests.exceptions.Timeout:
        return "**Error**: Request timed out after 5 minutes."
    except requests.exceptions.ConnectionError as e:
        return f"**Error**: Could not connect to Databricks workspace.\n\n`{e}`"

    if resp.status_code != 200:
        return (f"**Error** (HTTP {resp.status_code}):\n\n"
                f"```\n{resp.text[:2000]}\n```")

    try:
        result = resp.json()
        messages = result.get("messages") or result.get("choices", [])
        if messages:
            last = messages[-1]
            content = last.get("content") or last.get("message", {}).get("content", "")
        else:
            content = json.dumps(result, indent=2)
    except Exception:
        content = resp.text

    return f"{content}\n\n---\n*Completed in {elapsed:.1f}s via `{endpoint_name}`*"


def run_genie_direct(query: str, space_id: str):
    """Call Databricks Genie API directly for quick single-query testing."""
    if not query.strip():
        return "Please enter a query."
    if not space_id.strip():
        return "Please enter the Genie Space ID (from Genie Agent tab)."

    host, token = _get_databricks_creds()
    if not host or not token:
        return ("**Error**: `DATABRICKS_HOST` and `DATABRICKS_TOKEN` must be set "
                "in `.env` or environment variables.")

    start_url = f"{host}/api/2.0/genie/spaces/{space_id.strip()}/start-conversation"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    t0 = time.time()
    try:
        resp = requests.post(start_url, headers=headers, json={"content": query.strip()}, timeout=30)
        if resp.status_code != 200:
            return f"**Error starting conversation** (HTTP {resp.status_code}):\n\n```\n{resp.text[:2000]}\n```"

        data = resp.json()
        conversation_id = data.get("conversation_id")
        message_id = data.get("message_id")

        if not conversation_id or not message_id:
            return f"**Unexpected response**:\n\n```json\n{json.dumps(data, indent=2)}\n```"

        result_url = f"{host}/api/2.0/genie/spaces/{space_id.strip()}/conversations/{conversation_id}/messages/{message_id}"

        for _ in range(60):
            time.sleep(5)
            poll = requests.get(result_url, headers=headers, timeout=30)
            if poll.status_code != 200:
                continue

            msg_data = poll.json()
            status = msg_data.get("status")

            if status == "COMPLETED":
                elapsed = time.time() - t0
                attachments = msg_data.get("attachments", [])
                parts = []
                for att in attachments:
                    if att.get("text", {}).get("content"):
                        parts.append(att["text"]["content"])
                    if att.get("query", {}).get("query"):
                        parts.append(f"**SQL Generated:**\n```sql\n{att['query']['query']}\n```")

                content = "\n\n".join(parts) if parts else json.dumps(msg_data, indent=2)
                return f"{content}\n\n---\n*Completed in {elapsed:.1f}s via Genie Space `{space_id}`*"

            if status in ("FAILED", "CANCELLED"):
                return f"**Genie query {status}**:\n\n```json\n{json.dumps(msg_data, indent=2)}\n```"

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
    font=gr.themes.GoogleFont("Inter"),
)

CSS = """
.main-header { text-align: center; margin-bottom: 0.5rem; }
.main-header h1 { font-size: 1.8rem; font-weight: 700; color: #1B3A4B; }
.main-header p  { color: #6B7280; font-size: 0.95rem; }
.prompt-box textarea { font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; }
footer { display: none !important; }
"""

with gr.Blocks(title="Genie Deep Research — Config") as app:

    gr.HTML(
        '<div class="main-header">'
        "<h1>Genie Deep Research Agent</h1>"
        "<p>Configuration & Execution</p>"
        "</div>"
    )

    all_fields = []

    with gr.Tabs():
        # ── Tab 0: Run Agent ────────────────────────────────────────────
        with gr.Tab("Run Agent", id="run"):
            gr.Markdown("### Query the Agent")
            with gr.Row():
                f_run_mode = gr.Radio(
                    choices=["Full Agent", "Direct Genie", "Serving Endpoint"],
                    value="Full Agent",
                    label="Execution Mode",
                    info="Full Agent = supervisor + parallel execution + quality check (same as notebook)",
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
            )

            def _fill_sample(sample):
                return sample if sample else gr.update()

            f_sample.change(fn=_fill_sample, inputs=[f_sample], outputs=[f_query])

            btn_run = gr.Button("Run", variant="primary", size="lg")
            f_output = gr.Markdown(label="Agent Response", value="*Response will appear here...*")

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

        # ── Tab 1: Connection & Infrastructure ──────────────────────────
        with gr.Tab("Connection & Infrastructure"):
            gr.Markdown("### Databricks Connection")
            with gr.Row():
                f_catalog = gr.Textbox(label="Catalog", placeholder="e.g. bfl_std_lake")
                f_schema = gr.Textbox(label="Schema", placeholder="e.g. digital360_preprod")
                f_model = gr.Textbox(label="Model", placeholder="e.g. bajaj_funnel_analysis")
            with gr.Row():
                f_workspace_url = gr.Textbox(label="Workspace URL", placeholder="https://...")
                f_sql_warehouse_id = gr.Textbox(label="SQL Warehouse ID", placeholder="e.g. 4b9b953939869799")
            f_mlflow_experiment = gr.Textbox(label="MLflow Experiment Name", placeholder="e.g. bajaj_research")

            gr.Markdown("### Agent Settings")
            with gr.Row():
                f_agent_name = gr.Textbox(label="Agent Name", placeholder="e.g. journey-kpi-reasoning-agent")
                f_max_messages = gr.Number(label="Max Conversation Messages", value=7, precision=0, minimum=1, maximum=50)

            gr.Markdown("### LLM Configuration")
            with gr.Row():
                f_endpoint = gr.Textbox(label="LLM Endpoint Name", placeholder="e.g. databricks-claude-3-7-sonnet")
                f_temperature = gr.Slider(label="Temperature", minimum=0.0, maximum=1.0, step=0.05, value=0.1)

        # ── Tab 2: Genie Agent ──────────────────────────────────────────
        with gr.Tab("Genie Agent"):
            gr.Markdown("### Genie Space Configuration")
            f_genie_space_id = gr.Textbox(label="Genie Space ID", placeholder="e.g. 01f0d6ff25da...")
            f_genie_description = gr.Textbox(
                label="Genie Agent Description",
                lines=10,
                placeholder="Describe the data and capabilities available to the Genie agent...",
                elem_classes=["prompt-box"],
            )

        # ── Tab 3: Parallel Executor ────────────────────────────────────
        with gr.Tab("Parallel Executor"):
            gr.Markdown("### Parallel Executor Agent")
            f_parallel_description = gr.Textbox(
                label="Parallel Executor Description",
                lines=8,
                placeholder="Describe the parallel execution capabilities...",
                elem_classes=["prompt-box"],
            )

        # ── Tab 4: Supervisor & Prompts ─────────────────────────────────
        with gr.Tab("Supervisor & Prompts"):
            gr.Markdown("### Supervisor Settings")
            f_max_iterations = gr.Number(label="Max Supervisor Iterations", value=6, precision=0, minimum=1, maximum=20)

            gr.Markdown("### System Prompt")
            f_system_prompt = gr.Textbox(
                label="System Prompt",
                lines=20,
                placeholder="The core supervisor system prompt with HITL workflow...",
                elem_classes=["prompt-box"],
            )

            gr.Markdown("### Research Prompt")
            f_research_prompt = gr.Textbox(
                label="Research Prompt",
                lines=20,
                placeholder="Routing decision criteria with HITL support...",
                elem_classes=["prompt-box"],
            )

            gr.Markdown("### Final Answer Prompt")
            f_final_answer_prompt = gr.Textbox(
                label="Final Answer Prompt",
                lines=15,
                placeholder="Instructions for generating the final answer...",
                elem_classes=["prompt-box"],
            )

            gr.Markdown("### Post-Worker Routing Prompt")
            f_post_worker_routing_prompt = gr.Textbox(
                label="Post-Worker Routing Prompt",
                lines=10,
                placeholder="Lightweight routing prompt used after workers report back...",
                elem_classes=["prompt-box"],
            )

            gr.Markdown("### Quality Check Prompt")
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

    # ── Action Buttons ──────────────────────────────────────────────────
    with gr.Row():
        btn_load = gr.Button("Load Config", variant="secondary", size="lg")
        btn_save = gr.Button("Save Config", variant="primary", size="lg")
        btn_export = gr.DownloadButton("Export as YAML", variant="secondary", size="lg")

    btn_load.click(fn=load_config, inputs=[], outputs=all_fields)
    btn_save.click(fn=save_config, inputs=all_fields, outputs=[])
    btn_export.click(fn=export_yaml, inputs=all_fields)

    def _load_all():
        """Load config and also populate the Run tab's Genie Space ID."""
        vals = load_config()
        genie_space = vals[10]  # index of genie space_id in the tuple
        return (*vals, genie_space)

    app.load(fn=_load_all, inputs=[], outputs=[*all_fields, f_run_space_id])


if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "8000")),
        theme=THEME,
        css=CSS,
    )
