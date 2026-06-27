"""Agent 结果可视化 Dashboard。

依赖：
runs/latest/summary.json
runs/latest/iteration_xx.json
runs/latest/traces/
runs/latest/experience/
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "runs"


MAIN_METRIC = "Accepted"
AUX_METRICS = ["baseline", "Candidate", "Accepted"]
DETAIL_METRICS = ["function_match", "argument_match", "order_match", "json_valid"]

METRIC_LABELS = {
    "baseline": "Before Update",
    "Candidate": "Candidate Update",
    "Accepted": "Accepted/Deployed",
    "function_match": "Function match",
    "argument_match": "Argument match",
    "order_match": "Order match",
    "json_valid": "JSON valid",
}


def read_json(path: str | Path) -> Any:
    return pd.read_json(path, typ="series").to_dict() if False else __import__("json").loads(Path(path).read_text(encoding="utf-8"))


def find_run_dirs(root: Path = RUNS_DIR) -> list[Path]:
    if not root.exists():
        return []
    runs = [path for path in root.iterdir() if path.is_dir() and (path / "summary.json").exists()]
    return sorted(runs, key=lambda path: path.stat().st_mtime, reverse=True)


def resolve_artifact_path(path_text: str | None) -> Path | None:
    if not path_text:
        return None
    path = Path(path_text)
    if path.exists():
        return path
    parts = list(path.parts)
    if "runs" in parts:
        candidate = ROOT / Path(*parts[parts.index("runs") :])
        if candidate.exists():
            return candidate
    return None


def reports_to_frame(summary: list[dict], run_name: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if summary:
        initial_details = summary[0].get("gate_details", {}) or {}
        initial_baseline = initial_details.get("before_exact_match_rate")
        rows.append(
            {
                "run": run_name,
                "iteration": None,
                "round": 0,
                "round_label": "Initial",
                "accepted": False,
                "decision": "initial",
                "baseline": initial_baseline,
                "Candidate": None,
                "Accepted": initial_baseline,
                "function_match": None,
                "argument_match": None,
                "order_match": None,
                "json_valid": None,
                "bad_cases": None,
                "success_cases": None,
                "new_rules": 0,
                "candidate_fewshots": None,
                "improved_cases": None,
                "regressed_cases": None,
                "changed_cases": None,
                "avg_latency_seconds": None,
                "avg_prompt_chars": None,
                "gate_reason": None,
            }
        )
    deployed_so_far: float | None = None
    for report in summary:
        details = report.get("gate_details", {}) or {}
        regression_before = details.get("regression_before", {}) or {}
        regression_after = details.get("regression_after", {}) or {}
        iteration = report.get("iteration")
        deployed = details.get("deployed_exact_match_rate")
        if deployed is None:
            deployed = regression_after.get("exact_match_rate") if report.get("accepted") else regression_before.get("exact_match_rate")
        if deployed is None:
            deployed = deployed_so_far
        deployed_so_far = deployed
        diff = details.get("regression_diff", {}) or {}
        rows.append(
            {
                "run": run_name,
                "iteration": iteration,
                "round": (iteration + 1) if iteration is not None else None,
                "round_label": f"Round {iteration + 1}" if iteration is not None else None,
                "accepted": bool(report.get("accepted")),
                "decision": details.get("decision"),
                "baseline": details.get("before_exact_match_rate") if details.get("before_exact_match_rate") is not None else regression_before.get("exact_match_rate"),
                "Candidate": details.get("after_exact_match_rate") if details.get("after_exact_match_rate") is not None else regression_after.get("exact_match_rate"),
                "Accepted": deployed,
                "function_match": regression_after.get("function_match_rate") or regression_before.get("function_match_rate"),
                "argument_match": regression_after.get("argument_match_rate") or regression_before.get("argument_match_rate"),
                "order_match": regression_after.get("order_match_rate") or regression_before.get("order_match_rate"),
                "json_valid": regression_after.get("json_valid_rate") or regression_before.get("json_valid_rate"),
                "bad_cases": details.get("bad_case_count"),
                "success_cases": details.get("success_case_count"),
                "new_rules": len(details.get("new_rules", []) or []),
                "candidate_fewshots": details.get("candidate_fewshot_count"),
                "improved_cases": diff.get("improved_count"),
                "regressed_cases": diff.get("regressed_count"),
                "changed_cases": diff.get("changed_count"),
                "avg_latency_seconds": regression_after.get("avg_latency_seconds") or regression_before.get("avg_latency_seconds"),
                "avg_prompt_chars": regression_after.get("avg_prompt_chars") or regression_before.get("avg_prompt_chars"),
                "gate_reason": report.get("gate_reason"),
            }
        )
    return pd.DataFrame(rows)


def line_chart(df: pd.DataFrame, metrics: list[str], title: str, y_title: str = "Score") -> alt.Chart:
    chart_df = df[["round", "round_label", *metrics]].melt(["round", "round_label"], var_name="metric", value_name="value").dropna()
    chart_df["metric"] = chart_df["metric"].map(METRIC_LABELS).fillna(chart_df["metric"])
    if chart_df.empty:
        y_min, y_max = 0.0, 1.0
    else:
        y_min = float(chart_df["value"].min())
        y_max = float(chart_df["value"].max())
        pad = 0.15 if y_min == y_max else (y_max - y_min) * 0.15
        y_min -= pad
        y_max += pad
        y_min = max(0.0, y_min)
    show_legend = chart_df["metric"].nunique() > 1 if not chart_df.empty else False
    return (
        alt.Chart(chart_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("round:O", title="Round"),
            y=alt.Y("value:Q", title=y_title, scale=alt.Scale(domain=[y_min, y_max])),
            color=alt.Color("metric:N", title="Metric", legend=alt.Legend() if show_legend else None),
            tooltip=["round_label:N", "metric:N", alt.Tooltip("value:Q", format=".4f")],
        )
        .properties(height=320, title=title)
    )


def bar_chart(df: pd.DataFrame, metrics: list[str], title: str) -> alt.Chart:
    chart_df = df[["round", "round_label", *metrics]].melt(["round", "round_label"], var_name="signal", value_name="count").dropna()
    return (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x=alt.X("round:O", title="Round"),
            y=alt.Y("count:Q", title="Count"),
            color=alt.Color("signal:N", title="Signal"),
            tooltip=["round_label:N", "signal:N", "count:Q"],
        )
        .properties(height=260, title=title)
    )


def load_trace_records(trace_path: str | None) -> pd.DataFrame:
    path = resolve_artifact_path(trace_path)
    if path is None:
        return pd.DataFrame()
    payload = read_json(path)
    rows: list[dict[str, Any]] = []
    for record in payload.get("records", []) or []:
        score = record.get("score", {}) or {}
        rows.append(
            {
                "id": record.get("id"),
                "exact_match": score.get("exact_match"),
                "function_match": score.get("function_match"),
                "argument_match": score.get("argument_match"),
                "json_valid": score.get("json_valid"),
                "errors": ", ".join(score.get("error_types", []) or []),
                "latency_seconds": record.get("latency_seconds"),
                "prompt_chars": record.get("prompt_chars"),
                "user_query": record.get("user_query"),
                "gold_calls": record.get("gold_calls"),
                "predicted_calls": record.get("predicted_calls"),
                "raw_output": record.get("raw_output"),
            }
        )
    return pd.DataFrame(rows)


st.set_page_config(page_title="Agent Harness Dashboard", layout="wide")
st.title("Agent Dashboard")
st.caption("Read-only view of saved runs. Run experiments with `run_demo.py`; use this page for presentation.")

run_dirs = find_run_dirs()
if not run_dirs:
    st.warning("No saved runs found. Run `python run_demo.py ...` first, then refresh this page.")
    st.stop()

with st.sidebar:
    st.header("Run Browser")
    max_runs = st.slider("Recent runs to list", 1, max(2, min(10, len(run_dirs))), min(5, len(run_dirs)))
    selected_run = st.selectbox("Run", run_dirs[:max_runs], format_func=lambda p: str(p.relative_to(ROOT)))

summary = read_json(selected_run / "summary.json")
run_df = reports_to_frame(summary, str(selected_run.relative_to(ROOT)))

st.subheader("1. Overall BFCLv3 simple-multiple metric improvement")
initial_em = run_df.iloc[0]["Accepted"]
best_em = run_df["Accepted"].max()
final_em = run_df.iloc[-1]["Accepted"]
accepted_updates = int(run_df["accepted"].sum())
total_improvement = best_em - initial_em if pd.notna(best_em) and pd.notna(initial_em) else None

cols = st.columns(5)
cols[0].metric("Initial Exact Match", f"{initial_em:.4f}" if pd.notna(initial_em) else "N/A")
cols[1].metric("Best Deployed Exact Match", f"{best_em:.4f}" if pd.notna(best_em) else "N/A")
cols[2].metric("Final Deployed Exact Match", f"{final_em:.4f}" if pd.notna(final_em) else "N/A")
cols[3].metric("Accepted Updates", accepted_updates)
cols[4].metric("Total Improvement", f"{total_improvement:+.4f}" if total_improvement is not None else "N/A")

st.altair_chart(line_chart(run_df, [MAIN_METRIC], "Deployed exact-match trend"), use_container_width=True)

st.subheader("2. Update details")
left, right = st.columns([2, 1])
with left:
    st.altair_chart(line_chart(run_df, AUX_METRICS, "Before / candidate / deployed by round"), use_container_width=True)
with right:
    aux_df = run_df[["round_label", "baseline", "Candidate", "Accepted"]].copy()
    aux_df.columns = ["Round", "Before Update", "Candidate Update", "Accepted/Deployed"]
    st.dataframe(aux_df, use_container_width=True, hide_index=True)

st.subheader("3. Detail metric changes")
st.altair_chart(line_chart(run_df, DETAIL_METRICS, "Function / argument / order / JSON-valid metrics"), use_container_width=True)

with st.expander("Operational metrics", expanded=False):
    op_df = run_df[["round", "round_label", "avg_latency_seconds", "avg_prompt_chars"]].copy()
    st.dataframe(op_df, use_container_width=True, hide_index=True)
    st.line_chart(op_df.set_index("round")[["avg_latency_seconds", "avg_prompt_chars"]])

st.subheader("4. Learning signals")
st.altair_chart(
    bar_chart(
        run_df[run_df["round"] > 0],
        ["success_cases", "bad_cases", "new_rules", "candidate_fewshots", "improved_cases", "regressed_cases"],
        "Learning-loop signals by round",
    ),
    use_container_width=True,
)

st.subheader("5. Iteration details")
iteration_options = [report.get("iteration") for report in summary]
selected_iteration = st.selectbox(
    "Round",
    iteration_options,
    index=len(iteration_options) - 1,
    format_func=lambda it: f"Round {it + 1}" if it is not None else "N/A",
)
report = next(item for item in summary if item.get("iteration") == selected_iteration)
details = report.get("gate_details", {}) or {}

cols = st.columns(4)
cols[0].metric("Decision", details.get("decision", "N/A"))
cols[1].metric("Improvement", details.get("improvement", "N/A"))
cols[2].metric("Regression drop", details.get("regression_drop", "N/A"))
cols[3].metric("Changed cases", (details.get("regression_diff", {}) or {}).get("changed_count", "N/A"))
st.info(report.get("gate_reason"))

with st.expander("Skill changes", expanded=True):
    diff = details.get("candidate_skill_diff", "")
    if diff:
        st.code(diff, language="diff")
    else:
        st.write("No Skill diff was proposed in this iteration.")
    st.markdown("**New rules**")
    st.json(details.get("new_rules", []))

with st.expander("Few-shot candidates", expanded=False):
    exp_path = resolve_artifact_path(details.get("experience_path"))
    if exp_path:
        experience = read_json(exp_path)
        st.json(experience.get("candidate_fewshots", []))
    else:
        st.write("No experience file found.")

st.subheader("6. Agent trajectory")
trace_choice = st.radio(
    "Trace view",
    ["optimize", "regression_baseline", "regression_candidate"],
    horizontal=True,
)
trace_key = {
    "optimize": "optimize_trace",
    "regression_baseline": "regression_before_trace",
    "regression_candidate": "regression_after_trace",
}[trace_choice]
trace_df = load_trace_records(details.get(trace_key))
if trace_df.empty:
    st.write("No trace available for this view.")
else:
    compact_cols = ["id", "exact_match", "function_match", "argument_match", "json_valid", "errors", "latency_seconds", "user_query"]
    st.dataframe(trace_df[compact_cols].head(80), use_container_width=True)
    selected_case = st.selectbox("Inspect case", trace_df["id"].tolist())
    case = trace_df[trace_df["id"] == selected_case].iloc[0].to_dict()
    left, right = st.columns(2)
    with left:
        st.markdown("**Gold calls**")
        st.json(case.get("gold_calls"))
    with right:
        st.markdown("**Predicted calls**")
        st.json(case.get("predicted_calls"))
    with st.expander("Raw model output", expanded=False):
        st.code(str(case.get("raw_output", "")))

with st.expander("Raw report JSON", expanded=False):
    st.json(report)
