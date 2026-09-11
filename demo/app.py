import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from demo.inference import (
    ACTION_NAMES,
    SCENARIOS,
    risk_label,
    score_portfolio,
    synthetic_portfolio,
    verify_frozen_artifact,
)

STATUS_LABELS = {
    -2: "-2 · no consumption",
    -1: "-1 · paid duly",
    0: "0 · current / revolving",
    1: "1 · one month delayed",
    2: "2 · two months delayed",
    3: "3 · three months delayed",
    4: "4 · four months delayed",
    5: "5 · five months delayed",
    6: "6 · six months delayed",
    7: "7 · seven months delayed",
    8: "8 · eight months delayed",
}

st.set_page_config(
    page_title="AutoCollections Research Demo",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="auto",
)
st.markdown(
    """
    <style>
    :root {
        --canvas: #061525;
        --surface: #0a1d31;
        --surface-2: #0d233a;
        --border: #28445f;
        --text: #edf4ff;
        --muted: #a8b8cb;
        --teal: #2dc7b5;
        --coral: #f07465;
        --amber: #e8b75f;
    }
    .stApp, [data-testid="stAppViewContainer"] { background: var(--canvas); color: var(--text); }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] { background: var(--surface); border-right: 1px solid var(--border); }
    [data-testid="stSidebar"] * { color: var(--text); }
    .block-container { max-width: 1500px; padding-top: 1.2rem; padding-bottom: 3rem; }
    h1, h2, h3, p, label, [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
        font-family: "IBM Plex Sans", Inter, system-ui, sans-serif;
    }
    h1 { letter-spacing: -0.025em; font-weight: 600; }
    h2, h3 { letter-spacing: -0.015em; font-weight: 550; }
    p { color: var(--muted); line-height: 1.6; }
    div[data-testid="stMetric"] { background: transparent; border-left: 2px solid var(--border); padding-left: 1rem; }
    div[data-testid="stMetricValue"] { color: var(--text); font-size: 1.65rem; }
    .safety-banner { border: 1px solid var(--amber); padding: 0.9rem 1rem; margin: 0.5rem 0 1.4rem; }
    .safety-banner strong { color: #ffd48a; }
    .safety-banner span { color: var(--muted); margin-left: 1rem; }
    .result-panel { border: 1px solid var(--border); background: var(--surface); padding: 1.1rem 1.25rem; margin: 0.75rem 0 1rem; }
    .result-panel .action { color: var(--teal); font-size: 1.55rem; font-weight: 650; letter-spacing: 0.01em; }
    .result-panel .human { color: var(--coral); }
    .result-panel .no-contact { color: var(--amber); }
    .section-rule { border-top: 1px solid var(--border); margin: 1.5rem 0; }
    .workflow { color: var(--text); font-size: 1rem; word-spacing: 0.35rem; padding: 1rem 0; }
    .frozen-note { border-left: 3px solid var(--coral); padding: 0.7rem 1rem; color: var(--muted); }
    .stButton > button, [data-testid="stFormSubmitButton"] button {
        background: #148f85; color: white; border: 1px solid #28b6a8; border-radius: 4px;
        min-height: 44px; font-weight: 600; transition: background-color 180ms ease;
    }
    .stButton > button:hover, [data-testid="stFormSubmitButton"] button:hover { background: #197f78; }
    .stButton > button:focus-visible, [data-testid="stFormSubmitButton"] button:focus-visible {
        outline: 3px solid #8be9df; outline-offset: 2px;
    }
    [data-testid="stDataFrame"] { border: 1px solid var(--border); }
    @media (max-width: 760px) {
        .block-container { padding-left: 1rem; padding-right: 1rem; }
        .safety-banner span { display: block; margin: 0.4rem 0 0; }
        .workflow { line-height: 2; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def apply_scenario() -> None:
    profile = SCENARIOS[st.session_state.scenario_name]
    for name, value in profile.items():
        st.session_state[f"field_{name}"] = value


def initialize_inputs() -> None:
    if "scenario_name" not in st.session_state:
        st.session_state.scenario_name = next(iter(SCENARIOS))
    profile = SCENARIOS[st.session_state.scenario_name]
    for name, value in profile.items():
        st.session_state.setdefault(f"field_{name}", value)
    st.session_state.setdefault("evaluated_profile", profile.copy())


def profile_form() -> dict:
    st.selectbox(
        "Synthetic scenario",
        list(SCENARIOS),
        key="scenario_name",
        on_change=apply_scenario,
    )
    with st.form("scenario_form"):
        st.number_input(
            "Credit limit (synthetic units)",
            min_value=1_000,
            step=1_000,
            key="field_LIMIT_BAL",
        )
        st.number_input(
            "Current bill balance",
            min_value=0,
            step=500,
            key="field_BILL_AMT1",
        )
        st.number_input(
            "Recent payment",
            min_value=0,
            step=500,
            key="field_PAY_AMT1",
        )
        st.selectbox(
            "Recent repayment status",
            list(STATUS_LABELS),
            format_func=STATUS_LABELS.get,
            key="field_PAY_0",
        )
        with st.expander("Six-month history", expanded=False):
            st.caption("Older monthly values remain synthetic and manually editable.")
            for month, status_month in zip(range(2, 7), (2, 3, 4, 5, 6), strict=True):
                st.markdown(f"**Month {month}**")
                bill, payment, status = st.columns(3)
                bill.number_input(
                    "Bill",
                    min_value=0,
                    step=500,
                    key=f"field_BILL_AMT{month}",
                )
                payment.number_input(
                    "Payment",
                    min_value=0,
                    step=500,
                    key=f"field_PAY_AMT{month}",
                )
                status.selectbox(
                    "Status",
                    list(STATUS_LABELS),
                    format_func=STATUS_LABELS.get,
                    key=f"field_PAY_{status_month}",
                )
        submitted = st.form_submit_button("Evaluate scenario", width="stretch")
    profile = {name: st.session_state[f"field_{name}"] for name in SCENARIOS[next(iter(SCENARIOS))]}
    if submitted:
        st.session_state.evaluated_profile = profile
    return st.session_state.evaluated_profile


def safety_banner() -> None:
    st.markdown(
        """
        <div class="safety-banner">
          <strong>Research simulation — not a real collections recommendation system.</strong>
          <span>Business values are simulated_inr_units, not realized savings.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def decision_lab() -> None:
    with st.sidebar:
        st.header("Scenario and inputs")
        st.caption("Select a synthetic profile, edit approved financial fields, then evaluate.")
        profile = profile_form()
        st.divider()
        st.caption(
            "Synthetic data only. No real customer records, demographic inputs, or production recommendations."
        )

    portfolio = synthetic_portfolio(profile)
    scored = score_portfolio(portfolio)
    selected = scored.iloc[0]
    top_count = int(scored["inside_top_35"].sum())

    st.header("Individual analysis")
    st.caption(
        "The selected synthetic profile is evaluated inside a fixed 24-account synthetic portfolio."
    )
    metric_columns = st.columns(5)
    metric_columns[0].metric(
        "Predicted default probability", f"{selected.predicted_probability:.1%}"
    )
    metric_columns[1].metric("Exposure", f"{selected.exposure:,.0f}")
    metric_columns[2].metric("Utilization", f"{selected.utilization:.1%}")
    metric_columns[3].metric(
        "Recent payment / bill", f"{selected.recent_payment_to_bill_ratio:.2f}"
    )
    metric_columns[4].metric("Human-allocation score", f"{selected.allocation_score:,.1f}")

    action_class = {
        "HUMAN_ESCALATION": "human",
        "NO_CONTACT": "no-contact",
    }.get(selected.action, "")
    st.markdown(
        f"""
        <div class="result-panel">
          <div>Portfolio-ranked treatment</div>
          <div class="action {action_class}">{selected.action}</div>
          <p>Rank {int(selected["rank"])} of {len(scored)} by the frozen allocation score.
          The top {top_count} accounts are selected for human escalation.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    risk = risk_label(float(selected.predicted_probability))
    exposure_label = (
        "None"
        if selected.exposure == 0
        else ("High" if selected.exposure >= 100_000 else "Moderate")
    )
    payment_label = (
        "Low"
        if selected.recent_payment_to_bill_ratio < 0.25
        else "Strong"
        if selected.recent_payment_to_bill_ratio >= 0.75
        else "Moderate"
    )
    if selected.action == "HUMAN_ESCALATION":
        reason = "The profile ranks inside the portfolio’s top 35% under the frozen value-aware allocation rule."
    elif selected.action == "NO_CONTACT":
        reason = "Current positive exposure is below 1,000 synthetic units, triggering the frozen no-contact rule."
    else:
        reason = "The profile remains outside the portfolio’s human-capacity allocation and has material exposure."
    st.subheader("Why this decision?")
    st.markdown(
        f"**Predicted risk:** {risk}  \n"
        f"**Exposure:** {exposure_label}  \n"
        f"**Credit utilization:** {selected.utilization:.1%}  \n"
        f"**Recent repayment ratio:** {payment_label}  \n\n"
        f"{reason}"
    )
    st.info(
        "Human escalation is not an isolated-account threshold. It depends on ranking the account "
        "within a portfolio and applying the assumed 35% capacity ceiling."
    )

    st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
    ranking, distribution = st.columns([2.2, 1], gap="large")
    with ranking:
        st.subheader("Portfolio ranking · 24 synthetic accounts")
        display = scored.sort_values("rank").copy()
        display["risk"] = display["predicted_probability"].map(lambda value: f"{value:.1%}")
        display["allocation score"] = display["allocation_score"].map(lambda value: f"{value:,.1f}")
        display["top 35%"] = display["inside_top_35"].map({True: "Yes", False: "No"})
        st.dataframe(
            display[["rank", "scenario", "risk", "allocation score", "action", "top 35%"]],
            hide_index=True,
            width="stretch",
            height=470,
        )
    with distribution:
        st.subheader("Treatment distribution")
        counts = scored["action"].value_counts().reindex(ACTION_NAMES.values(), fill_value=0)
        st.bar_chart(counts[counts > 0].rename("accounts"), color="#2dc7b5", height=320)
        st.caption(" · ".join(f"{action}: {count}" for action, count in counts.items()))
        st.caption(
            f"Human capacity selects {top_count} of {len(scored)} accounts in this synthetic portfolio."
        )


def research_summary() -> None:
    final = json.loads((ROOT / "reports/final_hidden_evaluation.json").read_text())
    st.header("Research summary")
    st.caption("Frozen results from 50 autonomous experiments. No post-hidden tuning.")
    metrics = st.columns(4)
    metrics[0].metric("Autonomous experiments", "50")
    metrics[1].metric("KEEP decisions", "15")
    metrics[2].metric("Validation utility / account", "1007.5752")
    metrics[3].metric("Hidden utility / account", "1022.0782")
    st.markdown(
        "**Hidden B4:** 791.1042 utility/account  ·  "
        "**Hidden improvement:** +29.1964%  ·  "
        "**Paired uplift:** +230.9740/account  ·  "
        "**95% CI:** [181.1610, 285.8786]"
    )
    st.caption(final["disclaimer"])

    st.image(
        ROOT / "reports/figures/research_trajectory.png",
        caption="Protected validation utility and KEEP/REVERT history.",
        width="stretch",
    )
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        st.subheader("AutoResearch workflow")
        st.markdown(
            '<div class="workflow">Hypothesis → modify train.py → protected evaluation → '
            "KEEP / REVERT → Git commit → next experiment</div>",
            unsafe_allow_html=True,
        )
        st.caption(
            "The research agent could edit train.py only. Simulator economics, feasibility, "
            "splits, validation outcomes, and evaluator code were protected."
        )
    with right:
        st.subheader("Final study result")
        st.image(
            ROOT / "reports/figures/final_performance.png",
            caption="Frozen validation and one-time hidden-test comparison.",
            width="stretch",
        )

    st.subheader("Limitations")
    st.markdown(
        """
        - The public default dataset is a proxy, not collections-treatment data.
        - Treatment responses are simulated; no causal effects are estimated.
        - `simulated_inr_units` are not realized savings or revenue.
        - The 35% human capacity is an assumed operational constraint.
        - Demographic variables were excluded from model and policy inputs.
        - PAYMENT_PLAN_REVIEW was not selected by the final learned policy.
        - This is research-process-isolated software, not a hostile-code sandbox or live collections system.
        """
    )
    st.markdown(
        '<div class="frozen-note"><strong>Phase 7 is consumed.</strong> The demo reads only frozen '
        "aggregate results and cannot invoke the final evaluator.</div>",
        unsafe_allow_html=True,
    )


verify_frozen_artifact()
initialize_inputs()
st.title("AutoCollections Research Demo")
st.caption("Synthetic profiles. Frozen model. Portfolio-aware treatment allocation.")
mode = st.segmented_control(
    "View",
    ["Decision lab", "Research summary"],
    default="Decision lab",
    label_visibility="collapsed",
)
safety_banner()
if mode == "Research summary":
    research_summary()
else:
    decision_lab()
