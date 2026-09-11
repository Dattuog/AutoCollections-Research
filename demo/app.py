import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from demo.inference import (
    ACTION_NAMES,
    SCENARIOS,
    score_portfolio,
    synthetic_portfolio,
    verify_frozen_artifact,
)

STATUS_LABELS = {
    -2: "No consumption",
    -1: "Paid duly",
    0: "Current / revolving",
    1: "One month delayed",
    2: "Two months delayed",
    3: "Three months delayed",
    4: "Four months delayed",
    5: "Five months delayed",
    6: "Six months delayed",
    7: "Seven months delayed",
    8: "Eight months delayed",
}
NAVIGATION = (("decision", "Decision"), ("research", "Research"), ("experiments", "Experiments"))

st.set_page_config(
    page_title="AutoCollections Research Lab",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="auto",
)
st.markdown(
    """
    <style>
    :root {
        --ink: #20231f;
        --ink-soft: #343832;
        --nav: #1d211e;
        --canvas: #efeee9;
        --surface: #f8f7f3;
        --surface-soft: #ebe6da;
        --line: #d5d1c7;
        --muted: #74766f;
        --green: #316652;
        --green-soft: #dfe8e3;
        --orange: #a95e2b;
        --orange-soft: #f2dfcf;
    }

    *, *::before, *::after { box-sizing: border-box; }

    html, body, [class*="css"], [data-testid="stAppViewContainer"] {
        font-family: Arial, Helvetica, sans-serif;
    }
    .stApp, [data-testid="stAppViewContainer"] {
        background: var(--canvas);
        color: var(--ink);
    }
    [data-testid="stHeader"] { background: transparent; pointer-events: none; }
    [data-testid="stHeaderActionElements"] { display: none; }
    [data-testid="stToolbar"] { display: none; }
    [data-testid="stMainMenu"] { display: none; }
    [data-testid="stDecoration"] { display: none; }
    [data-testid="stMainBlockContainer"] {
        max-width: none;
        padding: 1.375rem .75rem 3rem 2.5rem;
    }
    .stHeading a { display: none; }

    [data-testid="stSidebar"] {
        width: 270px !important;
        min-width: 270px !important;
        background: var(--nav);
        border-right: 0;
    }
    [data-testid="stSidebarContent"] { background: var(--nav); }
    [data-testid="stSidebarHeader"] { height: 0; min-height: 0; }
    [data-testid="stSidebarUserContent"] {
        padding: 1.25rem .9rem 2rem;
        height: 100vh;
    }
    [data-testid="stSidebarCollapseButton"] { display: none; }
    .brand {
        color: #f5f2e9;
        font: 700 .88rem Georgia, serif;
        letter-spacing: .01em;
        margin-bottom: .28rem;
    }
    .brand-sub {
        color: #969990;
        font-size: .62rem;
        letter-spacing: .08em;
        text-transform: uppercase;
        padding-bottom: 1.75rem;
        border-bottom: 1px solid #444740;
    }
    .lab-nav { margin-top: 1.2rem; }
    .lab-nav a {
        display: block;
        color: #969990;
        font-size: .75rem;
        letter-spacing: .04em;
        padding: .72rem 0;
        text-decoration: none;
        text-transform: uppercase;
        transition: color 160ms ease;
    }
    .lab-nav a:hover, .lab-nav a:focus-visible { color: #f7f4eb; }
    .lab-nav a:focus-visible { outline: 2px solid #f7f4eb; outline-offset: 4px; }
    .lab-nav a.active {
        color: #ffffff;
        font-family: Georgia, serif;
        font-weight: 700;
    }
    .lab-nav .index { display: inline-block; width: 1.55rem; }
    .sidebar-study {
        position: fixed;
        left: 2.1rem;
        bottom: 2.2rem;
        width: 202px;
        color: #f1eee6;
        font-size: .78rem;
        line-height: 1.75;
    }
    .sidebar-study .label {
        color: #969990;
        font-size: .62rem;
        font-weight: 700;
        letter-spacing: .08em;
        margin-bottom: .45rem;
        text-transform: uppercase;
    }
    .sidebar-study .privacy {
        color: #969990;
        font-size: .64rem;
        margin-top: 1.55rem;
    }
    .mobile-nav { display: none; }

    .page-header {
        align-items: start;
        display: flex;
        justify-content: space-between;
        margin-bottom: 1.2rem;
    }
    .breadcrumb {
        color: #65675f;
        font: 700 .72rem Georgia, serif;
        letter-spacing: .035em;
        margin-bottom: 1.15rem;
    }
    h1.page-title {
        color: var(--ink);
        font-family: Arial, Helvetica, sans-serif !important;
        font-size: 2.22rem !important;
        font-weight: 750 !important;
        letter-spacing: -.025em;
        line-height: 1.05 !important;
        margin: 0 0 .25rem;
        padding: 0 !important;
    }
    .page-subtitle {
        color: var(--muted);
        font-size: .91rem;
        margin: 0;
    }
    .model-stamp {
        background: rgba(248,247,243,.62);
        border: 1px solid var(--line);
        color: #6d6f68;
        font-size: .7rem;
        letter-spacing: .025em;
        min-width: 296px;
        padding: 1rem 1.2rem;
        text-align: left;
    }

    .kpi-grid {
        display: grid;
        gap: 1.05rem;
        grid-template-columns: repeat(3, minmax(0, 1fr)) 1.55fr;
        margin-bottom: 1.8rem;
    }
    .kpi-card {
        background: rgba(248,247,243,.76);
        border: 1px solid var(--line);
        border-radius: 4px;
        min-height: 94px;
        padding: .75rem 1rem;
    }
    .kpi-card.boundary { background: var(--surface-soft); border-color: #cfc5b3; }
    .kpi-label, .section-label {
        color: #70736b;
        font-size: .61rem;
        font-weight: 700;
        letter-spacing: .045em;
        text-transform: uppercase;
    }
    .kpi-value {
        color: var(--ink);
        font: 700 1.9rem Georgia, serif;
        letter-spacing: .015em;
        margin-top: .6rem;
    }
    .kpi-card.boundary .kpi-value { font-size: 1.06rem; margin-top: .5rem; }
    .boundary-note { color: #6f6a60; font-size: .59rem; line-height: 1.2; margin-top: .15rem; }

    div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] h3#scenario-configuration) {
        background: rgba(248,247,243,.82);
        border: 1px solid var(--line);
        border-radius: 4px;
        min-height: 665px;
        padding: 1.05rem .75rem .95rem;
    }
    h3#scenario-configuration {
        color: var(--ink);
        font: 700 1.05rem/1.25 Georgia, serif !important;
        letter-spacing: .01em;
        margin: 0 0 .8rem !important;
        padding: 0 !important;
    }
    [data-testid="stWidgetLabel"] p {
        color: #70736b;
        font-size: .61rem;
        font-weight: 700;
        letter-spacing: .04em;
        text-transform: uppercase;
    }
    [data-testid="stNumberInputContainer"], [data-testid="stSelectbox"] div[role="group"] {
        background: transparent !important;
        border: 0 !important;
        border-bottom: 1px solid var(--line) !important;
        border-radius: 0 !important;
        box-shadow: none !important;
        min-height: 2.35rem;
    }
    [data-testid="stNumberInputContainer"] input,
    [data-testid="stSelectbox"] input,
    [data-testid="stSelectbox"] button {
        color: var(--ink) !important;
        font-size: .79rem !important;
    }
    [data-testid="stNumberInputStepDown"], [data-testid="stNumberInputStepUp"] { display: none; }
    [data-testid="stForm"] { border: 0; padding: 15px 8px; }
    [data-testid="stFormSubmitButton"] { margin-top: .45rem; }
    [data-testid="stFormSubmitButton"] button {
        background: var(--ink);
        border: 1px solid var(--ink);
        border-radius: 2px;
        color: #fff;
        font-size: .68rem;
        font-weight: 700;
        letter-spacing: .04em;
        min-height: 49px;
        text-transform: uppercase;
        transition: background-color 160ms ease;
    }
    [data-testid="stFormSubmitButton"] button:hover { background: #343832; }
    [data-testid="stFormSubmitButton"] button:focus-visible {
        outline: 3px solid #7f9f92;
        outline-offset: 2px;
    }
    .capacity-foot { color: #6f716b; font-size: .63rem; margin-top: .65rem; }

    .analysis-block { padding: .55rem 0 0 .5rem; }
    .metric-grid {
        display: grid;
        gap: 1.6rem;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        margin: 1.15rem 0 1.5rem;
    }
    .metric {
        border-bottom: 1px solid var(--line);
        padding-bottom: .45rem;
    }
    .metric-name { color: #777970; font-size: .68rem; line-height: 1.2; margin-bottom: .45rem; }
    .metric-number { color: var(--ink); font-size: 1.42rem; font-weight: 700; letter-spacing: .01em; line-height: 1.15; }
    .treatment-panel {
        background: var(--green-soft);
        border: 1px solid #afc2b8;
        border-radius: 4px;
        min-height: 138px;
        padding: 1.15rem 1.45rem;
    }
    .treatment-panel.human { background: var(--orange-soft); border-color: #d8baa2; }
    .treatment-panel.no-contact { background: var(--surface-soft); border-color: #cfc5b3; }
    .treatment-panel .eyebrow {
        color: #5b796d;
        font-size: .61rem;
        font-weight: 700;
        letter-spacing: .04em;
        text-transform: uppercase;
    }
    .treatment-panel.human .eyebrow { color: #8e552f; }
    .treatment-action {
        color: var(--green);
        font-size: 1.55rem;
        font-weight: 800;
        letter-spacing: .035em;
        margin: 1.35rem 0 .45rem;
    }
    .treatment-panel.human .treatment-action { color: var(--orange); }
    .treatment-panel.no-contact .treatment-action { color: #756c5e; }
    .treatment-reason { color: #6f726b; font-size: .76rem; line-height: 1.45; }

    .lower-grid {
        display: grid;
        gap: 1.25rem;
        grid-template-columns: minmax(0, 1.85fr) minmax(245px, 1fr);
        margin-top: 1.5rem;
    }
    .editorial-card {
        background: rgba(248,247,243,.82);
        border: 1px solid var(--line);
        border-radius: 4px;
        padding: 1.28rem;
    }
    .lower-grid .editorial-card { height: 371px; overflow: hidden; }
    .editorial-card h3 {
        color: var(--ink);
        font: 700 1.05rem Georgia, serif;
        letter-spacing: .01em;
        line-height: 1.2 !important;
        margin: 0 0 1rem !important;
        padding: 0 !important;
    }
    .rank-table { border-collapse: collapse; width: 100%; }
    .rank-table th {
        background: #ebe9e3;
        color: #6f716a;
        font-size: .69rem;
        letter-spacing: .03em;
        padding: .72rem .62rem;
        text-align: left;
    }
    .rank-table th:first-child { border-radius: 10px 0 0 10px; }
    .rank-table th:last-child { border-radius: 0 10px 10px 0; }
    .rank-table td {
        border-bottom: 1px solid #e3e0d8;
        color: #363932;
        font-size: .67rem;
        padding: .45rem .62rem;
        white-space: nowrap;
    }
    .rank-table .risk { color: var(--green); font-weight: 700; }
    .action-chip {
        border-radius: 16px;
        display: inline-block;
        font-size: .6rem;
        font-weight: 700;
        min-width: 86px;
        padding: .37rem .55rem;
        text-align: center;
    }
    .action-chip.human { background: var(--orange-soft); color: #8f4c24; }
    .action-chip.digital { background: var(--green-soft); color: var(--green); }
    .action-chip.no-contact { background: #e9e3d7; color: #6e6558; }

    .mix-chart {
        align-items: end;
        border-bottom: 1px solid #c9c6bd;
        border-left: 1px solid #c9c6bd;
        display: flex;
        gap: 12px;
        height: 170px;
        justify-content: center;
        margin: .4rem .9rem 0;
        padding: 0 .55rem;
    }
    .mix-column { align-items: center; display: flex; flex: 1; flex-direction: column; justify-content: end; height: 100%; }
    .mix-value { color: var(--ink); font-size: .68rem; font-weight: 700; margin-bottom: .42rem; }
    .mix-bar { max-width: 62px; min-height: 0; width: 100%; }
    .mix-bar.digital { background: var(--green); }
    .mix-bar.human { background: var(--orange); }
    .mix-bar.no-contact { background: #aea79a; }
    .mix-label {
        color: #6d6f68;
        font-size: .52rem;
        margin-bottom: -1.35rem;
        margin-top: .48rem;
        text-transform: uppercase;
        white-space: nowrap;
    }
    .mix-summary { color: var(--ink); font: 700 .93rem Georgia, serif; margin: 3rem 0 .15rem; }
    .mix-caption { color: #777970; font-size: .63rem; line-height: 1.45; }

    .research-copy { color: #555850; font-size: .82rem; line-height: 1.65; }
    .research-copy strong { color: var(--ink); }
    [data-testid="stImage"] { background: var(--surface); border: 1px solid var(--line); padding: .75rem; }

    @media (max-width: 1100px) {
        .kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        .lower-grid { grid-template-columns: 1fr; }
        .model-stamp { min-width: 230px; }
    }
    @media (max-width: 760px) {
        [data-testid="stMainBlockContainer"] { padding: 1rem 1rem 3rem; }
        [data-testid="stSidebar"], [data-testid="stExpandSidebarButton"] { display: none !important; }
        .mobile-nav {
            background: var(--nav);
            border-radius: 2px;
            display: flex;
            margin-bottom: 1.6rem;
            padding: .25rem;
        }
        .mobile-nav a {
            align-items: center;
            background: transparent;
            border: 0;
            border-radius: 0;
            color: #a8aaa3;
            display: flex;
            flex: 1;
            font-size: .58rem;
            justify-content: center;
            letter-spacing: .035em;
            min-height: 44px;
            padding: 0 .1rem;
            text-decoration: none;
            text-transform: uppercase;
        }
        .mobile-nav a.active {
            background: #30342f;
            color: #fff;
        }
        .mobile-nav .index { margin-right: .2rem; }
        .page-header { display: block; }
        .page-title { font-size: 2rem; }
        .model-stamp { margin-top: 1.2rem; min-width: 0; width: 100%; }
        .kpi-grid { grid-template-columns: 1fr 1fr; gap: .7rem; }
        .kpi-card { min-height: 86px; }
        .kpi-value { font-size: 1.45rem; }
        .metric-grid { grid-template-columns: 1fr 1fr; gap: 1rem; }
        .analysis-block { padding-left: 0; }
        .lower-grid { grid-template-columns: minmax(0, 1fr); }
        .editorial-card { overflow-x: auto; }
        .lower-grid .editorial-card { height: auto; }
        div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] h3#scenario-configuration) { min-height: 0; }
        .rank-table { min-width: 620px; }
        .treatment-action { font-size: 1.25rem; overflow-wrap: anywhere; }
        .sidebar-study { position: static; margin-top: 3rem; }
    }
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after { scroll-behavior: auto !important; transition: none !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def initialize_inputs() -> None:
    if "scenario_name" not in st.session_state:
        st.session_state.scenario_name = next(iter(SCENARIOS))
    profile = SCENARIOS[st.session_state.scenario_name]
    for name, value in profile.items():
        st.session_state.setdefault(f"field_{name}", value)
    st.session_state.setdefault("evaluated_profile", profile.copy())


def apply_scenario() -> None:
    profile = SCENARIOS[st.session_state.scenario_name]
    for name, value in profile.items():
        st.session_state[f"field_{name}"] = value


def navigation_links(view: str) -> str:
    return "".join(
        f'<a class="{"active" if view == name else ""}" href="/?view={name}" target="_self">'
        f'<span class="index">0{index}</span>{label}</a>'
        for index, (name, label) in enumerate(NAVIGATION, start=1)
    )


def sidebar(view: str) -> None:
    st.sidebar.markdown(
        f"""
        <div class="brand">AUTOCOLLECTIONS</div>
        <div class="brand-sub">Research lab</div>
        <nav class="lab-nav">{navigation_links(view)}</nav>
        <div class="sidebar-study">
          <div class="label">Frozen study</div>
          <div>50 experiments</div>
          <div>15 KEEP decisions</div>
          <div class="privacy">No real customer data</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_header(active_view: str, breadcrumb: str, title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <nav class="mobile-nav">{navigation_links(active_view)}</nav>
        <header class="page-header">
          <div>
            <div class="breadcrumb">{breadcrumb}</div>
            <h1 class="page-title">{title}</h1>
            <p class="page-subtitle">{subtitle}</p>
          </div>
          <div class="model-stamp">exp_039 · deterministic reconstruction</div>
        </header>
        """,
        unsafe_allow_html=True,
    )


def study_cards() -> None:
    st.markdown(
        """
        <section class="kpi-grid">
          <div class="kpi-card"><div class="kpi-label">Validation utility</div><div class="kpi-value">1007.58</div></div>
          <div class="kpi-card"><div class="kpi-label">Hidden utility</div><div class="kpi-value">1022.08</div></div>
          <div class="kpi-card"><div class="kpi-label">Hidden uplift</div><div class="kpi-value">+29.2%</div></div>
          <div class="kpi-card boundary">
            <div class="kpi-label">Study boundary</div>
            <div class="kpi-value">Synthetic inputs only</div>
            <div class="boundary-note">Research simulation · never enter real customer or personally identifiable information.</div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def profile_form() -> dict:
    st.subheader("Scenario configuration")
    st.selectbox(
        "Scenario",
        list(SCENARIOS),
        key="scenario_name",
        on_change=apply_scenario,
    )
    with st.form("scenario_form"):
        st.number_input("Credit limit", min_value=1_000, step=1_000, key="field_LIMIT_BAL")
        st.number_input("Current bill balance", min_value=0, step=500, key="field_BILL_AMT1")
        st.number_input("Recent payment", min_value=0, step=500, key="field_PAY_AMT1")
        st.selectbox(
            "Repayment status",
            list(STATUS_LABELS),
            format_func=STATUS_LABELS.get,
            key="field_PAY_0",
        )
        submitted = st.form_submit_button("Evaluate scenario", width="stretch")
    profile = {name: st.session_state[f"field_{name}"] for name in next(iter(SCENARIOS.values()))}
    if submitted:
        st.session_state.evaluated_profile = profile
    st.markdown(
        '<div class="capacity-foot">35% human-capacity ceiling</div>', unsafe_allow_html=True
    )
    return st.session_state.evaluated_profile


def ranking_table(scored) -> str:
    rows = []
    for row in scored.sort_values("rank").head(6).itertuples():
        action = row.action.replace("_ESCALATION", "").replace("_REMINDER", "").replace("_", " ")
        action_class = (
            "human"
            if row.action == "HUMAN_ESCALATION"
            else ("no-contact" if row.action == "NO_CONTACT" else "digital")
        )
        rows.append(
            "<tr>"
            f"<td>{row.rank}</td>"
            f"<td>{row.scenario}</td>"
            f'<td class="risk">{row.predicted_probability:.1%}</td>'
            f"<td>{row.allocation_score:,.1f}</td>"
            f'<td><span class="action-chip {action_class}">{action}</span></td>'
            "</tr>"
        )
    return (
        '<table class="rank-table"><thead><tr><th>#</th><th>Scenario</th><th>Risk</th>'
        "<th>Allocation</th><th>Action</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def treatment_mix(scored) -> str:
    counts = scored["action"].value_counts().reindex(ACTION_NAMES.values(), fill_value=0)
    values = (
        ("DIGITAL", int(counts["DIGITAL_REMINDER"]), "digital"),
        ("HUMAN", int(counts["HUMAN_ESCALATION"]), "human"),
        ("NO CONTACT", int(counts["NO_CONTACT"]), "no-contact"),
    )
    maximum = max(value for _, value, _ in values) or 1
    bars = "".join(
        f'<div class="mix-column"><div class="mix-value">{value}</div>'
        f'<div class="mix-bar {kind}" style="height:{132 * value / maximum:.0f}px"></div>'
        f'<div class="mix-label">{label}</div></div>'
        for label, value, kind in values
    )
    human = int(counts["HUMAN_ESCALATION"])
    return (
        f'<div class="mix-chart">{bars}</div>'
        f'<div class="mix-summary">{human} of {len(scored)} human</div>'
        '<div class="mix-caption">Portfolio allocation under the fixed 35% capacity ceiling.</div>'
    )


def decision_page() -> None:
    page_header(
        "decision",
        "Decision Research / Synthetic Portfolio",
        "Collections treatment allocation",
        "Frozen model. Auditable logic. Portfolio-aware capacity allocation.",
    )
    study_cards()
    form_column, analysis_column = st.columns([1.01, 2.54], gap="medium")
    with form_column, st.container(border=True):
        profile = profile_form()

    portfolio = synthetic_portfolio(profile)
    scored = score_portfolio(portfolio)
    selected = scored.iloc[0]
    human_count = int(scored["inside_top_35"].sum())

    with analysis_column:
        st.markdown('<div class="analysis-block">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Individual analysis</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="metric-grid">
              <div class="metric"><div class="metric-name">Risk</div><div class="metric-number">{selected.predicted_probability:.1%}</div></div>
              <div class="metric"><div class="metric-name">Exposure</div><div class="metric-number">{selected.exposure:,.0f}</div></div>
              <div class="metric"><div class="metric-name">Utilization</div><div class="metric-number">{selected.utilization:.1%}</div></div>
              <div class="metric"><div class="metric-name">Payment / bill</div><div class="metric-number">{selected.recent_payment_to_bill_ratio:.2f}</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        panel_class = (
            "human"
            if selected.action == "HUMAN_ESCALATION"
            else "no-contact"
            if selected.action == "NO_CONTACT"
            else ""
        )
        cutoff = "inside" if selected.inside_top_35 else "below"
        st.markdown(
            f"""
            <section class="treatment-panel {panel_class}">
              <div class="eyebrow">Portfolio-ranked treatment</div>
              <div class="treatment-action">{selected.action}</div>
              <div class="treatment-reason">Allocation score {selected.allocation_score:,.1f} · Rank {int(selected["rank"])} of {len(scored)} · {cutoff} the human-allocation cutoff.</div>
            </section>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <section class="lower-grid">
              <div class="editorial-card"><h3>Portfolio ranking</h3>{ranking_table(scored)}</div>
              <div class="editorial-card"><h3>Treatment mix</h3>{treatment_mix(scored)}
                <div class="mix-caption">Human escalation is determined by portfolio rank, not an isolated-account threshold. Top {human_count} accounts are selected.</div>
              </div>
            </section>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)


def research_page() -> None:
    final = json.loads((ROOT / "reports/final_hidden_evaluation.json").read_text())
    page_header(
        "research",
        "Decision Research / Frozen Study",
        "Research record",
        "Fifty autonomous experiments with protected evaluation and no post-hidden tuning.",
    )
    study_cards()
    st.markdown(
        f'<div class="editorial-card research-copy"><h3>Study result</h3><strong>15 KEEP decisions</strong> culminated in exp_039. '
        "Hidden paired uplift was <strong>+230.9740 per account</strong> with a 95% bootstrap interval of "
        f"<strong>[181.1610, 285.8786]</strong>.<br><br>{final['disclaimer']}</div>",
        unsafe_allow_html=True,
    )
    left, right = st.columns([1.5, 1], gap="large")
    with left:
        st.image(ROOT / "reports/figures/research_trajectory.png", width="stretch")
    with right:
        st.image(ROOT / "reports/figures/final_performance.png", width="stretch")
    st.markdown(
        """
        <div class="editorial-card research-copy">
          <h3>Limitations</h3>
          Public default data is a proxy for collections. Treatment responses are simulated, not causal estimates.
          <code>simulated_inr_units</code> are not realized savings. Human capacity is an assumed 35% constraint.
          Demographic inputs are excluded, payment-plan review was not selected, and this is research software—not
          a production collections decision system. Do not enter real customer or personally identifiable information.
        </div>
        """,
        unsafe_allow_html=True,
    )


def experiments_page() -> None:
    page_header(
        "experiments",
        "Decision Research / Experiment Ledger",
        "Experiment record",
        "A frozen, append-only research trajectory from B4 to the selected exp_039 policy.",
    )
    study_cards()
    st.image(ROOT / "reports/figures/research_trajectory.png", width="stretch")
    st.markdown(
        """
        <div class="editorial-card research-copy">
          <h3>What changed</h3>
          The strongest improvements came from exposure-aware ranking, utilization, a compact
          HistGradientBoosting model, six-month payment behavior, a low-exposure no-contact rule,
          constrained tree depth, and slower learning. Linear exposure, extra delinquency multipliers,
          reduced capacity, calibration, ensembles, and payment-plan fallbacks did not improve the protected score.
        </div>
        """,
        unsafe_allow_html=True,
    )


try:
    verify_frozen_artifact()
except (OSError, KeyError, TypeError, ValueError) as error:
    st.error("Frozen model integrity verification failed. Inference is disabled.")
    st.code(str(error))
    st.stop()

initialize_inputs()
requested_view = st.query_params.get("view", "decision")
if requested_view not in {"decision", "research", "experiments"}:
    requested_view = "decision"
view = requested_view
sidebar(view)
{"decision": decision_page, "research": research_page, "experiments": experiments_page}[view]()
