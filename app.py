"""
Chicago Booth EMBA AXP-25 — Massive Analytics Dashboard
Derrick Teo | Autumn 2024 – Spring 2026
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import networkx as nx
import json
from datetime import date, datetime, timedelta
from collections import Counter

from course_data import (
    COURSES, QUARTERS, MILESTONES, CATEGORY_COLORS,
    CONCEPT_CONNECTIONS, CAPSTONE_MAP,
    get_courses_df, get_units_by_category, get_units_by_quarter,
    get_all_frameworks, get_all_topics, search_frameworks,
    PROGRAM_START, PROGRAM_END,
)

# ── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Booth EMBA AXP-25 Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .main { font-family: 'Inter', sans-serif; }

    .stMetric > div { background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1rem; border-radius: 10px; border-left: 4px solid #800000; }

    .big-header { font-size: 2.5rem; font-weight: 700; color: #800000;
        margin-bottom: 0; line-height: 1.2; }
    .sub-header { font-size: 1.1rem; color: #555; margin-top: 0.2rem; }

    .card { background: white; border-radius: 12px; padding: 1.5rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08); margin-bottom: 1rem;
        border: 1px solid #eee; }

    .framework-tag { display: inline-block; background: #800000; color: white;
        padding: 4px 12px; border-radius: 20px; margin: 3px; font-size: 0.82rem; }
    .topic-tag { display: inline-block; background: #1a3c5e; color: white;
        padding: 4px 12px; border-radius: 20px; margin: 3px; font-size: 0.82rem; }

    div[data-testid="stSidebar"] { background: linear-gradient(180deg, #800000 0%, #4a0000 100%); }
    div[data-testid="stSidebar"] .stMarkdown p,
    div[data-testid="stSidebar"] .stMarkdown h1,
    div[data-testid="stSidebar"] .stMarkdown h2,
    div[data-testid="stSidebar"] .stMarkdown h3,
    div[data-testid="stSidebar"] label { color: white !important; }

    .capstone-area { background: linear-gradient(135deg, #800000 0%, #b22222 100%);
        color: white; border-radius: 12px; padding: 1.2rem; margin: 0.5rem 0; }
    .capstone-area h4 { color: white; margin-top: 0; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("# 🎓 Booth EMBA")
    st.markdown("### AXP-25 Analytics")
    st.markdown("---")

    today = date.today()
    days_total = (PROGRAM_END - PROGRAM_START).days
    days_elapsed = (today - PROGRAM_START).days
    progress = min(max(days_elapsed / days_total, 0), 1)
    days_remaining = max((PROGRAM_END - today).days, 0)

    st.markdown(f"**Progress:** {progress:.0%}")
    st.progress(progress)
    st.markdown(f"**{days_remaining}** days to graduation")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        [
            "📊 Learning Dashboard",
            "🕸️ Knowledge Graph",
            "📈 Content Analytics",
            "💰 MBA ROI Calculator",
            "🎯 Capstone Prep Hub",
        ],
        index=0,
    )
    st.markdown("---")
    st.markdown("*Derrick Teo*")
    st.markdown("*Class of 2026*")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: LEARNING DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

if page == "📊 Learning Dashboard":
    st.markdown('<p class="big-header">Learning Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Your complete AXP-25 journey at a glance</p>', unsafe_allow_html=True)
    st.markdown("")

    # ── KPI Row ───────────────────────────────────────────────────────────────
    df = get_courses_df()
    total_units = df["Units"].sum()
    total_courses = len(df)
    total_frameworks = df["Frameworks"].sum()
    total_sessions = df["Sessions"].sum()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Courses", total_courses)
    c2.metric("Total Units", f"{total_units:,}")
    c3.metric("Frameworks Learned", total_frameworks)
    c4.metric("Class Sessions", total_sessions)
    c5.metric("Case Studies", df["Case Studies"].sum())

    st.markdown("---")

    # ── Program Timeline (Gantt) ──────────────────────────────────────────────
    st.subheader("Program Timeline")

    gantt_data = []
    for q in QUARTERS:
        if q["week1_start"]:
            gantt_data.append({
                "Quarter": q["quarter"],
                "Start": q["week1_start"],
                "End": q.get("week1_end") or q["week1_start"],
                "Location": q["location"],
                "Phase": q["label"],
            })
        if q.get("week2_start"):
            gantt_data.append({
                "Quarter": q["quarter"],
                "Start": q["week2_start"],
                "End": q["week2_end"],
                "Location": q["location"],
                "Phase": q["label"],
            })

    gantt_df = pd.DataFrame(gantt_data)
    gantt_df["Start"] = pd.to_datetime(gantt_df["Start"])
    gantt_df["End"] = pd.to_datetime(gantt_df["End"]) + timedelta(days=1)

    fig_gantt = px.timeline(
        gantt_df, x_start="Start", x_end="End", y="Phase",
        color="Location", hover_data=["Quarter"],
        color_discrete_sequence=["#800000", "#1a3c5e", "#2d6a2e", "#6b3fa0"],
    )
    fig_gantt.update_yaxes(autorange="reversed")
    fig_gantt.add_vline(x=datetime.now(), line_dash="dash", line_color="red",
                        annotation_text="TODAY", annotation_position="top")
    fig_gantt.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_gantt, use_container_width=True)

    # ── Milestone tracker ─────────────────────────────────────────────────────
    st.subheader("Key Milestones")
    mile_cols = st.columns(len(MILESTONES))
    for i, m in enumerate(MILESTONES):
        mdate = datetime.strptime(m["date"], "%Y-%m-%d").date()
        status = "✅" if mdate <= today else "⏳"
        with mile_cols[i]:
            st.markdown(f"**{m['icon']} {status}**")
            st.caption(f"{m['event']}")
            st.caption(f"{m['date']}")

    st.markdown("---")

    # ── Units Distribution ────────────────────────────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Units by Category")
        units_cat = get_units_by_category()
        fig_cat = px.pie(
            values=list(units_cat.values()),
            names=list(units_cat.keys()),
            color=list(units_cat.keys()),
            color_discrete_map=CATEGORY_COLORS,
            hole=0.4,
        )
        fig_cat.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=350)
        st.plotly_chart(fig_cat, use_container_width=True)

    with col_right:
        st.subheader("Units by Quarter")
        units_q = get_units_by_quarter()
        fig_q = px.bar(
            x=list(units_q.keys()), y=list(units_q.values()),
            color_discrete_sequence=["#800000"],
            labels={"x": "Quarter", "y": "Units"},
        )
        fig_q.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=350, showlegend=False)
        st.plotly_chart(fig_q, use_container_width=True)

    # ── Course Table ──────────────────────────────────────────────────────────
    st.subheader("Complete Course Catalogue")
    st.dataframe(
        df.style.background_gradient(subset=["Units"], cmap="YlOrRd"),
        use_container_width=True,
        height=600,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: KNOWLEDGE GRAPH
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🕸️ Knowledge Graph":
    st.markdown('<p class="big-header">Knowledge Graph</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">How 120+ frameworks and concepts connect across your MBA</p>',
                unsafe_allow_html=True)
    st.markdown("")

    # Filter
    categories = ["All"] + list(CATEGORY_COLORS.keys())
    sel_cat = st.selectbox("Filter by discipline", categories)

    # Build NetworkX graph
    G = nx.Graph()

    # Add course nodes
    for c in COURSES:
        if sel_cat != "All" and c["category"] != sel_cat:
            continue
        G.add_node(c["name"], node_type="course", category=c["category"],
                    size=max(c["units"], 30), color=CATEGORY_COLORS.get(c["category"], "#999"))

        # Add framework nodes connected to course
        for f in c["frameworks"]:
            G.add_node(f, node_type="framework", category=c["category"],
                        size=15, color=CATEGORY_COLORS.get(c["category"], "#999"))
            G.add_edge(c["name"], f, weight=2)

    # Add cross-concept connections
    for src, tgt, label in CONCEPT_CONNECTIONS:
        if G.has_node(src) and G.has_node(tgt):
            G.add_edge(src, tgt, weight=1, label=label)

    # Layout
    pos = nx.spring_layout(G, k=2.5, iterations=60, seed=42)

    # Build Plotly figure
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=0.5, color="#ccc"),
        hoverinfo="none",
    )

    # Separate course and framework nodes
    course_x, course_y, course_text, course_colors, course_sizes = [], [], [], [], []
    fw_x, fw_y, fw_text, fw_colors, fw_sizes = [], [], [], [], []

    for node in G.nodes():
        x, y = pos[node]
        data = G.nodes[node]
        if data.get("node_type") == "course":
            course_x.append(x); course_y.append(y)
            course_text.append(node)
            course_colors.append(data["color"])
            course_sizes.append(data["size"] / 3)
        else:
            fw_x.append(x); fw_y.append(y)
            # Count connections
            degree = G.degree(node)
            fw_text.append(f"{node} ({degree} connections)")
            fw_colors.append(data["color"])
            fw_sizes.append(8 + degree * 3)

    course_trace = go.Scatter(
        x=course_x, y=course_y, mode="markers+text",
        marker=dict(size=course_sizes, color=course_colors, line=dict(width=2, color="white")),
        text=course_text, textposition="top center",
        textfont=dict(size=10, color="#333"),
        hoverinfo="text", name="Courses",
    )

    fw_trace = go.Scatter(
        x=fw_x, y=fw_y, mode="markers",
        marker=dict(size=fw_sizes, color=fw_colors, opacity=0.7,
                    line=dict(width=1, color="white")),
        text=fw_text, hoverinfo="text", name="Frameworks",
    )

    fig_graph = go.Figure(data=[edge_trace, fw_trace, course_trace])
    fig_graph.update_layout(
        showlegend=True, height=700,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=0, r=0, t=30, b=0),
        plot_bgcolor="white",
    )
    st.plotly_chart(fig_graph, use_container_width=True)

    # ── Connection stats ──────────────────────────────────────────────────────
    st.subheader("Most Connected Concepts")
    degree_data = [(node, G.degree(node), G.nodes[node].get("node_type", ""))
                   for node in G.nodes() if G.degree(node) > 1]
    degree_data.sort(key=lambda x: x[1], reverse=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Top Frameworks (Hub Concepts)**")
        for name, deg, ntype in degree_data[:15]:
            if ntype == "framework":
                st.markdown(f"- **{name}** — {deg} connections")
    with col2:
        st.markdown("**Most Interconnected Courses**")
        for name, deg, ntype in degree_data[:10]:
            if ntype == "course":
                st.markdown(f"- **{name}** — {deg} connections")

    # ── Cross-discipline bridges ──────────────────────────────────────────────
    st.subheader("Cross-Discipline Bridges")
    st.markdown("Concepts that connect different domains — the integrative thinking your capstone requires:")

    bridges = []
    for src, tgt, label in CONCEPT_CONNECTIONS:
        if "→" in label:
            bridges.append({"From": src, "To": tgt, "Bridge": label})
    if bridges:
        st.dataframe(pd.DataFrame(bridges), use_container_width=True, height=300)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: CONTENT ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📈 Content Analytics":
    st.markdown('<p class="big-header">Content Analytics</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Deep dive into the intellectual landscape of your EMBA</p>',
                unsafe_allow_html=True)
    st.markdown("")

    df = get_courses_df()

    # ── Topic & Framework Density ─────────────────────────────────────────────
    st.subheader("Course Complexity: Topics & Frameworks per Course")

    fig_complexity = go.Figure()
    fig_complexity.add_trace(go.Bar(
        name="Key Topics", x=df["Course"], y=df["Topics"],
        marker_color="#800000",
    ))
    fig_complexity.add_trace(go.Bar(
        name="Frameworks", x=df["Course"], y=df["Frameworks"],
        marker_color="#1a3c5e",
    ))
    fig_complexity.update_layout(
        barmode="group", height=450,
        margin=dict(l=0, r=0, t=30, b=100),
        xaxis_tickangle=-45,
    )
    st.plotly_chart(fig_complexity, use_container_width=True)

    # ── Topic Heatmap across Categories ───────────────────────────────────────
    st.subheader("Topic Distribution Heatmap")

    all_topics = get_all_topics()
    cat_topic_counts = {}
    for topic, course, category in all_topics:
        if category not in cat_topic_counts:
            cat_topic_counts[category] = 0
        cat_topic_counts[category] += 1

    # Topic themes
    THEME_KEYWORDS = {
        "Valuation & Finance": ["valuation", "dcf", "present value", "npv", "wacc", "capm", "capital",
                                 "stock", "bond", "portfolio", "leverage", "debt", "equity"],
        "Strategy & Competition": ["strategy", "competitive", "entry", "advantage", "industry",
                                    "porter", "disruption", "innovation", "diversification"],
        "Pricing & Revenue": ["pricing", "price", "elasticity", "evc", "bundling", "tariff",
                               "revenue", "demand"],
        "Decision & Behavior": ["decision", "bias", "heuristic", "prospect", "nudge", "judgment",
                                 "risk", "uncertainty", "cognitive"],
        "Operations & Process": ["bottleneck", "inventory", "lean", "cycle", "throughput", "supply chain",
                                  "process", "operations", "eoq"],
        "Accounting & Metrics": ["accounting", "cost", "variance", "budget", "scorecard", "abc",
                                  "financial ratio", "dupont"],
        "Economics & Markets": ["equilibrium", "supply", "demand", "monopoly", "oligopoly",
                                 "externality", "gdp", "inflation", "monetary"],
        "Leadership & People": ["leadership", "negotiation", "batna", "incentive", "principal-agent",
                                 "ethics", "stakeholder", "communication"],
    }

    # Count topics per theme
    theme_course_matrix = {}
    for theme, keywords in THEME_KEYWORDS.items():
        theme_course_matrix[theme] = {}
        for c in COURSES:
            count = 0
            for t in c["key_topics"] + c["frameworks"]:
                if any(kw in t.lower() for kw in keywords):
                    count += 1
            if count > 0:
                theme_course_matrix[theme][c["name"]] = count

    # Build heatmap data
    course_names = [c["name"] for c in COURSES if c["units"] > 0]
    theme_names = list(THEME_KEYWORDS.keys())
    z_data = []
    for theme in theme_names:
        row = []
        for cname in course_names:
            row.append(theme_course_matrix.get(theme, {}).get(cname, 0))
        z_data.append(row)

    fig_heatmap = go.Figure(data=go.Heatmap(
        z=z_data, x=course_names, y=theme_names,
        colorscale="YlOrRd", hoverongaps=False,
    ))
    fig_heatmap.update_layout(
        height=450, margin=dict(l=0, r=0, t=30, b=120),
        xaxis_tickangle=-45,
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

    # ── Framework Universe ────────────────────────────────────────────────────
    st.subheader("Complete Framework Universe")

    all_fw = get_all_frameworks()
    fw_df = pd.DataFrame(all_fw, columns=["Framework", "Course", "Category"])

    # Treemap
    fig_tree = px.treemap(
        fw_df, path=["Category", "Course", "Framework"],
        color="Category", color_discrete_map=CATEGORY_COLORS,
    )
    fig_tree.update_layout(height=600, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_tree, use_container_width=True)

    # ── Course interconnection radar ──────────────────────────────────────────
    st.subheader("Course Interconnection Radar")
    sel_course = st.selectbox("Select a course", [c["name"] for c in COURSES])

    course_obj = next(c for c in COURSES if c["name"] == sel_course)

    categories_radar = list(THEME_KEYWORDS.keys())
    values = []
    for theme, keywords in THEME_KEYWORDS.items():
        count = 0
        for t in course_obj["key_topics"] + course_obj["frameworks"]:
            if any(kw in t.lower() for kw in keywords):
                count += 1
        values.append(count)

    fig_radar = go.Figure(data=go.Scatterpolar(
        r=values + [values[0]],
        theta=categories_radar + [categories_radar[0]],
        fill="toself",
        fillcolor="rgba(128,0,0,0.2)",
        line_color="#800000",
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, max(values) + 1])),
        height=450, margin=dict(l=50, r=50, t=30, b=30),
    )

    col_radar, col_detail = st.columns([1, 1])
    with col_radar:
        st.plotly_chart(fig_radar, use_container_width=True)
    with col_detail:
        st.markdown(f"**{course_obj['name']}** ({course_obj['code']})")
        st.markdown(f"*{course_obj['category']}* | {course_obj['units']} units | {course_obj['sessions']} sessions")
        if course_obj.get("professor") and course_obj["professor"] != "TBD":
            st.markdown(f"Professor: **{course_obj['professor']}**")
        st.markdown("**Key Frameworks:**")
        tags_html = " ".join(f'<span class="framework-tag">{f}</span>' for f in course_obj["frameworks"])
        st.markdown(tags_html, unsafe_allow_html=True)
        st.markdown("")
        st.markdown("**Key Topics:**")
        tags_html = " ".join(f'<span class="topic-tag">{t}</span>' for t in course_obj["key_topics"])
        st.markdown(tags_html, unsafe_allow_html=True)
        if course_obj["case_studies"]:
            st.markdown("")
            st.markdown("**Case Studies:**")
            for cs in course_obj["case_studies"]:
                st.markdown(f"- {cs}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: MBA ROI CALCULATOR
# ══════════════════════════════════════════════════════════════════════════════

elif page == "💰 MBA ROI Calculator":
    st.markdown('<p class="big-header">MBA ROI Calculator</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Financial modeling of your EMBA investment</p>',
                unsafe_allow_html=True)
    st.markdown("")

    col_inputs, col_results = st.columns([1, 2])

    with col_inputs:
        st.markdown("### Investment Inputs")
        tuition = st.number_input("Total Tuition (USD)", value=220_000, step=10_000, format="%d")
        travel_costs = st.number_input("Travel & Living (total, USD)", value=80_000, step=5_000, format="%d")
        opportunity_cost_annual = st.number_input("Annual Opportunity Cost (USD)",
                                                   value=30_000, step=5_000, format="%d",
                                                   help="Lost income/productivity during study periods")
        program_years = 2.0

        st.markdown("### Career Assumptions")
        pre_salary = st.number_input("Pre-MBA Annual Compensation (USD)", value=200_000, step=10_000, format="%d")
        post_salary = st.number_input("Post-MBA Annual Compensation (USD)", value=300_000, step=10_000, format="%d")
        salary_growth_pre = st.slider("Without-MBA Annual Salary Growth (%)", 2.0, 8.0, 3.0, 0.5) / 100
        salary_growth_post = st.slider("With-MBA Annual Salary Growth (%)", 3.0, 15.0, 7.0, 0.5) / 100
        years_to_model = st.slider("Years to model post-graduation", 5, 30, 15)
        discount_rate = st.slider("Discount rate (%)", 3.0, 12.0, 6.0, 0.5) / 100

        st.markdown("### Intangible Value")
        network_value = st.number_input("Estimated Network Value (annual, USD)",
                                         value=20_000, step=5_000, format="%d",
                                         help="Deal flow, referrals, board seats, etc.")
        brand_premium = st.number_input("Brand Premium (annual, USD)",
                                         value=15_000, step=5_000, format="%d",
                                         help="Chicago Booth brand premium on earning power")

    with col_results:
        # Calculations
        total_investment = tuition + travel_costs + (opportunity_cost_annual * program_years)

        years = list(range(1, years_to_model + 1))
        salary_without = [pre_salary * (1 + salary_growth_pre) ** y for y in years]
        salary_with = [post_salary * (1 + salary_growth_post) ** y for y in years]
        salary_diff = [w - wo for w, wo in zip(salary_with, salary_without)]
        intangible = [network_value + brand_premium] * years_to_model

        # NPV calculation
        cumulative_benefit = []
        npv_benefits = 0
        running_total = -total_investment
        breakeven_year = None

        for i, y in enumerate(years):
            annual_benefit = salary_diff[i] + intangible[i]
            discounted = annual_benefit / (1 + discount_rate) ** y
            npv_benefits += discounted
            running_total += annual_benefit
            cumulative_benefit.append(running_total)
            if breakeven_year is None and running_total > 0:
                breakeven_year = y

        npv = npv_benefits - total_investment
        roi = (npv_benefits / total_investment - 1) * 100

        # KPIs
        st.markdown("### Return Summary")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Investment", f"${total_investment:,.0f}")
        m2.metric("NPV of Returns", f"${npv:,.0f}",
                  delta=f"{'Positive' if npv > 0 else 'Negative'}")
        m3.metric("ROI", f"{roi:.0f}%")
        m4.metric("Breakeven Year", f"Year {breakeven_year}" if breakeven_year else "N/A")

        st.markdown("---")

        # Cumulative benefit chart
        st.markdown("### Cumulative Financial Impact")
        fig_roi = go.Figure()
        fig_roi.add_trace(go.Scatter(
            x=years, y=cumulative_benefit,
            mode="lines+markers", name="Cumulative Net Benefit",
            line=dict(color="#800000", width=3),
            fill="tozeroy",
            fillcolor="rgba(128,0,0,0.1)",
        ))
        fig_roi.add_hline(y=0, line_dash="dash", line_color="gray")
        if breakeven_year:
            fig_roi.add_vline(x=breakeven_year, line_dash="dash", line_color="green",
                             annotation_text=f"Breakeven: Year {breakeven_year}")
        fig_roi.update_layout(
            xaxis_title="Years After Graduation",
            yaxis_title="Cumulative Net Benefit (USD)",
            height=350, margin=dict(l=0, r=0, t=30, b=0),
            yaxis_tickformat="$,.0f",
        )
        st.plotly_chart(fig_roi, use_container_width=True)

        # Salary trajectory comparison
        st.markdown("### Salary Trajectory: With vs Without MBA")
        fig_salary = go.Figure()
        fig_salary.add_trace(go.Scatter(
            x=years, y=salary_with, mode="lines",
            name="With Booth EMBA", line=dict(color="#800000", width=3),
        ))
        fig_salary.add_trace(go.Scatter(
            x=years, y=salary_without, mode="lines",
            name="Without MBA", line=dict(color="#999", width=2, dash="dash"),
        ))
        fig_salary.add_trace(go.Bar(
            x=years, y=salary_diff, name="Annual Uplift",
            marker_color="rgba(128,0,0,0.2)",
        ))
        fig_salary.update_layout(
            xaxis_title="Years After Graduation",
            yaxis_title="Annual Compensation (USD)",
            height=350, margin=dict(l=0, r=0, t=30, b=0),
            yaxis_tickformat="$,.0f",
        )
        st.plotly_chart(fig_salary, use_container_width=True)

        # Investment breakdown
        st.markdown("### Investment Breakdown")
        fig_inv = px.pie(
            values=[tuition, travel_costs, opportunity_cost_annual * program_years],
            names=["Tuition", "Travel & Living", "Opportunity Cost"],
            color_discrete_sequence=["#800000", "#1a3c5e", "#b8860b"],
            hole=0.4,
        )
        fig_inv.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_inv, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5: CAPSTONE PREP HUB
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🎯 Capstone Prep Hub":
    st.markdown('<p class="big-header">Capstone Simulation Prep Hub</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Integrated Strategic Management — your MBA knowledge, unified</p>',
                unsafe_allow_html=True)
    st.markdown("")

    # ── Alert banner ──────────────────────────────────────────────────────────
    days_to_capstone = (date(2026, 4, 12) - date.today()).days
    if days_to_capstone > 0:
        st.warning(f"**{days_to_capstone} days until Capstone Week 2 begins** (April 12, 2026)")
    elif days_to_capstone >= -6:
        st.error("**CAPSTONE WEEK IS LIVE! Good luck!**")
    else:
        st.success("**Capstone completed!**")

    # ── Decision Area Cards ───────────────────────────────────────────────────
    st.subheader("Decision Areas & Framework Quick Reference")
    st.markdown("Each decision area in the simulation maps to frameworks from your coursework:")

    for area_name, area_data in CAPSTONE_MAP.items():
        with st.expander(f"**{area_name}**", expanded=False):
            col_fw, col_q = st.columns([2, 1])

            with col_fw:
                st.markdown("**Relevant Courses:**")
                for course in area_data["courses"]:
                    color = "#999"
                    for c in COURSES:
                        if c["name"] == course:
                            color = CATEGORY_COLORS.get(c["category"], "#999")
                            break
                    st.markdown(f'<span style="color:{color}; font-weight:600;">● {course}</span>',
                                unsafe_allow_html=True)

                st.markdown("")
                st.markdown("**Key Frameworks:**")
                fw_html = " ".join(f'<span class="framework-tag">{f}</span>'
                                   for f in area_data["frameworks"])
                st.markdown(fw_html, unsafe_allow_html=True)

            with col_q:
                st.markdown("**Key Questions to Ask:**")
                for q in area_data["key_questions"]:
                    st.markdown(f"- {q}")

    st.markdown("---")

    # ── Framework Search ──────────────────────────────────────────────────────
    st.subheader("Framework Search")
    query = st.text_input("Search for any concept, framework, or topic",
                          placeholder="e.g., WACC, bottleneck, Nash, pricing...")
    if query:
        results = search_frameworks(query)
        if results:
            st.success(f"Found **{len(results)}** matches for '{query}'")
            res_df = pd.DataFrame(results).rename(columns={
                "match": "Match", "type": "Type", "course": "Course", "category": "Category"
            })
            st.dataframe(res_df, use_container_width=True)
        else:
            st.info(f"No results for '{query}'. Try a broader term.")

    st.markdown("---")

    # ── Integration Matrix ────────────────────────────────────────────────────
    st.subheader("Cross-Course Integration Matrix")
    st.markdown("Shows which courses share concepts — key for the simulation where you need to integrate all disciplines:")

    # Build course-course connection count
    course_names_list = [c["name"] for c in COURSES if c["units"] > 0 and c["name"] != "LEAD (Leadership Exploration & Development)"]
    n = len(course_names_list)
    matrix = np.zeros((n, n))

    # Count shared connections
    for src, tgt, label in CONCEPT_CONNECTIONS:
        src_courses = set()
        tgt_courses = set()
        for c in COURSES:
            if src in c["frameworks"] or src in c["key_topics"]:
                src_courses.add(c["name"])
            if tgt in c["frameworks"] or tgt in c["key_topics"]:
                tgt_courses.add(c["name"])

        for sc in src_courses:
            for tc in tgt_courses:
                if sc != tc and sc in course_names_list and tc in course_names_list:
                    i = course_names_list.index(sc)
                    j = course_names_list.index(tc)
                    matrix[i][j] += 1
                    matrix[j][i] += 1

    fig_matrix = go.Figure(data=go.Heatmap(
        z=matrix,
        x=course_names_list,
        y=course_names_list,
        colorscale="Reds",
        hoverongaps=False,
    ))
    fig_matrix.update_layout(
        height=600, margin=dict(l=0, r=0, t=30, b=120),
        xaxis_tickangle=-45,
    )
    st.plotly_chart(fig_matrix, use_container_width=True)

    # ── Cheat Sheet Generator ─────────────────────────────────────────────────
    st.subheader("Capstone Cheat Sheet")
    st.markdown("A quick-reference of the most critical frameworks for your simulation:")

    critical_frameworks = [
        ("Porter's Five Forces", "Competitive Strategy", "Analyze industry attractiveness and competitive dynamics"),
        ("DCF valuation", "Corporate Finance", "Value any investment using discounted cash flows"),
        ("WACC", "Corporate Finance", "Calculate the firm's blended cost of capital"),
        ("EVC analysis", "Pricing Strategies", "Set price based on economic value to the customer"),
        ("Little's Law", "Operations Management", "Inventory = Throughput x Cycle Time"),
        ("Bottleneck analysis", "Operations Management", "Find and manage the constraining resource"),
        ("STP framework", "Marketing Management", "Segment, Target, Position your offering"),
        ("BATNA analysis", "Negotiations", "Know your best alternative before negotiating"),
        ("Prospect theory", "Managerial Decision Making", "People feel losses ~2x more than equivalent gains"),
        ("Capital structure optimization", "Financial Strategy", "Find the right debt/equity mix"),
        ("Balanced scorecard", "Managerial Accounting", "Track financial + non-financial KPIs"),
        ("Nash Equilibrium", "Microeconomics", "Predict stable outcomes in competitive interactions"),
        ("EOQ model", "Operations Management", "Optimal order quantity = sqrt(2DS/H)"),
        ("Price elasticity", "Microeconomics / Pricing", "% change in quantity / % change in price"),
        ("Scenario planning", "Capstone", "Prepare for multiple futures, don't bet on one"),
    ]

    cheat_df = pd.DataFrame(critical_frameworks,
                            columns=["Framework", "Source Course", "One-Line Summary"])
    st.dataframe(cheat_df, use_container_width=True, height=500)

    # ── Simulation Decision Checklist ─────────────────────────────────────────
    st.subheader("Simulation Decision Checklist")
    st.markdown("Before each round, ask yourself:")

    checklist = [
        "Have I analyzed the competitive landscape? (Porter's 5 Forces)",
        "Is my pricing consistent with customer WTP and competitor actions? (EVC, elasticity)",
        "Am I monitoring my financial ratios and capital structure? (DuPont, leverage)",
        "Have I identified and managed operational bottlenecks? (Little's Law)",
        "Am I investing in the right segments? (STP, conjoint analysis)",
        "What cognitive biases might be affecting my decisions? (System 1 vs 2)",
        "Am I creating value in negotiations or just claiming it? (ZOPA, integrative bargaining)",
        "What scenarios have I prepared for? (Macro risks, competitor moves)",
        "Are my team incentives aligned with our goals? (Principal-agent, tournament theory)",
        "Can I communicate my strategy in a compelling narrative? (Story arc)",
    ]

    for item in checklist:
        st.checkbox(item, key=f"check_{item[:20]}")
