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
import math as _math

# ══════════════════════════════════════════════════════════════════════════════
# CACHED SIMULATION FUNCTIONS (speeds up slider interactions 10-100x)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False, max_entries=50)
def _normal_cdf(x: float, mu: float, sigma: float) -> float:
    if sigma <= 0:
        return 1.0 if x < mu else 0.0
    z = (x - mu) / sigma
    return 0.5 * (1 + _math.erf(z / _math.sqrt(2)))


@st.cache_data(show_spinner=False, max_entries=30)
def simulate_bass_normal(M: int, p: float, q: float, p_ad_per_500: float,
                          ad_daily: float, ad_duration: int,
                          price: float, mean_wtp: float, std_wtp: float,
                          sim_days: int):
    """
    Simulate Bass model with Normal WTP and 3 arrival streams.
    Returns dict with innovators, imitators, advertising, arrivals, purchases, cumulative.
    """
    p_buy = 1 - _normal_cdf(price, mean_wtp, std_wtp)
    days = list(range(1, int(sim_days) + 1))
    innovators_list, imitators_list, advertising_list = [], [], []
    total_arrivals, purchases_list, cumulative_purchases = [], [], []
    cumulative_adopters = 0.0

    for t in days:
        remaining = max(0, M - cumulative_adopters)
        innovators = p * remaining
        imitators = q * (cumulative_adopters / M) * remaining if M > 0 else 0
        if t <= ad_duration and ad_daily > 0:
            p_ad_boost = (ad_daily / 500) * p_ad_per_500
            ad_customers = p_ad_boost * remaining
        else:
            ad_customers = 0
        arrivals = innovators + imitators + ad_customers
        buys = arrivals * p_buy
        innovators_list.append(innovators)
        imitators_list.append(imitators)
        advertising_list.append(ad_customers)
        total_arrivals.append(arrivals)
        purchases_list.append(buys)
        cumulative_adopters += buys
        cumulative_purchases.append(cumulative_adopters)

    return {
        "days": days, "p_buy": p_buy,
        "innovators": innovators_list, "imitators": imitators_list,
        "advertising": advertising_list, "total_arrivals": total_arrivals,
        "purchases": purchases_list, "cumulative": cumulative_purchases,
    }


@st.cache_data(show_spinner=False, max_entries=50)
def simulate_scenario_traj(price: float, ad_daily: float, ad_duration: int,
                            M: int, p: float, q: float, p_ad_per_500: float,
                            mean_wtp: float, std_wtp: float,
                            materials: float, mfg_oh: float,
                            shipping: float, handling: float, commission_frac: float,
                            days_total: int = 1460):
    """
    Simulate a single pricing/advertising scenario and return cumulative CM trajectory.
    """
    p_buy_sc = 1 - _normal_cdf(price, mean_wtp, std_wtp)
    var_cost = materials + shipping + handling + mfg_oh
    cm_per_unit = price * (1 - commission_frac) - var_cost
    cum_adopters = 0.0
    cum_cm = 0.0
    trajectory = []

    for t in range(1, days_total + 1):
        remaining = max(0, M - cum_adopters)
        innov = p * remaining
        imit = q * (cum_adopters / M) * remaining if M > 0 else 0
        if t <= ad_duration and ad_daily > 0:
            ad_cust = (ad_daily / 500) * p_ad_per_500 * remaining
        else:
            ad_cust = 0
        buys = (innov + imit + ad_cust) * p_buy_sc
        cum_adopters += buys
        cum_cm += cm_per_unit * buys
        if t <= ad_duration:
            cum_cm -= ad_daily
        trajectory.append(cum_cm)

    return {"p_buy": p_buy_sc, "cum_cm_final": cum_cm,
            "cum_units": cum_adopters, "cm_per_unit": cm_per_unit,
            "trajectory": trajectory}


@st.cache_data(show_spinner=False, max_entries=50)
def find_optimal_price_normal(price_min: int, price_max: int,
                                mean_wtp: float, std_wtp: float,
                                materials: float, shipping: float,
                                handling: float, commission_frac: float,
                                step: int = 5):
    """
    Find the retail price that maximizes CM per arriving customer under
    Normal WTP distribution.
    """
    best_p, best_cm = price_min, -1e18
    for test_p in range(price_min, price_max + 1, step):
        pb = 1 - _normal_cdf(test_p, mean_wtp, std_wtp)
        cm_u = test_p * (1 - commission_frac) - materials - shipping - handling
        cm_per_arr = cm_u * pb
        if cm_per_arr > best_cm:
            best_cm = cm_per_arr
            best_p = test_p
    return best_p

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

    .stMetric > div { background: rgba(128, 0, 0, 0.15);
        padding: 1rem; border-radius: 10px; border-left: 4px solid #800000; }
    .stMetric label { color: inherit !important; }
    .stMetric [data-testid="stMetricValue"] { color: inherit !important; }
    .stMetric [data-testid="stMetricLabel"] { color: inherit !important; }

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
            "✨ 14 New War Room",
            "🚀 14 Trial War Room",
            "🏭 13 Trial War Room",
            "⚔️ 12 Trial War Room",
            "🎮 ISM War Room",
            "📖 War Room Prep",
            "🕸️ Knowledge Graph",
        ],
        index=0,
    )
    st.markdown("")
    st.caption("📁 Misc")
    misc_page = st.radio(
        "Misc",
        [
            "— none —",
            "📊 Learning Dashboard",
            "📈 Content Analytics",
            "🎯 Capstone Prep Hub",
        ],
        index=0,
        label_visibility="collapsed",
    )
    if misc_page != "— none —":
        page = misc_page
    st.markdown("---")
    st.markdown("*Derrick Teo*")
    st.markdown("*Class of 2026*")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 0: ISM WAR ROOM — GLEACHER GAME STRATEGY
# ══════════════════════════════════════════════════════════════════════════════

if page == "🎮 ISM War Room":
    st.markdown('<p class="big-header">ISM Capstone War Room</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Gleacher Game Strategy — Integrated Strategic Management (42805)</p>',
                unsafe_allow_html=True)
    st.markdown("")

    # ── Countdown & Schedule ──────────────────────────────────────────────────
    capstone_start = date(2026, 4, 12)
    days_to_go = (capstone_start - date.today()).days
    if days_to_go > 0:
        st.warning(f"**{days_to_go} days until Capstone Week begins** (Sunday, April 12)")
    elif days_to_go >= -6:
        st.error("**CAPSTONE WEEK IS LIVE — EXECUTE THE PLAN!**")
    else:
        st.success("**Capstone completed!**")

    # ── KPI Cheat Sheet with hover tooltips ──────────────────────────────────
    st.subheader("Critical Game Parameters")
    st.caption("Hover over each card for details")

    PARAMS = [
        {"label": "Batch Size", "value": "100 units",
         "tip": "Products manufactured in batches of 100. Factory can only make 1 batch of 1 product at a time. Cost = $100 x 100 = $10,000 per batch."},
        {"label": "Material Cost", "value": "$100/unit",
         "tip": "Raw material cost per unit. Payable 15 days after incurred. Total batch cost = $10,000. This is your floor price — never sell below $100."},
        {"label": "Production Cycle", "value": "2.5 days",
         "tip": "Time to manufacture 1 batch. Factory can start next batch immediately while current ships. If both products run, they alternate batches."},
        {"label": "Factory→DC Transit", "value": "1 day",
         "tip": "Free shipping from your factory to your DC (same region). Total lead time = 2.5 + 1 = 3.5 days. If both products: ~6 days effective."},
        {"label": "Customer Arrival", "value": "0.01%/day",
         "tip": "Daily arrivals = 0.0001 x market size. Hormone (300K mkt) = 30 customers/day. Specialty (140K mkt) = 14 customers/day. Lost forever if no stock."},
        {"label": "Emergency Loan", "value": "40% APR",
         "tip": "If cash hits zero, you get forced loans at 40% APR — catastrophic. Auto-repaid as cash becomes available. AVOID AT ALL COSTS."},
        {"label": "Cash Interest", "value": "3% APR",
         "tip": "Idle cash earns 3% annually. Not amazing, but means holding cash isn't wasteful. The 37% spread vs emergency loans makes cash management critical."},
        {"label": "Tax Rate", "value": "35% quarterly",
         "tip": "Profits taxed at 35%, paid end of every quarter (90 days). Plan cash reserves for tax bills. High inventory write-offs reduce taxable income."},
    ]

    for row_start in range(0, len(PARAMS), 4):
        row_params = PARAMS[row_start:row_start + 4]
        param_cols = st.columns(4)
        for pi, param in enumerate(row_params):
            with param_cols[pi]:
                tip_escaped = param["tip"].replace("'", "&#39;")
                st.markdown(f"""<div style="position:relative;background:rgba(128,0,0,0.15);border-left:4px solid #800000;border-radius:10px;padding:0.8rem 1rem;cursor:help;" title="{tip_escaped}">
<span style="font-size:0.8rem;opacity:0.7;">{param["label"]}</span><br>
<span style="font-size:1.5rem;font-weight:700;">{param["value"]}</span>
</div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Week Schedule ─────────────────────────────────────────────────────────
    st.subheader("Week Schedule & Game Progression")

    schedule_data = [
        {"Day": "Sunday Apr 12", "Class (3-6pm)": "Microeconomics, Statistics, Operations",
         "Game (7-9pm)": "Monopoly + Trading Game", "Assignment": "A1 due by 3pm (Individual)"},
        {"Day": "Monday Apr 13", "Class (3-6pm)": "Operations, Financial Accounting",
         "Game (7-9pm)": "Production Game", "Assignment": "A2 due by 9am (Group)"},
        {"Day": "Tuesday Apr 14", "Class (3-6pm)": "Marketing, Finance, Strategy",
         "Game (7-9pm)": "Practice Game", "Assignment": "A3 due by 9am (Group)"},
        {"Day": "Wednesday Apr 15", "Class (3-6pm)": "Analyze Practice, Prep for Competition",
         "Game (7-9pm)": "COMPETITION", "Assignment": "A4 due by 9am (Group)"},
        {"Day": "Thursday Apr 16", "Class (3-6pm)": "Game Analysis",
         "Game (7-9pm)": "COMPETITION", "Assignment": "A5 due by 9am (Group)"},
        {"Day": "Friday Apr 17", "Class (3-6pm)": "Wrap-Up (11am-1pm)",
         "Game (7-9pm)": "—", "Assignment": "A6 + Final Project due May 3"},
    ]
    st.dataframe(pd.DataFrame(schedule_data), use_container_width=True, hide_index=True)

    # ── Grading Weights ───────────────────────────────────────────────────────
    grade_col1, grade_col2 = st.columns([1, 2])
    with grade_col1:
        st.subheader("Grading Breakdown")
        grade_data = {
            "Component": ["Assignment 1 (Individual)", "Assignments 2-4 (Group)", "Assignment 5 (Group)",
                          "Assignment 6 (Individual)", "Final Project (Group)", "Participation", "Game Performance"],
            "Points": [10, 25, 5, 10, 30, 15, 5],
        }
        grade_df = pd.DataFrame(grade_data)
        fig_grade = px.pie(grade_df, values="Points", names="Component",
                           color_discrete_sequence=["#800000", "#b22222", "#cd5c5c",
                                                     "#e8967a", "#1a3c5e", "#2d6a2e", "#b8860b"],
                           hole=0.4)
        fig_grade.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_grade, use_container_width=True)

    with grade_col2:
        st.subheader("Game Performance = 5% of grade")
        st.markdown("""
        The **final project** (30%) is the biggest single component — it requires:
        - Summary & analysis of your decisions and outcomes during the simulation
        - A projection for the future
        - A **valuation of the business**

        **Key insight:** Game Performance is only 5%, but the Final Project (30%) depends on
        how well you played. Playing well = better story to tell = higher combined score (35%).
        """)

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # THE 7 WINNING STRATEGIES
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("The 7 Winning Strategies")

    # ── Strategy 1: Pricing ───────────────────────────────────────────────────
    with st.expander("**1. PRICING — The #1 Lever**", expanded=True):
        st.markdown("""
        WTP is **uniformly distributed** from \\$0 to max WTP. This means the optimal monopoly price is:

        > **P* = max_WTP / 2**

        This maximizes `Revenue = arrival_rate x (max_WTP - P) / max_WTP x P`
        """)

        st.markdown("#### Interactive Pricing Simulator")
        sim_col1, sim_col2 = st.columns(2)
        with sim_col1:
            max_wtp = st.slider("Max WTP ($)", 200, 2000, 800, 50)
            market_size = st.selectbox("Market Size", [140000, 300000], index=1,
                                        format_func=lambda x: f"{x:,} ({'Hormone' if x==300000 else 'Specialty'})")
            price_range = np.arange(50, max_wtp, 10)

        arrival_rate = 0.0001 * market_size
        daily_demand = arrival_rate * (max_wtp - price_range) / max_wtp
        daily_revenue = daily_demand * price_range
        daily_cost = daily_demand * 100  # $100/unit marginal cost
        daily_profit = daily_revenue - daily_cost

        optimal_price = max_wtp / 2
        # Profit-maximizing price (accounting for marginal cost)
        profit_max_price = (max_wtp + 100) / 2

        with sim_col2:
            st.metric("Revenue-Maximizing Price", f"${optimal_price:,.0f}")
            st.metric("Profit-Maximizing Price (incl. $100 cost)", f"${profit_max_price:,.0f}")
            demand_at_opt = arrival_rate * (max_wtp - profit_max_price) / max_wtp
            st.metric("Daily Demand at Profit-Max Price", f"{demand_at_opt:,.1f} units/day")
            st.metric("Daily Profit at Profit-Max Price",
                       f"${demand_at_opt * (profit_max_price - 100):,.0f}/day")

        fig_pricing = go.Figure()
        fig_pricing.add_trace(go.Scatter(x=price_range, y=daily_revenue,
                                          name="Daily Revenue", line=dict(color="#800000", width=2)))
        fig_pricing.add_trace(go.Scatter(x=price_range, y=daily_profit,
                                          name="Daily Profit (after COGS)", line=dict(color="#2d6a2e", width=2)))
        fig_pricing.add_trace(go.Scatter(x=price_range, y=daily_demand * 100,
                                          name="Daily COGS", line=dict(color="#999", width=1, dash="dash")))
        fig_pricing.add_vline(x=profit_max_price, line_dash="dash", line_color="green",
                               annotation_text=f"Profit-max: ${profit_max_price:,.0f}",
                               annotation_position="top left",
                               annotation_font_size=11,
                               annotation_y=1.08)
        fig_pricing.add_vline(x=optimal_price, line_dash="dot", line_color="#800000",
                               annotation_text=f"Rev-max: ${optimal_price:,.0f}",
                               annotation_position="top right",
                               annotation_font_size=11,
                               annotation_y=0.95)
        fig_pricing.update_layout(height=350, xaxis_title="Price ($)", yaxis_title="$ per day",
                                   margin=dict(l=0, r=0, t=30, b=0), yaxis_tickformat="$,.0f")
        st.plotly_chart(fig_pricing, use_container_width=True)

        st.markdown("""
        **Tactical rules:**
        - Start at **profit-maximizing price** (slightly above max_WTP/2 to account for $100 cost)
        - If inventory piling up → **lower price** to move units
        - If stocking out → **raise price** to slow demand and capture more surplus
        - **End-game:** slash prices aggressively — unsold inventory = $0
        """)

    # ── Strategy 2: Inventory ─────────────────────────────────────────────────
    with st.expander("**2. INVENTORY MANAGEMENT — Never Stock Out, Never Over-Stock**"):
        st.markdown(r"""
        **From Operations Management (Little's Law, Newsvendor):**

        | Parameter | Hormone | Specialty |
        |---|---|---|
        | Market size | 300,000 | 140,000 |
        | Arrival rate (N) | 30 customers/day | 14 customers/day |
        | Production cycle | 2.5 days | 2.5 days |
        | Transit to DC | 1 day | 1 day |
        | **Lead time (L)** | **3.5 days** | **3.5 days** |

        **Demand Model:** Each day, N customers arrive (deterministic). Each has WTP ~ Uniform[0, maxWTP].
        They buy if WTP ≥ price, so P(buy) = (maxWTP − price) / maxWTP.

        Daily sales ~ **Binomial(N, p)**: mean = Np, variance = Np(1−p)

        Over deterministic lead time L:

        > **Expected demand during lead time** = NpL
        >
        > **Std dev of demand during lead time** = $\sqrt{Np(1-p)L}$
        >
        > **Safety Stock** = z × $\sqrt{Np(1-p)L}$
        >
        > **Reorder Point** = NpL + z × $\sqrt{Np(1-p)L}$

        Where z = service level z-score (1.65 for 95%, 1.96 for 97.5%, 2.33 for 99%)

        If both products running, factory alternates batches → effective cycle = 5 days + 1 day transit = **6 days lead time**

        **End-game protocol:**
        1. Calculate days remaining
        2. Set reorder point to **-1** (stop production) when: `days_remaining < lead_time + (inventory / daily_demand)`
        3. This ensures you sell through remaining stock with zero left over
        """)

        st.markdown("#### Reorder Point Calculator")
        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            calc_price = st.number_input("Your Price ($)", value=450, step=50)
            calc_max_wtp = st.number_input("Max WTP ($)", value=800, step=50)
        with rc2:
            calc_market = st.number_input("Market Size", value=300000, step=10000)
            calc_both = st.checkbox("Both products running?", value=True)
            calc_service = st.selectbox("Service Level", [90, 95, 97.5, 99], index=1,
                                         format_func=lambda x: f"{x}%")
        # z-scores for common service levels
        z_map = {90: 1.28, 95: 1.65, 97.5: 1.96, 99: 2.33}
        z_score = z_map[calc_service]
        lead_time = 6.0 if calc_both else 3.5
        N_arrivals = 0.0001 * calc_market
        p_buy = (calc_max_wtp - calc_price) / calc_max_wtp if calc_max_wtp > calc_price else 0
        daily_demand_calc = N_arrivals * p_buy
        # Demand during lead time: Binomial sum over L days
        demand_during_lt = daily_demand_calc * lead_time
        std_during_lt = (N_arrivals * p_buy * (1 - p_buy) * lead_time) ** 0.5
        safety_stock = z_score * std_during_lt
        reorder = demand_during_lt + safety_stock
        with rc3:
            st.metric("Daily Demand (Np)", f"{daily_demand_calc:,.1f} units")
            st.metric("Lead Time", f"{lead_time} days")
            st.metric("Safety Stock (z×σ)", f"{safety_stock:,.0f} units")
            st.metric("Recommended Reorder Point", f"{reorder:,.0f} units")
            st.caption(f"z={z_score}, σ_LT={std_during_lt:.1f}, P(buy)={p_buy:.1%}")

    # ── Strategy 3: Capacity ──────────────────────────────────────────────────
    with st.expander("**3. CAPACITY AS BOTTLENECK**"):
        st.markdown("""
        **Factory throughput:**
        - Batch = 100 units in 2.5 days → **40 units/day max** (single product)
        - Both products → alternating → **~20 units/day each**

        **Key question:** Can your factory keep up with demand at your chosen price?

        | Scenario | Demand/day | Capacity/day | Status |
        |---|---|---|---|
        | Hormone only, P=400, maxWTP=800 | 15 | 40 | Comfortable |
        | Both products, P=400/400 | 15 + 7 = 22 | 40 (alternating) | Tight but OK |
        | Lower prices to move volume | 25+ | 20 each | **Bottleneck!** |

        **If bottlenecked:** Raise prices. It's better to sell fewer units at higher margin than to stock out.

        **In Production Game:** You can build new factories (90-day build time, requires land + capital + daily labor).
        Only worth it if enough game days remain to recoup the investment.
        """)

    # ── Strategy 4: Trading ───────────────────────────────────────────────────
    with st.expander("**4. TRADING GAME — Negotiate Like You Learned in 38803**"):
        st.markdown("""
        **When trading opens, you can buy/sell products with other teams.**

        #### Shipping Costs
        | Mode | Cost | Transit | Best for |
        |---|---|---|---|
        | Mail (same region) | Free | 1 day | Internal |
        | Mail (between regions) | $400/10 units = $40/unit | 3 days | Small/fast orders |
        | Container (between regions) | $10,000/1000 units = $10/unit | 21 days | Bulk/long games |

        #### As a BUYER (importing Hormone):
        - Negotiate wholesale price **well below** your retail price
        - Your margin = retail price - wholesale price - shipping cost
        - Prefer mail for short games (3 days vs 21 days for containers)
        - Set reorder points carefully — you don't control their production

        #### As a SELLER (exporting your Specialty):
        - Your specialty is **proprietary** — leverage this!
        - Use **franchise fees** (one-time or ongoing) for guaranteed income
        - Use **revenue sharing** (% of retail) to participate in upside
        - Set **termination penalties** to lock in agreements
        - Set **retail price limits** to prevent buyers from undercutting your home market

        #### BATNA Analysis (from Negotiations):
        - Your BATNA = sell everything retail yourself
        - Their BATNA = not having your product at all
        - **ZOPA** = wholesale price between your marginal cost ($100) and their expected retail margin
        - Look for **integrative deals**: "I'll supply you Specialty if you supply me Hormone"
        """)

    # ── Strategy 5: Financial ─────────────────────────────────────────────────
    with st.expander("**5. CASH MANAGEMENT — The Objective Function**"):
        st.markdown("""
        **Objective = Maximize ending cash balance.**

        #### The Cash Flow Cycle
        ```
        Day 0: Order materials ($100/unit)
        Day 0-2.5: Production (daily labor expense running)
        Day 2.5: Batch complete, ships to DC
        Day 3.5: Arrives at DC, available for sale
        Day 3.5+: Sales generate revenue (immediate cash)
        Day 15: Materials payable comes due
        Day 90: Tax payment (35% of quarterly profit)
        ```

        #### Rules
        - **NEVER** hit emergency loans → 40% APR destroys your cash
        - Cash earns 3% APR → sitting on cash is not terrible
        - Cost of capital = 10% APR → use this for NPV calculations
        - Raw materials payable in 15 days → you get a working capital float
        - Other expenses payable in 30 days (Trading Game)

        #### Financial Statements to Monitor
        1. **Income Statement** → Are you profitable?
        2. **Balance Sheet** → Watch cash, inventory, payables
        3. **Working Capital Report** → Current ratio > 1.0 always
        4. **Market Grid** → Your market share vs competitors
        5. **Price Response** → How demand responds to your price changes
        """)

    # ── Strategy 6: Expansion ─────────────────────────────────────────────────
    with st.expander("**6. EXPANSION DECISIONS — NPV or Nothing**"):
        st.markdown("""
        **From Corporate Finance: Only invest if NPV > 0 given remaining game time.**

        #### New Factory
        - Requires: land ($100K) + capital + daily labor
        - Build time: **90 days**
        - Only viable in longer games
        - Can build in **other regions** to serve those markets directly

        #### New Distribution Center
        - Opens retail sales in another region
        - Infinite capacity (no need for more than 1 per region)
        - Build time required — plan ahead

        #### Product Design & Development
        - Create new products (takes time + R&D spend)
        - Consider: does the margin justify the investment given time remaining?

        #### Advertising
        - Increases demand (shifts demand curve right)
        - Costs daily $ per SKU per DC
        - **Test small, measure ROI** via Price Response report
        - Only worth it if you have excess capacity

        #### Bonds
        - Issue bonds to raise capital for expansion
        - Consider credit rating impact
        - Only issue if the investment NPV exceeds interest costs
        """)

    # ── Strategy 7: End-game ──────────────────────────────────────────────────
    with st.expander("**7. END-GAME EXECUTION — The Final 20% of Game Time**", expanded=True):
        st.markdown("""
        **All inventory becomes obsolete at game end. This phase determines winners.**

        #### End-Game Countdown Checklist

        **When ~30% of game time remains:**
        - [ ] Review inventory levels and production pipeline
        - [ ] Calculate how many days of inventory you have at current demand
        - [ ] Begin reducing reorder points

        **When ~15% of game time remains:**
        - [ ] Set reorder point to **-1** (stop production) — account for 3.5-day lead time
        - [ ] Start reducing prices to accelerate sales
        - [ ] Terminate costly shipping agreements (if penalty < ongoing cost)

        **Final days:**
        - [ ] **Slash prices** aggressively — selling at $101 is better than holding at $0
        - [ ] Verify no batches are in-process at the factory
        - [ ] Collect all receivables from trading partners
        - [ ] Pay off any remaining debts
        - [ ] Verify final cash position on Scoreboard

        #### The Math on End-Game Pricing
        If you have 100 units of inventory with 2 days left:
        - At current price $450: sell ~15 units = **$6,750 revenue**
        - At fire-sale price $150: sell ~24 units = **$3,600 revenue**
        - Unsold units at $450: 85 units x $100 cost = **-$8,500 wasted**
        - Unsold units at $150: 76 units x $100 cost = **-$7,600 wasted**

        **Lower price = more units sold = less wasted inventory = more net cash.**
        But the optimal fire-sale price depends on remaining time and inventory — use the Price Response report.
        """)

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # TEAM ROLES & DECISION FRAMEWORK
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("Team Panem (1-29)")

    st.markdown("""
    <div style="background: linear-gradient(135deg, #1a3c5e 0%, #2d5a8e 100%);
        color: white; border-radius: 12px; padding: 1.2rem; margin-bottom: 1rem;">
        <h4 style="color: #ffd700; margin-top: 0;">Team Panem</h4>
        <p style="margin-bottom: 0; font-size: 0.95rem;">
        Chris Ma &nbsp;|&nbsp; Shiyuan Tian &nbsp;|&nbsp; Yohei Nakadate &nbsp;|&nbsp;
        <strong>Derrick Teo</strong> &nbsp;|&nbsp; Jack Meng &nbsp;|&nbsp; Jason Weng
        </p>
    </div>
    """, unsafe_allow_html=True)

    role_col1, role_col2 = st.columns(2)
    with role_col1:
        st.markdown("#### Suggested Role Assignments")
        st.markdown("*Roles TBD — discuss with team and update. 6 members = can double up on key areas.*")
        roles_data = [
            {"Role": "CEO / Strategist", "Suggested": "TBD",
             "Focus": "Overall strategy, competitor monitoring, final calls",
             "Key Reports": "Scoreboard, Market Grid"},
            {"Role": "CFO", "Suggested": "TBD",
             "Focus": "Cash management, financial statements, bonds, tax planning",
             "Key Reports": "Financial Statements, Working Capital"},
            {"Role": "COO", "Suggested": "TBD",
             "Focus": "Factory ops, reorder points, inventory, production shipping",
             "Key Reports": "Inventory Status, Inventory Activities"},
            {"Role": "CMO / Pricing", "Suggested": "TBD",
             "Focus": "Pricing decisions, advertising, market research, focus groups",
             "Key Reports": "Price Response, Product Analysis"},
            {"Role": "Head of Trade", "Suggested": "TBD",
             "Focus": "Negotiate shipping agreements, manage inter-team deals",
             "Key Reports": "Shipping Agreement Sankey"},
            {"Role": "Analyst / Scout", "Suggested": "TBD",
             "Focus": "Monitor competitors' financials (quarterly), track market share, model scenarios",
             "Key Reports": "Market Grid, competitor Financial Statements"},
        ]
        st.dataframe(pd.DataFrame(roles_data), use_container_width=True, hide_index=True)

    with role_col2:
        st.markdown("""
        #### Decision Process (from the syllabus):
        > *"Establishing an organized decision-making and conflict-resolution process
        > as a team will help you avoid frustrations... This has been a key competitive
        > advantage for teams in the past."*

        **With 6 members, your advantage is bandwidth.** Pre-agree on:
        1. **Who has final say on pricing?** (CMO, with CEO override)
        2. **Who approves trade deals?** (Head of Trade + CEO sign-off)
        3. **Who monitors cash?** (CFO raises alarm if < threshold)
        4. **How do you resolve disagreements?** (Majority vote, 30-second timer)
        5. **Who watches competitors?** (Analyst tracks scoreboard + market grid)
        6. **Communication:** Use the in-game internal chat for real-time coordination

        **Cross-train everyone** — the syllabus explicitly says teams that are
        cross-trained on all aspects perform better. Roles will shift as the business grows.

        #### 6-Person Advantage
        - **Always-on monitoring:** Someone can always watch the scoreboard while others execute
        - **Parallel negotiations:** Trade with 2-3 teams simultaneously
        - **Shift coverage:** During the 2-hour game sessions, rotate breaks
        """)

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # COURSE INTEGRATION MAP
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("How Each Course Powers Your Game Decisions")

    course_game_map = [
        {"Course": "Microeconomics", "Game Application": "Demand curves, WTP, price elasticity, Nash equilibrium in pricing wars",
         "When It Matters": "Every pricing decision"},
        {"Course": "Operations Management", "Game Application": "Little's Law, bottleneck analysis, reorder points, inventory management",
         "When It Matters": "Factory & inventory management"},
        {"Course": "Competitive Strategy", "Game Application": "Porter's 5 Forces, entry barriers, competitive positioning",
         "When It Matters": "Trading game, expansion decisions"},
        {"Course": "Corporate Finance", "Game Application": "NPV for investments, cost of capital (10%), WACC, bond decisions",
         "When It Matters": "Factory/DC expansion, bonds, valuation"},
        {"Course": "Financial Accounting", "Game Application": "Reading income statements, balance sheets, working capital ratios",
         "When It Matters": "Monitoring financial health"},
        {"Course": "Pricing Strategies", "Game Application": "EVC, price discrimination, competitor reaction, elasticity optimization",
         "When It Matters": "Setting and adjusting prices"},
        {"Course": "Marketing Management", "Game Application": "STP, positioning, advertising ROI, product line decisions",
         "When It Matters": "Product design, advertising spend"},
        {"Course": "Managerial Accounting", "Game Application": "Cost analysis, ABC, variance analysis, performance metrics",
         "When It Matters": "Tracking profitability by product/region"},
        {"Course": "Negotiations", "Game Application": "BATNA, ZOPA, integrative bargaining, coalition building",
         "When It Matters": "Every trading negotiation"},
        {"Course": "Managerial Decision Making", "Game Application": "Avoiding anchoring bias, prospect theory, System 1 vs 2",
         "When It Matters": "Under time pressure — slow down and think"},
        {"Course": "Financial Strategy", "Game Application": "Capital structure, leverage, debt capacity",
         "When It Matters": "Bond issuance decisions"},
        {"Course": "Macroeconomics", "Game Application": "Interest rate environment, risk assessment",
         "When It Matters": "Scenario planning"},
    ]
    st.dataframe(pd.DataFrame(course_game_map), use_container_width=True, hide_index=True, height=460)

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # PRE-READ PRIORITY
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("Pre-Read Priority Order")

    preread_data = [
        {"Priority": "1", "Document": "Monopoly Game Case Brief", "Why": "Understand the core mechanics — pricing, inventory, production",
         "Status": "Must Read"},
        {"Priority": "2", "Document": "Trading Game Case Brief", "Why": "Shipping agreement terms are complex — learn them cold",
         "Status": "Must Read"},
        {"Priority": "3", "Document": "Gleacher Game Player Manual", "Why": "Know the UI — don't waste game time clicking around",
         "Status": "Must Read"},
        {"Priority": "4", "Document": "Note on Microeconomics for Strategists", "Why": "Demand/supply/pricing foundations for the game",
         "Status": "Must Read"},
        {"Priority": "5", "Document": "D0: Managing Inventories / Newsvendor Model", "Why": "Reorder point optimization",
         "Status": "High"},
        {"Priority": "6", "Document": "D0: Note on Competitive Positioning", "Why": "How to position vs other teams in trading",
         "Status": "High"},
        {"Priority": "7", "Document": "D0: Financial Statement Analysis", "Why": "Reading the in-game financial reports",
         "Status": "High"},
        {"Priority": "8", "Document": "D0: Dynamics of Price Competition (Garicano & Gertner)", "Why": "How price wars evolve — avoid them",
         "Status": "Medium"},
        {"Priority": "9", "Document": "D0: Can You Say What Your Strategy Is?", "Why": "Final project narrative — articulate your strategy",
         "Status": "Medium"},
        {"Priority": "10", "Document": "D0: What Is Strategy? (Porter)", "Why": "Backdrop for final project analysis",
         "Status": "Medium"},
    ]
    st.dataframe(pd.DataFrame(preread_data), use_container_width=True, hide_index=True)

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # COMPETITOR INTELLIGENCE
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("Competitor Intelligence")

    PHOTO_BASE = "https://raw.githubusercontent.com/derrick-pixel/booth-emba-analytics/main/photos"

    TEAMS = {
        "B612": {"id": "1-25", "color": "#6b3fa0",
                  "members": [("Masa Ishigaki", "masa_ishigaki"), ("Takeshi Tanaka", "takeshi_tanaka"),
                              ("Hyeyoung Lee", "hyeyoung_lee"), ("Valerii Egorov", "valerii_egorov"),
                              ("Carlos Naibryf", "carlos_naibryf"), ("Fengshu Jin", "fengshu_jin")],
                  "theme": "The Little Prince"},
        "Dune": {"id": "1-26", "color": "#b8860b",
                  "members": [("Jo Hayes", "jo_hayes"), ("Jenny Yang", "jenny_yang"),
                              ("Morry Mori", "morry_mori"), ("Suliya Suliya", "suliya_suliya"),
                              ("Betty Wang", "betty_wang"), ("Kosuke Okura", "kosuke_okura")],
                  "theme": "Dune"},
        "Globex": {"id": "1-27", "color": "#c44e00",
                    "members": [("Prashanth Palepu", "prashanth_palepu"), ("Lisa Lau", "lisa_lau"),
                                ("George Chia", "george_chia"), ("Jeffrey Chen", "jeffrey_chen"),
                                ("Kacey Du", "kacey_du"), ("Lambert Xu", "lambert_xu")],
                    "theme": "The Simpsons"},
        "Gotham": {"id": "1-28", "color": "#333333",
                    "members": [("Inge Supatra", "inge_supatra"), ("Bryan Wong", "bryan_wong"),
                                ("Delphine Terrien", "delphine_terrien"), ("Benjamin Jiang", "benjamin_jiang"),
                                ("Dai Kato", "dai_kato"), ("Ngiap Seng Khoo", "ngiap_seng_khoo")],
                    "theme": "Batman / Gotham"},
        "Panem": {"id": "1-29", "color": "#800000",
                   "members": [("Chris Ma", "chris_ma"), ("Shiyuan Tian", "shiyuan_tian"),
                               ("Yohei Nakadate", "yohei_nakadate"), ("Derrick Teo", "derrick_teo"),
                               ("Jack Meng", "jack_meng"), ("Jason Weng", "jason_weng")],
                   "theme": "The Hunger Games",
                   "is_us": True},
        "Vulcan": {"id": "1-31", "color": "#1a3c5e",
                    "members": [("Eric Zhang", "eric_zhang"), ("Jony Hu", "jony_hu"),
                                ("Laurence Zhu", "laurence_zhu"), ("Yehwan Kim", "yehwan_kim"),
                                ("Tom Hsieh", "tom_hsieh"), ("Sudeep Rathee", "sudeep_rathee")],
                    "theme": "Star Trek"},
        "Westeros": {"id": "1-32", "color": "#2d6a2e",
                      "members": [("Kosuke Shinohara", "kosuke_shinohara"), ("Ken Ng", "ken_ng"),
                                  ("Taku Yasuda", "taku_yasuda"), ("Ryan Kim", "ryan_kim"),
                                  ("Andy Yoo", "andy_yoo"), ("Jumpei Maruyama", "jumpei_maruyama")],
                      "theme": "Game of Thrones"},
        "Zion": {"id": "1-33", "color": "#0e7c7b",
                  "members": [("Ken Chew", "ken_chew"), ("Ryo Aikawa", "ryo_aikawa"),
                              ("Louis Woenardi", "louis_woenardi"), ("Dimas Purnama", "dimas_purnama"),
                              ("Chris Kwan", "chris_kwan"), ("Alex Wang", "alex_wang")],
                  "theme": "The Matrix"},
    }

    import base64, os
    PHOTO_DIR = os.path.join(os.path.dirname(__file__), "photos")

    def get_photo_base64(file_key):
        path = os.path.join(PHOTO_DIR, f"{file_key}.jpg")
        if os.path.exists(path):
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode()
        return None

    # Team overview cards in 4-column grid, 2 rows of 3 photos each
    team_items = list(TEAMS.items())
    for row_start in range(0, len(team_items), 4):
        cols = st.columns(4)
        for col_idx in range(4):
            if row_start + col_idx >= len(team_items):
                break
            team_name, team_data = team_items[row_start + col_idx]
            is_us = team_data.get("is_us", False)
            border = "3px solid #ffd700" if is_us else "1px solid rgba(255,255,255,0.1)"
            badge = " (US)" if is_us else ""

            members = team_data["members"]
            row1 = members[:3]
            row2 = members[3:]

            def build_photo_row(member_list):
                html = ""
                for display_name, file_key in member_list:
                    b64 = get_photo_base64(file_key)
                    if b64:
                        img_tag = f'<img src="data:image/jpeg;base64,{b64}" style="width:71px;height:71px;border-radius:50%;object-fit:cover;border:1.5px solid rgba(255,255,255,0.5);"/>'
                    else:
                        initials = "".join(w[0] for w in display_name.split())
                        img_tag = f'<div style="width:71px;height:71px;border-radius:50%;background:rgba(255,255,255,0.2);display:flex;align-items:center;justify-content:center;font-weight:700;font-size:0.65rem;">{initials}</div>'
                    html += f'<div style="text-align:center;flex:1;">{img_tag}<div style="font-size:0.72rem;margin-top:2px;line-height:1.15;">{display_name}</div></div>'
                return html

            with cols[col_idx]:
                st.markdown(f"""
                <div style="background:{team_data['color']};color:white;border-radius:8px;
                    padding:0.6rem 0.8rem;margin-bottom:0.5rem;border:{border};">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem;">
                        <span style="font-weight:700;font-size:0.95rem;">{team_name}{badge}</span>
                        <span style="font-size:0.65rem;opacity:0.6;">{team_data['id']}</span>
                    </div>
                    <div style="display:flex;gap:2px;margin-bottom:6px;">{build_photo_row(row1)}</div>
                    <div style="display:flex;gap:2px;">{build_photo_row(row2)}</div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("")
    st.markdown("**8 teams x 6 members = 48 players** competing in the same simulation.")
    st.markdown("In the Trading Game & Competition, these are the teams you'll negotiate with and compete against.")

    # Competitor tracking table (placeholder for live game)
    with st.expander("**Live Competitor Tracker** (update during game)"):
        st.markdown("*During the game, update this tracker with intel from the Scoreboard and Market Grid reports.*")
        tracker_data = []
        for team_name, team_data in TEAMS.items():
            tracker_data.append({
                "Team": team_name,
                "Seat": team_data["id"],
                "Cash Position": "—",
                "Strategy Observed": "—",
                "Pricing (Hormone)": "—",
                "Pricing (Specialty)": "—",
                "Trading Partner?": "—",
                "Threat Level": "—",
            })
        st.dataframe(pd.DataFrame(tracker_data), use_container_width=True, hide_index=True)
        st.caption("Update this table manually during game play to track competitor moves.")

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # THE ONE-LINE STRATEGY
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("""
    <div style="background: linear-gradient(135deg, #800000 0%, #b22222 100%);
        color: white; border-radius: 12px; padding: 2rem; text-align: center; margin: 1rem 0;">
        <h3 style="color: white; margin-top: 0;">The One-Line Strategy</h3>
        <p style="font-size: 1.2rem; margin-bottom: 0;">
        Price at ~half of max WTP, never stock out, manage cash obsessively,
        trade smartly, and wind down production before game end so you finish
        with <strong>maximum cash and zero inventory</strong>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # WHY THIS STRATEGY WORKS — Deep Dive Cards
    # ══════════════════════════════════════════════════════════════════════════
    with st.expander("**Why does this strategy work? (Deep Dive)**"):

        DEEP_CARDS = [
            {
                "title": "Price at ~half of max WTP",
                "color": "#800000",
                "ref": "1. PRICING — The #1 Lever",
                "body": """WTP is <b>uniformly distributed \\$0 to max_WTP</b>.<br><br>
                Revenue = 0.0001 x mkt x [(maxWTP - P) / maxWTP] x P<br><br>
                Quadratic in P → derivative = 0 →<br>
                <b>P* = max_WTP / 2</b> (revenue-max)<br><br>
                With \\$100 marginal cost:<br>
                <b>P*_profit = (max_WTP + 100) / 2</b><br><br>
                <i>e.g. maxWTP=\\$800 → profit-max = \\$450</i><br><br>
                <span style="opacity:0.6;font-size:0.75rem;">Microeconomics + Pricing Strategies</span>""",
            },
            {
                "title": "Never stock out",
                "color": "#0e7c7b",
                "ref": "2. INVENTORY MANAGEMENT",
                "body": """<i>"If no on-hand inventory, the customer is lost <b>forever</b>."</i><br><br>
                Not "comes back tomorrow" — <b>permanently gone</b>. Stockouts compound into
                massive market shrinkage.<br><br>
                <b>Safety Stock = z × √(Np(1-p)L)</b><br>
                where N=arrivals/day, p=P(buy), L=lead time, z=service level<br><br>
                <b>Reorder Point = NpL + Safety Stock</b><br><br>
                Cost of holding (\\$100/unit) is trivial vs permanently losing customers.<br><br>
                <span style="opacity:0.6;font-size:0.75rem;">Operations Management (Little's Law, newsvendor)</span>""",
            },
            {
                "title": "Manage cash obsessively",
                "color": "#1a3c5e",
                "ref": "5. CASH MANAGEMENT",
                "body": """<b>Cash IS the score.</b> Not revenue, not profit.<br><br>
                <table style="width:100%;font-size:0.8rem;color:white;border-collapse:collapse;">
                <tr><td>Cash interest</td><td><b>3% APR</b></td></tr>
                <tr><td>Emergency loans</td><td><b style="color:#ff6b6b;">40% APR</b></td></tr>
                <tr><td>Cost of capital</td><td><b>10% APR</b></td></tr>
                <tr><td>Tax rate</td><td><b>35% quarterly</b></td></tr>
                </table><br>
                <b>37% spread</b> between earning & borrowing. Running dry = catastrophe.<br><br>
                <span style="opacity:0.6;font-size:0.75rem;">Corporate Finance + Financial Strategy</span>""",
            },
            {
                "title": "Trade smartly",
                "color": "#6b3fa0",
                "ref": "4. TRADING GAME",
                "body": """Your specialty is <b>proprietary</b> — leverage it.<br><br>
                <b>BATNA:</b> Sell retail yourself (140K market)<br>
                <b>Their BATNA:</b> Zero revenue from your product<br>
                <b>ZOPA:</b> \\$100 (your cost) to their retail price<br><br>
                Shipping eats margins: Mail \\$40/unit (3d), Container \\$10/unit (21d).<br>
                Bad deals lock you in. Price wars → zero profit.<br>
                <b>Integrative deals create value:</b> swap Specialty for Hormone.<br><br>
                <span style="opacity:0.6;font-size:0.75rem;">Negotiations (BATNA/ZOPA) + Competitive Strategy</span>""",
            },
            {
                "title": "Wind down before game end",
                "color": "#b22222",
                "ref": "7. END-GAME EXECUTION",
                "body": """<i>"All inventory becomes obsolete."</i><br><br>
                Every unsold unit = <b>\\$100 wasted</b>. 500 units = \\$50K lost.<br><br>
                <b>Stop when:</b> days_remaining &lt; lead_time + (inventory / daily_demand)<br><br>
                <i>e.g. lead=3.5d, inv=200, demand=15/d → stop at 16.8 days left</i><br><br>
                <b>End-game:</b> slash prices — selling at \\$101 beats holding at \\$0.
                This is the newsvendor model with zero salvage value.<br><br>
                <span style="opacity:0.6;font-size:0.75rem;">Operations Mgmt (newsvendor) + Pricing Strategies</span>""",
            },
            {
                "title": "How it all connects",
                "color": "#333333",
                "ref": "",
                "body": """Pricing ──→ <b>Cash inflow</b><br>Inventory ──→ <b>Preserves market</b><br>Cash mgmt ──→ <b>Protects balance</b><br>Trading ──→ <b>New revenue</b><br>Wind-down ──→ <b>Inv → Cash</b><br><br><span style="padding:0.4rem 0.8rem;background:rgba(255,255,255,0.15);border-radius:6px;font-size:1rem;font-weight:700;">FINAL CASH = SCORE</span><br><br><b>All 5 cylinders must fire simultaneously.</b> That's why this is the capstone.""",
            },
        ]

        # Render cards in rows of 3
        for row_start in range(0, len(DEEP_CARDS), 3):
            row_cards = DEEP_CARDS[row_start:row_start + 3]
            card_cols = st.columns(len(row_cards))
            for ci, card in enumerate(row_cards):
                ref_line = f'See above: {card["ref"]}' if card["ref"] else ""
                with card_cols[ci]:
                    card_html = f'<div style="background:{card["color"]};color:white;border-radius:10px;padding:1rem;min-height:320px;font-size:0.8rem;line-height:1.45;">'
                    card_html += f'<h4 style="color:white;margin:0 0 0.3rem 0;font-size:0.95rem;">{card["title"]}</h4>'
                    if ref_line:
                        card_html += f'<span style="font-size:0.65rem;opacity:0.5;">{ref_line}</span><br>'
                    card_html += card["body"]
                    card_html += '</div>'
                    st.markdown(card_html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 0.5: TRIAL WAR ROOM
# ══════════════════════════════════════════════════════════════════════════════

elif page == "⚔️ 12 Trial War Room":
    st.markdown('<p class="big-header">12 Trial War Room</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Monopoly & Trading Game — April 12 (WTP, uniform demand, simple pricing)</p>',
                unsafe_allow_html=True)
    st.markdown("")

    # ── Game Parameters (all adjustable) ─────────────────────────────────────
    st.subheader("Game Parameters")
    st.caption("Adjust these to match the current game scenario — all sections below recalculate automatically")

    gp_col1, gp_col2, gp_col3, gp_col4, gp_col5 = st.columns(5)
    with gp_col1:
        H_MAX_WTP = st.number_input("Hormone Max WTP ($)", value=500, step=50, key="gp_hmaxwtp")
    with gp_col2:
        S_MAX_WTP = st.number_input("Specialty Max WTP ($)", value=800, step=50, key="gp_smaxwtp",
                                     help="Proprietary product — likely higher than Hormone. Discover via Price Response report.")
    with gp_col3:
        MC_PRODUCTION = st.number_input("Materials Cost ($/unit)", value=100, step=10, key="gp_mc")
    with gp_col4:
        BATCH_SIZE = st.number_input("Batch Size (units)", value=100, step=10, key="gp_batch")
    with gp_col5:
        PRODUCTION_DAYS = st.number_input("Production Time (days)", value=2.5, step=0.5, key="gp_proddays")

    gp2_col1, gp2_col2, gp2_col3, gp2_col4, gp2_col5 = st.columns(5)
    with gp2_col1:
        ARRIVAL_RATE = st.number_input("Arrival Rate", value=0.0001, step=0.00001, format="%.5f", key="gp_arrival")
    with gp2_col2:
        HORMONE_MARKET = st.number_input("Hormone Market Size", value=300000, step=10000, key="gp_hmkt")
    with gp2_col3:
        SPECIALTY_MARKET = st.number_input("Specialty Market Size", value=140000, step=10000, key="gp_smkt")
    with gp2_col4:
        BASE_LEAD_TIME = st.number_input("Lead Time (days)", value=3.5, step=0.5, key="gp_lt")
    with gp2_col5:
        COST_OF_CAPITAL = st.number_input("Cost of Capital (%)", value=10.0, step=1.0, key="gp_coc")

    CAPACITY_PER_DAY = BATCH_SIZE / PRODUCTION_DAYS if PRODUCTION_DAYS > 0 else 40

    SHIPPING = {
        "Same region": {"cost_per_unit": 0, "days": 1, "label": "Free, 1 day"},
        "Mail (between regions)": {"cost_per_unit": 40, "days": 3, "label": "$40/unit, 3 days"},
        "Container (between regions)": {"cost_per_unit": 10, "days": 21, "label": "$10/unit, 21 days"},
    }

    # ── Dynamic banner (recalculates from inputs above) ──────────────────────
    _h_opt = (H_MAX_WTP + MC_PRODUCTION) / 2
    _s_opt = (S_MAX_WTP + MC_PRODUCTION) / 2
    _h_demand = ARRIVAL_RATE * HORMONE_MARKET * (H_MAX_WTP - _h_opt) / H_MAX_WTP if H_MAX_WTP > _h_opt else 0
    _s_demand = ARRIVAL_RATE * SPECIALTY_MARKET * (S_MAX_WTP - _s_opt) / S_MAX_WTP if S_MAX_WTP > _s_opt else 0
    _excess = CAPACITY_PER_DAY - _h_demand - _s_demand

    st.markdown(f"""
<div style="background:linear-gradient(135deg,#800000,#b22222);color:white;
    border-radius:12px;padding:1.2rem 1.5rem;margin-bottom:1rem;">
<h4 style="color:#ffd700;margin:0 0 0.5rem 0;">Derived Metrics (auto-calculated)</h4>
<div style="display:flex;gap:2rem;flex-wrap:wrap;">
<div><span style="opacity:0.7;">Hormone Optimal</span><br><b style="font-size:1.3rem;">${_h_opt:.0f}</b><br><span style="font-size:0.75rem;opacity:0.6;">({H_MAX_WTP}+{MC_PRODUCTION})/2</span></div>
<div><span style="opacity:0.7;">Specialty Optimal</span><br><b style="font-size:1.3rem;">${_s_opt:.0f}</b><br><span style="font-size:0.75rem;opacity:0.6;">({S_MAX_WTP}+{MC_PRODUCTION})/2</span></div>
<div><span style="opacity:0.7;">Hormone Demand</span><br><b style="font-size:1.3rem;">{_h_demand:.1f} /day</b><br><span style="font-size:0.75rem;opacity:0.6;">{ARRIVAL_RATE*HORMONE_MARKET:.0f} × ({H_MAX_WTP}-{_h_opt:.0f})/{H_MAX_WTP}</span></div>
<div><span style="opacity:0.7;">Specialty Demand</span><br><b style="font-size:1.3rem;">{_s_demand:.1f} /day</b><br><span style="font-size:0.75rem;opacity:0.6;">{ARRIVAL_RATE*SPECIALTY_MARKET:.0f} × ({S_MAX_WTP}-{_s_opt:.0f})/{S_MAX_WTP}</span></div>
<div><span style="opacity:0.7;">Factory Capacity</span><br><b style="font-size:1.3rem;">{CAPACITY_PER_DAY:.0f} /day</b><br><span style="font-size:0.75rem;opacity:0.6;">{BATCH_SIZE} / {PRODUCTION_DAYS} days</span></div>
<div><span style="opacity:0.7;">Excess Capacity</span><br><b style="font-size:1.3rem;">{_excess:.1f} /day</b><br><span style="font-size:0.75rem;opacity:0.6;">{CAPACITY_PER_DAY:.0f} - {_h_demand:.1f} - {_s_demand:.1f}</span></div>
</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 0: WTP DISCOVERY FROM PRICE RESPONSE DATA
    # ══════════════════════════════════════════════════════════════════════════
    with st.expander("**WTP Discovery Tool** — estimate Max WTP from Price Response data", expanded=False):
        st.markdown("""
Enter observed (Price, Units Sold per Day) data points from the **Price Response** report.
The demand model is `Q = N × (maxWTP − P) / maxWTP`, which is linear: `P = a − bQ`.
Linear regression gives **a = Max WTP** (the price intercept where demand hits zero).
        """)

        wtp_disc_product = st.radio("Discover WTP for", ["Hormone", "Specialty"],
                                     horizontal=True, key="wtp_disc_prod")

        st.markdown("**Enter observed data points** (from Price Response report)")
        wtp_col1, wtp_col2 = st.columns([1, 2])
        with wtp_col1:
            wtp_n_points = st.number_input("Number of observations", min_value=2, max_value=20,
                                            value=4, step=1, key="wtp_npts")
            wtp_prices = []
            wtp_demands = []
            for i in range(int(wtp_n_points)):
                pc1, pc2 = st.columns(2)
                with pc1:
                    p_val = st.number_input(f"Price {i+1} ($)", value=200 + i * 100, step=25,
                                             key=f"wtp_p_{i}")
                with pc2:
                    q_val = st.number_input(f"Qty/day {i+1}", value=max(0.0, 15.0 - i * 4.0),
                                             step=0.5, format="%.1f", key=f"wtp_q_{i}")
                wtp_prices.append(p_val)
                wtp_demands.append(q_val)

        with wtp_col2:
            prices_arr = np.array(wtp_prices)
            demands_arr = np.array(wtp_demands)

            # Filter out zero-demand points for regression (they're censored)
            mask = demands_arr > 0
            if mask.sum() >= 2:
                # Fit P = a - bQ (linear regression of P on Q)
                coeffs = np.polyfit(demands_arr[mask], prices_arr[mask], 1)
                b_slope = coeffs[0]  # negative slope
                a_intercept = coeffs[1]  # = estimated Max WTP

                # Also fit Q = α + βP to get x-intercept
                coeffs_qp = np.polyfit(prices_arr[mask], demands_arr[mask], 1)
                beta_qp = coeffs_qp[0]
                alpha_qp = coeffs_qp[1]
                wtp_from_q = -alpha_qp / beta_qp if beta_qp != 0 else 0  # price where Q=0

                estimated_wtp = max(a_intercept, wtp_from_q)  # take the more conservative
                r_squared = 1 - np.sum((prices_arr[mask] - np.polyval(coeffs, demands_arr[mask]))**2) / \
                            np.sum((prices_arr[mask] - np.mean(prices_arr[mask]))**2) if len(prices_arr[mask]) > 1 else 0

                # Plot
                fig_wtp = go.Figure()
                # Data points
                fig_wtp.add_trace(go.Scatter(x=demands_arr, y=prices_arr, mode="markers",
                                              marker=dict(size=10, color="#800000"),
                                              name="Observed"))
                # Fitted line extended to Q=0
                q_fit = np.linspace(0, max(demands_arr) * 1.2, 100)
                p_fit = a_intercept + b_slope * q_fit
                fig_wtp.add_trace(go.Scatter(x=q_fit, y=p_fit, mode="lines",
                                              line=dict(color="#2d6a2e", width=2, dash="dash"),
                                              name=f"Fitted: P = {a_intercept:.0f} − {abs(b_slope):.1f}Q"))
                # Max WTP marker
                fig_wtp.add_trace(go.Scatter(x=[0], y=[a_intercept], mode="markers",
                                              marker=dict(size=14, color="#ffd700", symbol="star"),
                                              name=f"Max WTP = ${a_intercept:,.0f}"))
                fig_wtp.add_hline(y=a_intercept, line_dash="dot", line_color="rgba(255,200,50,0.5)")
                fig_wtp.update_layout(height=350, xaxis_title="Quantity (units/day)",
                                       yaxis_title="Price ($)", yaxis_tickformat="$,.0f",
                                       margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig_wtp, use_container_width=True)

                # Results
                wtp_color = "#2d6a2e" if r_squared > 0.8 else ("#b8860b" if r_squared > 0.5 else "#b22222")
                st.markdown(f"""
<div style="border:2px solid {wtp_color};border-radius:10px;padding:1rem;">
<div style="display:flex;gap:2rem;align-items:center;">
<div><span style="opacity:0.7;">Estimated Max WTP</span><br>
<b style="font-size:1.8rem;color:{wtp_color};">${a_intercept:,.0f}</b></div>
<div><span style="opacity:0.7;">Optimal Price</span><br>
<b style="font-size:1.3rem;">${(a_intercept + MC_PRODUCTION)/2:,.0f}</b><br>
<span style="font-size:0.75rem;opacity:0.6;">({a_intercept:.0f}+{MC_PRODUCTION})/2</span></div>
<div><span style="opacity:0.7;">R²</span><br>
<b style="font-size:1.3rem;">{r_squared:.3f}</b></div>
<div><span style="opacity:0.7;">Demand Slope</span><br>
<b style="font-size:1.3rem;">{b_slope:.1f}</b><br>
<span style="font-size:0.75rem;opacity:0.6;">units/day per $1</span></div>
</div>
</div>
""", unsafe_allow_html=True)
                st.caption(f"Update the **{wtp_disc_product} Max WTP** parameter above to ${a_intercept:,.0f} to use this estimate across all sections.")
            else:
                st.info("Enter at least 2 data points with positive demand to fit the demand curve.")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 1: PRICING OPTIMIZER
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("1. Retail Pricing Optimizer")
    st.caption("Monopoly pricing with uniform WTP distribution [0, MaxWTP] — drag the price sliders to simulate")

    # Optimal reference prices (separate per product)
    pr_h_theoretical = (H_MAX_WTP + MC_PRODUCTION) / 2
    pr_s_theoretical = (S_MAX_WTP + MC_PRODUCTION) / 2

    pr_h_col, pr_s_col = st.columns(2)
    with pr_h_col:
        st.markdown(f"**Hormone** (Max WTP: ${H_MAX_WTP:.0f})")
        pr_h_price = st.slider("Hormone Price ($)", int(MC_PRODUCTION), int(H_MAX_WTP),
                                int(pr_h_theoretical), step=5, key="pr_h_price")
    with pr_s_col:
        st.markdown(f"**Specialty** (Max WTP: ${S_MAX_WTP:.0f})")
        pr_s_price = st.slider("Specialty Price ($)", int(MC_PRODUCTION), int(S_MAX_WTP),
                                int(pr_s_theoretical), step=5, key="pr_s_price")

    # Compute for both products with separate max WTP
    pr_h_arrival = ARRIVAL_RATE * HORMONE_MARKET
    pr_s_arrival = ARRIVAL_RATE * SPECIALTY_MARKET
    pr_h_demand = pr_h_arrival * (H_MAX_WTP - pr_h_price) / H_MAX_WTP if pr_h_price < H_MAX_WTP else 0
    pr_s_demand = pr_s_arrival * (S_MAX_WTP - pr_s_price) / S_MAX_WTP if pr_s_price < S_MAX_WTP else 0
    pr_h_profit = pr_h_demand * (pr_h_price - MC_PRODUCTION)
    pr_s_profit = pr_s_demand * (pr_s_price - MC_PRODUCTION)
    pr_h_revenue = pr_h_demand * pr_h_price
    pr_s_revenue = pr_s_demand * pr_s_price

    # Optimal reference (each product has its own)
    pr_h_demand_opt = pr_h_arrival * (H_MAX_WTP - pr_h_theoretical) / H_MAX_WTP if H_MAX_WTP > pr_h_theoretical else 0
    pr_s_demand_opt = pr_s_arrival * (S_MAX_WTP - pr_s_theoretical) / S_MAX_WTP if S_MAX_WTP > pr_s_theoretical else 0
    pr_h_profit_opt = pr_h_demand_opt * (pr_h_theoretical - MC_PRODUCTION)
    pr_s_profit_opt = pr_s_demand_opt * (pr_s_theoretical - MC_PRODUCTION)

    # Metrics row
    pm1, pm2, pm3, pm4 = st.columns(4)
    pm1.metric("Hormone Demand", f"{pr_h_demand:.1f}/day",
               delta=f"{(pr_h_profit/pr_h_profit_opt - 1)*100:+.1f}% vs optimal" if pr_h_profit_opt > 0 else None)
    pm2.metric("Hormone Profit", f"${pr_h_profit:,.0f}/day")
    pm3.metric("Specialty Demand", f"{pr_s_demand:.1f}/day",
               delta=f"{(pr_s_profit/pr_s_profit_opt - 1)*100:+.1f}% vs optimal" if pr_s_profit_opt > 0 else None)
    pm4.metric("Specialty Profit", f"${pr_s_profit:,.0f}/day")

    st.metric("Combined Daily Profit", f"${pr_h_profit + pr_s_profit:,.0f}/day")

    # Side-by-side profit curves with your chosen price marked
    fig_h_col, fig_s_col = st.columns(2)
    with fig_h_col:
        h_price_range = np.arange(MC_PRODUCTION, H_MAX_WTP, max(1, (H_MAX_WTP - MC_PRODUCTION) // 80))
        h_profit_arr = pr_h_arrival * (H_MAX_WTP - h_price_range) / H_MAX_WTP * (h_price_range - MC_PRODUCTION)
        fig_h = go.Figure()
        fig_h.add_trace(go.Scatter(x=h_price_range, y=h_profit_arr,
                                    name="Profit Curve", line=dict(color="#800000", width=3)))
        fig_h.add_vline(x=pr_h_theoretical, line_dash="dot", line_color="gray",
                         annotation_text=f"Optimal: ${pr_h_theoretical:,.0f}",
                         annotation_position="top left", annotation_font_size=10)
        fig_h.add_vline(x=pr_h_price, line_dash="dash", line_color="#ffd700",
                         annotation_text=f"Your price: ${pr_h_price}",
                         annotation_position="top right", annotation_font_size=10)
        fig_h.add_trace(go.Scatter(x=[pr_h_price], y=[pr_h_profit], mode="markers",
                                    marker=dict(size=12, color="#ffd700", symbol="star"),
                                    name=f"${pr_h_price} → ${pr_h_profit:,.0f}/day", showlegend=True))
        fig_h.update_layout(height=300, title=f"Hormone (WTP: ${H_MAX_WTP:.0f})", xaxis_title="Price ($)",
                             yaxis_title="Daily Profit ($)", yaxis_tickformat="$,.0f",
                             margin=dict(l=0, r=0, t=40, b=0), showlegend=True)
        st.plotly_chart(fig_h, use_container_width=True)

    with fig_s_col:
        s_price_range = np.arange(MC_PRODUCTION, S_MAX_WTP, max(1, (S_MAX_WTP - MC_PRODUCTION) // 80))
        s_profit_arr = pr_s_arrival * (S_MAX_WTP - s_price_range) / S_MAX_WTP * (s_price_range - MC_PRODUCTION)
        fig_s = go.Figure()
        fig_s.add_trace(go.Scatter(x=s_price_range, y=s_profit_arr,
                                    name="Profit Curve", line=dict(color="#1a3c5e", width=3)))
        fig_s.add_vline(x=pr_s_theoretical, line_dash="dot", line_color="gray",
                         annotation_text=f"Optimal: ${pr_s_theoretical:,.0f}",
                         annotation_position="top left", annotation_font_size=10)
        fig_s.add_vline(x=pr_s_price, line_dash="dash", line_color="#ffd700",
                         annotation_text=f"Your price: ${pr_s_price}",
                         annotation_position="top right", annotation_font_size=10)
        fig_s.add_trace(go.Scatter(x=[pr_s_price], y=[pr_s_profit], mode="markers",
                                    marker=dict(size=12, color="#ffd700", symbol="star"),
                                    name=f"${pr_s_price} → ${pr_s_profit:,.0f}/day", showlegend=True))
        fig_s.update_layout(height=300, title=f"Specialty (WTP: ${S_MAX_WTP:.0f})", xaxis_title="Price ($)",
                             yaxis_title="Daily Profit ($)", yaxis_tickformat="$,.0f",
                             margin=dict(l=0, r=0, t=40, b=0), showlegend=True)
        st.plotly_chart(fig_s, use_container_width=True)

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2: TRADE DEAL EVALUATOR (Double Marginalization)
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("2. Trade Deal Evaluator")
    st.caption("Evaluate wholesale deals — watch for double marginalization!")

    td_col1, td_col2, td_col3 = st.columns(3)
    with td_col1:
        st.markdown("**Deal Parameters**")
        td_product = st.radio("Product", ["Hormone", "Specialty"], horizontal=True, key="td_prod")
        td_max_wtp = H_MAX_WTP if td_product == "Hormone" else S_MAX_WTP
        td_wholesale = st.number_input("Wholesale Price ($/unit)", value=300, step=25, key="td_wholesale")
        td_ship_mode = st.selectbox("Shipping Mode", list(SHIPPING.keys()), index=1, key="td_ship")
        td_ship_cost = SHIPPING[td_ship_mode]["cost_per_unit"]
        td_ship_days = SHIPPING[td_ship_mode]["days"]
        td_buyer_market = st.number_input("Buyer's Market Size", value=HORMONE_MARKET, step=10000, key="td_buyermkt")
        td_buyer_arrival = ARRIVAL_RATE * td_buyer_market

    # Seller's total cost to deliver
    seller_total_cost = MC_PRODUCTION + td_ship_cost
    # Buyer's effective cost = wholesale price
    buyer_mc = td_wholesale
    # Buyer optimizes their retail price given their cost
    buyer_optimal_retail = (td_max_wtp + buyer_mc) / 2
    buyer_demand = td_buyer_arrival * (td_max_wtp - buyer_optimal_retail) / td_max_wtp
    buyer_daily_profit = buyer_demand * (buyer_optimal_retail - buyer_mc)

    # Supply chain optimal (if vertically integrated)
    sc_optimal_retail = (td_max_wtp + seller_total_cost) / 2
    sc_demand = td_buyer_arrival * (td_max_wtp - sc_optimal_retail) / td_max_wtp
    sc_daily_profit = sc_demand * (sc_optimal_retail - seller_total_cost)

    # Actual supply chain profit under double marginalization
    actual_sc_profit = buyer_demand * (buyer_optimal_retail - seller_total_cost)

    # Seller's profit
    seller_daily_profit = buyer_demand * (td_wholesale - MC_PRODUCTION - td_ship_cost)

    # Deadweight loss
    dw_loss = sc_daily_profit - actual_sc_profit

    with td_col2:
        st.markdown("**Buyer's Perspective**")
        st.metric("Buyer's Optimal Retail", f"${buyer_optimal_retail:,.0f}")
        st.metric("Buyer's Daily Demand", f"{buyer_demand:,.1f} units")
        st.metric("Buyer's Daily Profit", f"${buyer_daily_profit:,.0f}")
        st.metric("Buyer's Margin/Unit", f"${buyer_optimal_retail - buyer_mc:,.0f}")

    with td_col3:
        st.markdown("**Seller's Perspective (You)**")
        st.metric("Seller's Margin/Unit", f"${td_wholesale - MC_PRODUCTION - td_ship_cost:,.0f}")
        st.metric("Seller's Daily Profit", f"${seller_daily_profit:,.0f}")
        st.metric("Shipping Cost", f"${td_ship_cost}/unit, {td_ship_days} days")

    # Double marginalization analysis
    dm_col1, dm_col2 = st.columns(2)
    with dm_col1:
        pct_captured = (actual_sc_profit / sc_daily_profit * 100) if sc_daily_profit > 0 else 0
        loss_color = "#2d6a2e" if pct_captured > 85 else ("#b8860b" if pct_captured > 70 else "#b22222")
        st.markdown(f"""
<div style="border:2px solid {loss_color};border-radius:10px;padding:1rem;">
<h4 style="margin:0;color:{loss_color};">Double Marginalization Analysis</h4>
<table style="width:100%;margin-top:0.5rem;">
<tr><td>Optimal SC Profit (integrated)</td><td style="text-align:right;font-weight:700;">${sc_daily_profit:,.0f}/day</td></tr>
<tr><td>Actual SC Profit (with wholesale)</td><td style="text-align:right;font-weight:700;">${actual_sc_profit:,.0f}/day</td></tr>
<tr><td>Deadweight Loss</td><td style="text-align:right;color:#b22222;font-weight:700;">-${dw_loss:,.0f}/day ({100-pct_captured:.1f}%)</td></tr>
<tr><td>SC Profit Captured</td><td style="text-align:right;color:{loss_color};font-weight:700;">{pct_captured:.1f}%</td></tr>
<tr><td>Seller's Share</td><td style="text-align:right;">${seller_daily_profit:,.0f} ({seller_daily_profit/actual_sc_profit*100:.0f}%)</td></tr>
<tr><td>Buyer's Share</td><td style="text-align:right;">${buyer_daily_profit:,.0f} ({buyer_daily_profit/actual_sc_profit*100:.0f}%)</td></tr>
</table>
</div>
""", unsafe_allow_html=True)

    with dm_col2:
        # Wholesale price sweep
        ws_range = np.arange(seller_total_cost + 10, td_max_wtp * 0.6, 10)
        seller_profits = []
        buyer_profits = []
        sc_profits = []
        for w in ws_range:
            b_retail = (td_max_wtp + w) / 2
            b_demand = td_buyer_arrival * (td_max_wtp - b_retail) / td_max_wtp
            seller_profits.append(b_demand * (w - MC_PRODUCTION - td_ship_cost))
            buyer_profits.append(b_demand * (b_retail - w))
            sc_profits.append(b_demand * (b_retail - seller_total_cost))

        fig_dm = go.Figure()
        fig_dm.add_trace(go.Scatter(x=ws_range, y=sc_profits,
                                     name="Total SC Profit", line=dict(color="#333", width=2, dash="dash")))
        fig_dm.add_trace(go.Scatter(x=ws_range, y=seller_profits,
                                     name="Seller Profit", line=dict(color="#800000", width=2)))
        fig_dm.add_trace(go.Scatter(x=ws_range, y=buyer_profits,
                                     name="Buyer Profit", line=dict(color="#1a3c5e", width=2)))
        fig_dm.add_hline(y=sc_daily_profit, line_dash="dot", line_color="green",
                          annotation_text=f"Integrated optimum: ${sc_daily_profit:,.0f}")
        fig_dm.add_vline(x=td_wholesale, line_dash="dash", line_color="orange",
                          annotation_text=f"Your W=${td_wholesale}")
        fig_dm.update_layout(height=350, xaxis_title="Wholesale Price ($)",
                              yaxis_title="Daily Profit ($)", yaxis_tickformat="$,.0f",
                              margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_dm, use_container_width=True)

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # SHARED GAME STATE (feeds into sections 3, 4, 5)
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("3. Live Game Dashboard")
    st.caption("Set your current game state — all sections below update automatically")

    # ── Shared inputs in a prominent panel ────────────────────────────────────
    _h_opt_default = int((H_MAX_WTP + MC_PRODUCTION) / 2)
    _s_opt_default = int((S_MAX_WTP + MC_PRODUCTION) / 2)
    gs_col1, gs_col2, gs_col3, gs_col4 = st.columns(4)
    with gs_col1:
        gs_hormone_price = st.number_input("Hormone Retail Price ($)", value=_h_opt_default, step=25, key="gs_p1")
    with gs_col2:
        gs_specialty_price = st.number_input("Specialty Retail Price ($)", value=_s_opt_default, step=25, key="gs_p2")
    with gs_col3:
        gs_ws_price = st.number_input("Wholesale Price ($/unit)", value=int(MC_PRODUCTION * 2), step=25, key="gs_ws_price")
        gs_ws_units = st.number_input("Wholesale units/day", value=0, step=5, key="gs_ws_units")
    with gs_col4:
        gs_ws_ship = st.selectbox("Wholesale Shipping", list(SHIPPING.keys()), index=1, key="gs_ws_ship")
        gs_days_left = st.number_input("Game Days Remaining", value=30, step=1, key="gs_days")

    gs_ship_cost = SHIPPING[gs_ws_ship]["cost_per_unit"]

    # ── Derived values (shared across all sections) ──────────────────────────
    p1_market = HORMONE_MARKET
    p2_market = SPECIALTY_MARKET
    p1_demand = ARRIVAL_RATE * p1_market * (H_MAX_WTP - gs_hormone_price) / H_MAX_WTP if gs_hormone_price < H_MAX_WTP else 0
    p2_demand = ARRIVAL_RATE * p2_market * (S_MAX_WTP - gs_specialty_price) / S_MAX_WTP if gs_specialty_price < S_MAX_WTP else 0
    ws_margin = gs_ws_price - MC_PRODUCTION - gs_ship_cost
    ws_daily_profit = gs_ws_units * ws_margin
    total_demand = p1_demand + p2_demand + gs_ws_units
    both_running = p1_demand > 0 and p2_demand > 0

    p1_profit = p1_demand * (gs_hormone_price - MC_PRODUCTION)
    p2_profit = p2_demand * (gs_specialty_price - MC_PRODUCTION)
    total_daily_profit = p1_profit + p2_profit + ws_daily_profit

    utilization = total_demand / CAPACITY_PER_DAY * 100
    util_color = "#2d6a2e" if utilization < 80 else ("#b8860b" if utilization < 100 else "#b22222")

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 3A: CAPACITY GAUGE + BREAKDOWN
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("#### Capacity & Production")

    cap_col1, cap_col2 = st.columns([3, 2])
    with cap_col1:
        # Stacked bar showing demand breakdown vs capacity
        fig_cap = go.Figure()
        fig_cap.add_trace(go.Bar(
            x=["Demand"], y=[p1_demand], name=f"Hormone @ ${gs_hormone_price}",
            marker_color="#800000", text=[f"{p1_demand:.1f}"], textposition="inside",
        ))
        fig_cap.add_trace(go.Bar(
            x=["Demand"], y=[p2_demand], name=f"Specialty @ ${gs_specialty_price}",
            marker_color="#1a3c5e", text=[f"{p2_demand:.1f}"], textposition="inside",
        ))
        if gs_ws_units > 0:
            fig_cap.add_trace(go.Bar(
                x=["Demand"], y=[gs_ws_units], name=f"Wholesale @ ${gs_ws_price}",
                marker_color="#b8860b", text=[f"{gs_ws_units:.0f}"], textposition="inside",
            ))
        fig_cap.add_hline(y=CAPACITY_PER_DAY, line_dash="dash", line_color="red",
                           annotation_text=f"Capacity: {CAPACITY_PER_DAY:.0f}/day",
                           annotation_position="top right")
        fig_cap.update_layout(barmode="stack", height=300, yaxis_title="Units/day",
                               margin=dict(l=0, r=0, t=30, b=0), showlegend=True,
                               legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig_cap, use_container_width=True)

    with cap_col2:
        # Profit breakdown table
        cap_rows = [
            {"Channel": "Hormone (retail)", "Demand/day": f"{p1_demand:.1f}",
             "Price": f"${gs_hormone_price}", "Margin/unit": f"${gs_hormone_price - MC_PRODUCTION}",
             "Daily Profit": f"${p1_profit:,.0f}"},
            {"Channel": "Specialty (retail)", "Demand/day": f"{p2_demand:.1f}",
             "Price": f"${gs_specialty_price}", "Margin/unit": f"${gs_specialty_price - MC_PRODUCTION}",
             "Daily Profit": f"${p2_profit:,.0f}"},
            {"Channel": "Wholesale", "Demand/day": f"{gs_ws_units:.0f}",
             "Price": f"${gs_ws_price}", "Margin/unit": f"${ws_margin:,.0f}",
             "Daily Profit": f"${ws_daily_profit:,.0f}"},
        ]
        st.dataframe(pd.DataFrame(cap_rows), use_container_width=True, hide_index=True)

        # Summary metrics
        sm1, sm2 = st.columns(2)
        sm1.metric("Total Daily Profit", f"${total_daily_profit:,.0f}")
        sm2.metric("Utilization", f"{utilization:.0f}%")

        if utilization > 100:
            st.error(f"BOTTLENECK: Demand ({total_demand:.1f}/day) > Capacity ({CAPACITY_PER_DAY:.0f}/day). Raise prices or cut wholesale.")
        elif utilization > 80:
            st.warning(f"Approaching capacity — {CAPACITY_PER_DAY - total_demand:.1f} units/day slack remaining.")
        else:
            st.success(f"{CAPACITY_PER_DAY - total_demand:.1f} units/day excess — room for {CAPACITY_PER_DAY - total_demand:.0f} wholesale units.")

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 3B: INVENTORY (uses shared prices)
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("#### Inventory & Reorder Points")
    st.caption("Safety stock formula: z × √(Np(1-p)L) where N=arrivals/day, p=P(buy), L=lead time")

    inv_product = st.radio("Product", ["Hormone", "Specialty"], horizontal=True, key="inv_prod")
    inv_price = gs_hormone_price if inv_product == "Hormone" else gs_specialty_price
    inv_market = p1_market if inv_product == "Hormone" else p2_market
    inv_daily_demand = p1_demand if inv_product == "Hormone" else p2_demand

    inv_default_lead = BASE_LEAD_TIME * 2 - 1 if both_running else BASE_LEAD_TIME
    inv_max_wtp = H_MAX_WTP if inv_product == "Hormone" else S_MAX_WTP
    inv_N = ARRIVAL_RATE * inv_market  # arrivals per day
    inv_p = (inv_max_wtp - inv_price) / inv_max_wtp if inv_price < inv_max_wtp else 0

    inv_col1, inv_col2, inv_col3 = st.columns(3)
    with inv_col1:
        st.markdown(f"**{inv_product}** at **${inv_price}**")
        inv_lead = st.number_input("Lead Time (days)", value=float(inv_default_lead),
                                    step=0.5, format="%.1f", key="inv_lead",
                                    help=f"Auto: {inv_default_lead:.1f} days ({'both products' if both_running else 'single product'}). "
                                         f"= production ({PRODUCTION_DAYS}d) + transit. Adjust for actual observed lead time.")
        inv_service = st.selectbox("Service Level", [90, 95, 97.5, 99], index=1,
                                    format_func=lambda x: f"{x}%", key="inv_svc")
        inv_z = {90: 1.28, 95: 1.65, 97.5: 1.96, 99: 2.33}[inv_service]

        # Proper safety stock: z × √(Np(1-p)L)
        inv_demand_lt = inv_daily_demand * inv_lead
        inv_std_lt = (inv_N * inv_p * (1 - inv_p) * inv_lead) ** 0.5
        inv_safety = inv_z * inv_std_lt
        inv_reorder = inv_demand_lt + inv_safety

        st.metric("Daily Demand (Np)", f"{inv_daily_demand:.1f} units")
        st.metric("Safety Stock (z×σ)", f"{inv_safety:.0f} units")
        st.metric("Reorder Point", f"{inv_reorder:.0f} units")
        st.caption(f"z={inv_z}, σ_LT={inv_std_lt:.1f}, P(buy)={inv_p:.1%}")

    with inv_col2:
        st.markdown("**Current Inventory**")
        inv_on_hand = st.number_input("On-Hand", value=100, step=10, key="inv_oh")
        inv_in_transit = st.number_input("In-Transit", value=0, step=10, key="inv_it")
        inv_in_process = st.number_input("In-Process", value=100, step=10, key="inv_ip")

    with inv_col3:
        total_pipeline = inv_on_hand + inv_in_transit + inv_in_process
        days_cover = total_pipeline / inv_daily_demand if inv_daily_demand > 0 else float('inf')
        st.markdown("**Status**")
        st.metric("Total Pipeline", f"{total_pipeline} units")
        st.metric("Days of Coverage", f"{days_cover:.1f} days")
        st.metric("Demand During Lead Time", f"{inv_demand_lt:.0f} units")
        if total_pipeline < inv_reorder:
            st.error(f"BELOW REORDER POINT: {total_pipeline} < {inv_reorder:.0f}. Order now!")
        elif days_cover < inv_lead:
            st.error(f"STOCKOUT RISK: {days_cover:.1f} days coverage vs {inv_lead} day lead time!")
        elif days_cover < inv_lead * 1.5:
            st.warning(f"Thin buffer — {days_cover:.1f} days coverage")
        else:
            st.success(f"Healthy: {days_cover:.1f} days coverage")

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 3C: END-GAME (uses shared prices and days)
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("#### End-Game Wind-Down")

    eg_col1, eg_col2 = st.columns(2)
    with eg_col1:
        eg_total_inv = st.number_input("Total Inventory (all products)", value=200, step=50, key="eg_inv")
        eg_product = st.radio("Analyze for", ["Hormone", "Specialty"], horizontal=True, key="eg_prod")

    eg_price = gs_hormone_price if eg_product == "Hormone" else gs_specialty_price
    eg_mkt = p1_market if eg_product == "Hormone" else p2_market
    eg_max_wtp = H_MAX_WTP if eg_product == "Hormone" else S_MAX_WTP
    eg_demand = ARRIVAL_RATE * eg_mkt * (eg_max_wtp - eg_price) / eg_max_wtp if eg_price < eg_max_wtp else 0
    eg_units_sellable = eg_demand * gs_days_left
    eg_surplus = eg_total_inv - eg_units_sellable

    with eg_col2:
        st.metric("Daily Demand at Current Price", f"{eg_demand:.1f} units ({eg_product} @ ${eg_price})")
        st.metric("Units Sellable in {0} Days".format(gs_days_left), f"{eg_units_sellable:.0f}")
        if eg_surplus > 0:
            st.metric("Surplus (will be wasted)", f"{eg_surplus:.0f} units",
                       delta=f"-${eg_surplus * MC_PRODUCTION:,.0f} wasted", delta_color="inverse")
            needed_prob = eg_total_inv / (ARRIVAL_RATE * eg_mkt * gs_days_left) if gs_days_left > 0 else 1
            fire_sale = max(MC_PRODUCTION, eg_max_wtp * (1 - needed_prob))
            st.metric("Fire-Sale Price to Clear All", f"${fire_sale:,.0f}")
        else:
            st.metric("Headroom", f"{-eg_surplus:.0f} more units sellable",
                       delta="No waste", delta_color="normal")

        eg_stop_day = eg_total_inv / eg_demand if eg_demand > 0 else float('inf')
        if gs_days_left > eg_stop_day:
            st.warning(f"Stop production when {gs_days_left - eg_stop_day:.0f} days remain")
        else:
            st.info("Current inventory won't last — keep producing or cut price")

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 6: ASSIGNMENT 2 QUICK REFERENCE
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("6. Assignment 2 — Quick Reference Answers")

    with st.expander("**Part 1: Trading Game Questions**", expanded=False):
        st.markdown(f"""
**Q1 — Production Capacity:** {CAPACITY_PER_DAY:.0f} units/day (100 units / 2.5 days)

**Q2 — Excess Capacity:**
- Hormone demand at $550: {0.0001 * 300000 * (1000-550)/1000:.1f} units/day
- Specialty demand at $550: {0.0001 * 140000 * (1000-550)/1000:.1f} units/day
- Total home demand: ~19.8 units/day → **Excess: ~20.2 units/day**

**Q3 — Market size variation:** Smaller specialty markets = less bargaining power for buyers.
Larger markets = more valuable trade partnerships. Use market size data to calibrate wholesale pricing.

**Q4 — Order quantity considerations:**
- Container (1000 units, $10/unit, 21 days) vs Mail (10 units, $40/unit, 3 days)
- Match to demand rate: don't order 1000 if you sell 5/day (200 days of stock!)
- Account for game end: avoid ordering more than you can sell

**Q5 — Wholesale selling price:** ~$250-350 recommended.
At $300 wholesale (mail): seller margin = $300-$140 = $160/unit.
Buyer retails at $(1000+300)/2 = $650, demand = footfall × 35%.
Higher wholesale → more double marginalization → less total profit.

**Q6 — Wholesale buying price:** Push for $140-200.
At $200: your retail = $(1000+200)/2 = $600, margin = $400/unit.
At $300: your retail = $650, margin = $350/unit.

**Q7 — Contribution margin comparison:**
- Retail Hormone at $550: margin = **$450/unit**
- Wholesale selling at $300 (mail): margin = **$160/unit**
- Retail is ~2.8x more profitable per unit

**Q8 — Capacity allocation:** Prioritize retail (higher margin), then wholesale with excess.
Between products: choose higher-margin product if capacity-constrained.

**Q9 — Reorder point adjustment:** Raise reorder point when wholesaling — retail stockouts
lose customers forever, wholesale can wait.

**Q10 — Avoiding stockouts:** Monitor days-of-supply, set reorder > lead_time × demand + safety stock,
watch Inventory Status report, keep buffer of 50+ units.
        """)

    with st.expander("**Part 2: Supply Chain Profit (Hypothetical)**", expanded=False):
        st.markdown("""
Given: MC=$100, Shipping=$40, Max WTP=$1,000, Footfall=10/day

**Q11:** Buyer's optimal retail at W=$570: **(1000+570)/2 = $785**

**Q12:** Demand at $785: 10 × (1000-785)/1000 = **2.15 units/day**

**Q13:** Buyer's daily profit: 2.15 × ($785-$570) = **$462.25/day**

**Q14:** SC profit: 2.15 × ($785-$140) = **$1,386.75/day**

**Q15:** Seller gets $924.50 (67%), Buyer gets $462.25 (33%)

**Q16:** Demand at retail $570: 10 × 0.43 = **4.3 units/day**

**Q17:** SC profit at retail $570: 4.3 × ($570-$140) = **$1,849/day**

**Q18:** Seller gets $1,849 (100%), Buyer gets **$0** (zero margin!)

**Q19:** Ideal: charge W at cost ($140) + use franchise fees to split profit.
In practice: W=$200-300 with revenue sharing. Key insight: **double marginalization
destroys 25% of supply chain profit** when W=$570.
        """)

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 7: PRODUCTION GAME REFERENCE (for Monday)
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("7. Production Game Reference (Monday)")
    st.caption("Key parameters for the Production Game — different from Monopoly/Trading!")

    with st.expander("**Production Function (Cobb-Douglas)**", expanded=False):
        st.markdown(r"""
**Factory throughput:** Y = AK$^{\alpha}$L$^{\beta}$ (yearly), daily: $\lambda$ = AK$^{\alpha}$(l×364)$^{\beta}$ / 364

| Parameter | Value |
|---|---|
| A | 0.009 |
| α (capital exponent) | 0.10 |
| β (labor exponent) | 0.85 |
| Starting capital (K) | $100,000 |
| Starting daily labor (l) | $2,500 |
| Setup time | 0.05 days |
| Add-on capex lead time | 30 days |
| Capital depreciation | 15 years |

**Key insight:** β >> α means labor spending drives throughput far more than capital.
Increasing daily labor from $2,500 to $5,000 has a much bigger impact than doubling capital.
        """)

        st.markdown("#### Throughput Calculator")
        cd_col1, cd_col2 = st.columns(2)
        with cd_col1:
            cd_capital = st.number_input("Capital ($)", value=100000, step=10000, key="cd_k")
            cd_labor = st.number_input("Daily Labor ($/day)", value=2500, step=500, key="cd_l")
        cd_A, cd_alpha, cd_beta = 0.009, 0.10, 0.85
        yearly_labor = cd_labor * 364
        yearly_throughput = cd_A * (cd_capital ** cd_alpha) * (yearly_labor ** cd_beta)
        daily_throughput = yearly_throughput / 364
        with cd_col2:
            st.metric("Daily Throughput", f"{daily_throughput:.2f} units/day")
            st.metric("Yearly Throughput", f"{yearly_throughput:,.0f} units/year")
            st.metric("Overhead per Unit", f"${(cd_labor + cd_capital * 0.15/364) / daily_throughput:,.0f}/unit" if daily_throughput > 0 else "N/A")

    with st.expander("**DC & Shipping Costs**", expanded=False):
        st.markdown("""
**Distribution Center:**

| Parameter | Value |
|---|---|
| Capital | $2,500,000 |
| Land | $100,000 |
| Build time | 60 days |
| Daily expenditure | $2,000 |
| Handling cost | $10/unit |
| **Sales commission** | **20% of revenue** |
| Depreciation | 15 years |

**Shipping (Factory→DC and DC→DC):**

| Mode | Cost / Unit | Cost / Shipment | Transit |
|---|---|---|---|
| Mail in region | **$20/unit** | $200 / 10 units | 1 day |
| Mail between regions | **$40/unit** | $400 / 10 units | 3 days |
| Container in region | $5/unit | $5,000 / 1,000 units | 7 days |
| Container between regions | $10/unit | $10,000 / 1,000 units | 21 days |

**Mail vs Container trade-off:**
- Mail = faster (1-3 days) but 4× the per-unit cost
- Container = cheaper but 7-21 days; fixed cost whether 1 or 1,000 units sent
- A container costs $5,000 (in-region) regardless of volume → only cheap at scale (≥ 500 units)

**Key difference from Monopoly Game:** DC takes 20% revenue commission — this significantly
affects optimal pricing. Effective MC = materials + handling + 20% × price + shipping.
        """)

    with st.expander("**Financial Parameters**", expanded=False):
        st.markdown("""
| Parameter | Production Game | Monopoly Game |
|---|---|---|
| **Asset cost of capital** | **15% APR** | 10% APR |
| Emergency loan | 40% APR | 40% APR |
| Cash interest | 3% APR | 3% APR |
| Raw materials payable | **30 days** | 15 days |
| Other payables | 15 days | 30 days |
| Tax rate | 35% | 35% |
| Dividends | **6.5% APR after-tax** | N/A |
| **Return to Investors** | Cash + debt − dividends paid + after-tax return | Cash balance |

**Key differences:** Higher cost of capital (15% vs 10%) makes expansion NPV harder to justify.
Dividends option available. Return calculation is more complex.
        """)

    with st.expander("**Market Research — Specialty Markets**", expanded=False):
        st.markdown("""
**5 Specialty Market Segments:**

| Market | Core Feature | WTP Range | Market Size/Region | Bass p | Bass q | DSO |
|---|---|---|---|---|---|---|
| Clinical Fertility | Hormone (LH) | $130-300 | 40K-60K | 0.00025 | 0.004 | 10 |
| Clinical Fertility | Hormone (LH/FSH) | $230-400 | 40K-60K | 0.00025 | 0.004 | 10 |
| Law (Narcotic) | Toxicology | $1,100-1,600 | 5K-15K | 0.00025 | 0.0025 | 90 |
| MD Cancer (Breast) | Cancer (Base) | $0-900 | 10K-20K | 0.0002 | 0.0035 | 30 |
| MD Cancer (Breast) | Cancer (Breast) | $900-1,600 | 10K-20K | 0.0002 | 0.0035 | 30 |
| MD Fertility (Estrogen) | Hormone | $575-965 | 10K-20K | 0.0002 | 0.0035 | 30 |
| MD Heart (Pulse) | Heartbeat | $0-115 | 20K-40K | 0.0002 | 0.0035 | 30 |
| MD Heart (Temporal) | Heartbeat | $600-865 | 20K-40K | 0.0002 | 0.0035 | 30 |

**Deal Breakers by Market:**
- **Fertility:** No battery packs (bulky), slight wrist preference
- **Law (Narcotic):** MUST have GPS + cellular network
- **Cancer:** None
- **Heart:** GPS significantly affects perceived value (safety mechanism)

**Bass Model:** Demand = [p + q × F(t)] × [1 − F(t)] × market_size
- p = innovation (base adoption rate)
- q = imitation (word-of-mouth)
- Advertising adds incremental p per $500/day spent
        """)

    with st.expander("**Product Design Guide**", expanded=False):
        st.markdown("""
| Attribute | Feature | Design Days | Design Cost | Materials $/unit |
|---|---|---|---|---|
| **Heartbeat** | None | 3 | $1,000 | $0 |
| | Pulse only | 15 | $30,000 | $15 |
| | Temporal | 90 | $135,000 | $25 |
| **Blood vessel** | None | 3 | $1,000 | $0 |
| | Systolic only | 30 | $75,000 | $10 |
| | Systolic & diastolic | 90 | $135,000 | $15 |
| | Full profile | 120 | $180,000 | $40 |
| **Dissolved gasses** | None | 3 | $1,000 | $0 |
| | O2 only | 30 | $75,000 | $15 |
| | O2, N2, CO2 | 90 | $135,000 | $20 |
| | Full C, N, O | 90 | $135,000 | $40 |
| **Toxicology** | None | 3 | $1,000 | $0 |
| | Ethanol | 30 | $150,000 | $95 |
| | Amphetamine | 90 | $250,000 | $140 |
| | THC | 90 | $250,000 | $140 |
| | Barbiturate | 90 | $250,000 | $140 |
| | Narcotic | 90 | $250,000 | $140 |
| **Hormone** | None | 3 | $1,000 | $0 |
| | LH | 30 | $45,000 | $20 |
| | LH and FSH | 60 | $75,000 | $50 |
| | Estrogen | 60 | $75,000 | $60 |
| | Progesterone | 60 | $75,000 | $60 |
| | Testosterone | 60 | $75,000 | $50 |
| **Metabolic** | None | 3 | $1,000 | $0 |
| | Thyroxine | 90 | $90,000 | $155 |
| | Bilirubin | 90 | $90,000 | $150 |
| | Proteins | 90 | $90,000 | $170 |
| | Uric acid | 90 | $90,000 | $160 |

**Focus Groups:** $20,000, 10 participants, 7 days to complete
        """)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 0.54: 14 NEW WAR ROOM (Practice Game Market Research 8 Teams)
# Full rewrite with region-tiered markets, new detection attributes, Athlete market
# ══════════════════════════════════════════════════════════════════════════════

elif page == "✨ 14 New War Room":
    st.markdown('<p class="big-header">14 New War Room ✨</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Practice Game Market Research (01/09/2026) — supersedes D2. Region-tiered markets, expanded product design, Athlete market.</p>',
                unsafe_allow_html=True)
    st.markdown("")

    # ── WHAT'S NEW BANNER ────────────────────────────────────────────────────
    st.error("""
**⚠ THIS DOCUMENT SUPERSEDES D2**. Key differences:

1. **3-tier regional market sizes** — Serenity (small / military HUGE) | Metropolis (2-3× others) | Other Regions
2. **Military is SERENITY-only** (not Metropolis as previously assumed) — Botulinum 100-140K, Anatoxin-a 50-70K, zero elsewhere
3. **6 NEW markets**: Clinical Cardiovascular, MD Dissolved Gasses, MD Metabolic (Bilirubin), MD Cancer Bladder & Kidney, Military×2, Athlete (General + Fad)
4. **Athlete uses ADDITIVE WTP** — summed across features (unique mechanic)
5. **Product design has 3 NEW attributes**: Cancer, Neurotoxins, Motion
6. **Base features now have REAL design costs** — GPS = $45K + 30d + $50/u (was $0 in our model). Platform varies 10× (Chest $3K vs Wrists $135K)
    """)

    # ── Game Parameters ──────────────────────────────────────────────────────
    st.subheader("Game Parameters")

    n1, n2, n3, n4, n5 = st.columns(5)
    with n1:
        N14_CASH = st.number_input("Starting Cash", value=1579530, step=10000, key="n14_cash")
    with n2:
        N14_COMMISSION = st.number_input("Commission %", value=20.0, step=1.0, key="n14_comm")
    with n3:
        N14_HANDLING = st.number_input("Handling $/u", value=10, step=1, key="n14_hand")
    with n4:
        N14_SHIPPING = st.number_input("Ship Mail $/u", value=20, step=5, key="n14_ship")
    with n5:
        N14_TAX = st.number_input("Tax %", value=35.0, step=1.0, key="n14_tax")

    n14_comm_frac = N14_COMMISSION / 100

    # ── REGION SELECTOR (global for page) ────────────────────────────────────
    reg_col1, reg_col2 = st.columns([1, 3])
    with reg_col1:
        N14_REGION = st.selectbox("Region",
                                    ["Metropolis", "Other Region", "Serenity"],
                                    index=1, key="n14_region",
                                    help="Metropolis: 2-3× market sizes. Serenity: very small EXCEPT military (huge). Other: standard.")
    with reg_col2:
        if N14_REGION == "Serenity":
            st.warning("🏜️ **Serenity mode** — All medical/law/athlete markets very small (250-5000). "
                       "BUT military markets are HUGE (Botulinum 100-140K, Anatoxin-a 50-70K) — only region where military exists.")
        elif N14_REGION == "Metropolis":
            st.info("🏙️ **Metropolis mode** — All non-military markets 2-3× larger than other regions. No military.")
        else:
            st.caption("🌍 **Standard Region** — medium market sizes, no military.")

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # MASTER DATABASES
    # ══════════════════════════════════════════════════════════════════════════
    # Markets: 3-tier region sizes (mid-range values for defaults)
    N14_MARKETS = {
        "Clinical Cardiovascular": {
            "sizes": {"Serenity": 2000, "Metropolis": 40000, "Other Region": 20000},
            "wtp_tiers": [
                ("Systolic + O2 + GPS", 40, 290),
                ("Sys & Dia + O2/N2/CO2 + GPS", 85, 380),
                ("Full BP + Full DG + GPS", 350, 600),
            ],
            "core_feature": "Blood pressure + Dissolved gasses + GPS",
            "p": 0.0002, "p_adv": 0.0002, "q": 0.0035,
            "dso": 30, "dealbreaker": "Lack of GPS (significant)",
            "type": "normal",
        },
        "Clinical Fertility (LH)": {
            "sizes": {"Serenity": 2500, "Metropolis": 100000, "Other Region": 50000},
            "wtp_low": 130, "wtp_high": 300,
            "core_feature": "Hormone LH",
            "p": 0.00025, "p_adv": 0.00025, "q": 0.004,
            "dso": 10, "dealbreaker": "Bulky battery packs",
            "type": "normal",
        },
        "Clinical Fertility (LH/FSH)": {
            "sizes": {"Serenity": 2500, "Metropolis": 100000, "Other Region": 50000},
            "wtp_low": 230, "wtp_high": 400,
            "core_feature": "Hormone LH/FSH",
            "p": 0.00025, "p_adv": 0.00025, "q": 0.004,
            "dso": 10, "dealbreaker": "Bulky battery packs",
            "type": "normal",
        },
        "Law (Narcotic)": {
            "sizes": {"Serenity": 500, "Metropolis": 20000, "Other Region": 10000},
            "wtp_low": 1100, "wtp_high": 1600,
            "core_feature": "Toxicology Narcotic",
            "p": 0.00025, "p_adv": 0.00025, "q": 0.0025,
            "dso": 90, "dealbreaker": "Lack of GPS / cellular",
            "type": "normal",
        },
        "MD Cancer (Base Panel)": {
            "sizes": {"Serenity": 750, "Metropolis": 30000, "Other Region": 15000},
            "wtp_low": 0, "wtp_high": 900,
            "core_feature": "Cancer Base",
            "p": 0.0002, "p_adv": 0.0002, "q": 0.0035,
            "dso": 30, "dealbreaker": "None",
            "type": "normal",
        },
        "MD Cancer (Breast)": {
            "sizes": {"Serenity": 750, "Metropolis": 30000, "Other Region": 15000},
            "wtp_low": 900, "wtp_high": 1600,
            "core_feature": "Cancer Breast",
            "p": 0.0002, "p_adv": 0.0002, "q": 0.0035,
            "dso": 30, "dealbreaker": "None",
            "type": "normal",
        },
        "MD Cancer (Bladder & Kidney)": {
            "sizes": {"Serenity": 750, "Metropolis": 30000, "Other Region": 15000},
            "wtp_low": 900, "wtp_high": 1700,
            "core_feature": "Cancer Bladder & Kidney",
            "p": 0.0002, "p_adv": 0.0002, "q": 0.0035,
            "dso": 30, "dealbreaker": "None",
            "type": "normal",
        },
        "MD Dissolved Gasses": {
            "sizes": {"Serenity": 750, "Metropolis": 30000, "Other Region": 15000},
            "wtp_low": 350, "wtp_high": 550,
            "core_feature": "Full C, N, O",
            "p": 0.0002, "p_adv": 0.0002, "q": 0.0035,
            "dso": 30, "dealbreaker": "None",
            "type": "normal",
        },
        "MD Fertility (Estrogen)": {
            "sizes": {"Serenity": 750, "Metropolis": 30000, "Other Region": 15000},
            "wtp_low": 575, "wtp_high": 965,
            "core_feature": "Hormone Estrogen",
            "p": 0.0002, "p_adv": 0.0002, "q": 0.0035,
            "dso": 30, "dealbreaker": "None",
            "type": "normal",
        },
        "MD Heart (Pulse only)": {
            "sizes": {"Serenity": 2500, "Metropolis": 60000, "Other Region": 30000},
            "wtp_low": 0, "wtp_high": 115,
            "core_feature": "Heartbeat Pulse",
            "p": 0.0002, "p_adv": 0.0002, "q": 0.0035,
            "dso": 30, "dealbreaker": "Lack of GPS (safety)",
            "type": "normal",
        },
        "MD Heart (Temporal)": {
            "sizes": {"Serenity": 2500, "Metropolis": 60000, "Other Region": 30000},
            "wtp_low": 600, "wtp_high": 865,
            "core_feature": "Heartbeat Temporal",
            "p": 0.0002, "p_adv": 0.0002, "q": 0.0035,
            "dso": 30, "dealbreaker": "Lack of GPS (safety)",
            "type": "normal",
        },
        "MD Metabolic (Bilirubin)": {
            "sizes": {"Serenity": 750, "Metropolis": 30000, "Other Region": 15000},
            "wtp_low": 750, "wtp_high": 1300,
            "core_feature": "Metabolic Bilirubin",
            "p": 0.0002, "p_adv": 0.0002, "q": 0.0035,
            "dso": 30, "dealbreaker": "None",
            "type": "normal",
        },
        "Military Botulinum (Serenity-only)": {
            "sizes": {"Serenity": 120000, "Metropolis": 0, "Other Region": 0},
            "wtp_low": 800, "wtp_high": 1300,
            "core_feature": "Neurotoxin Botulinum",
            "p": 0.0003, "p_adv": 0.0003, "q": 0.0045,
            "dso": 60, "dealbreaker": "Lack of GPS OR polymer battery pack",
            "type": "normal",
        },
        "Military Anatoxin-a (Serenity-only)": {
            "sizes": {"Serenity": 60000, "Metropolis": 0, "Other Region": 0},
            "wtp_low": 800, "wtp_high": 1300,
            "core_feature": "Neurotoxin Anatoxin-a",
            "p": 0.0003, "p_adv": 0.0003, "q": 0.0045,
            "dso": 60, "dealbreaker": "Lack of GPS OR polymer battery pack",
            "type": "normal",
        },
        "Athlete (General)": {
            "sizes": {"Serenity": 10000, "Metropolis": 220000, "Other Region": 115000},
            "wtp_low": 0, "wtp_high": 500,   # placeholder; actual is additive per-feature
            "core_feature": "Motion / Pulse / BP / Dissolved Gas",
            "p": 0.0003, "p_adv": 0.0003, "q": 0.003,
            "dso": 5, "dealbreaker": "Bulky battery packs",
            "type": "athlete",
        },
        "Athlete (Fad)": {
            "sizes": {"Serenity": 10000, "Metropolis": 220000, "Other Region": 115000},
            "wtp_low": 0, "wtp_high": 500,
            "core_feature": "Motion + preferred finish/platform",
            "p": 0.0009, "p_adv": 0.0009, "q": 0.009,
            "dso": 5, "dealbreaker": "Wrong finish or platform (fad customers)",
            "type": "athlete_fad",
        },
    }

    # Athlete WTP is additive by feature (from Practice Game Market Research p.22)
    N14_ATHLETE_WTP = {
        "Heartbeat": {"None": 0, "Pulse only": 150, "Pulse + temporal": 150},
        "Blood vessel": {"None": 0, "Systolic only": 27, "Systolic & diastolic": 35, "Full profile": 35},
        "Dissolved gasses": {"None": 0, "O2 only": 22, "O2, N2, CO2": 27, "Full C,N,O": 27},
        "Motion": {"None": 0, "Steps": 20, "Steps + balance": 37, "Steps + balance + gait": 57},
        "Platform": {"Chest": 5, "Stockings": 20, "Sleeves": 30, "Wrists": 37},
    }

    # Product design attributes — FULL from page 23-24 of new research doc
    N14_DETECTION = {
        "Heartbeat": {
            "None": (3, 1000, 0), "Pulse only": (15, 30000, 15), "Temporal": (90, 135000, 25),
        },
        "Blood vessel": {
            "None": (3, 1000, 0), "Systolic only": (30, 75000, 10),
            "Systolic & diastolic": (90, 135000, 15), "Full profile": (120, 180000, 40),
        },
        "Dissolved gasses": {
            "None": (3, 1000, 0), "O2 only": (30, 75000, 15),
            "O2, N2, CO2": (90, 135000, 20), "Full C,N,O": (90, 135000, 40),
        },
        "Toxicology": {
            "None": (3, 1000, 0), "Ethanol": (30, 150000, 95),
            "Amphetamine": (90, 250000, 140), "THC": (90, 250000, 140),
            "Barbiturate": (90, 250000, 140), "Narcotic": (90, 250000, 140),
        },
        "Hormone": {
            "None": (3, 1000, 0), "LH": (30, 45000, 20),
            "LH and FSH": (60, 75000, 50), "Estrogen": (60, 75000, 60),
            "Progesterone": (60, 75000, 60), "Testosterone": (60, 75000, 50),
        },
        "Metabolic": {
            "None": (3, 1000, 0), "Thyroxine": (90, 90000, 155),
            "Bilirubin": (90, 90000, 150), "Proteins": (90, 90000, 170),
            "Uric acid": (90, 90000, 160),
        },
        "Cancer": {
            "None": (3, 1000, 0), "Base": (60, 200000, 100),
            "Prostate": (90, 300000, 210), "Breast": (90, 300000, 200),
            "Bladder & Kidney": (90, 300000, 300), "Lymphoma": (90, 300000, 250),
            "Blood & Bone": (90, 300000, 310),
        },
        "Neurotoxins": {
            "None": (3, 1000, 0), "Botulinum": (90, 135000, 190),
            "Anatoxin-a": (90, 135000, 210), "Sarin & Cyclosarin": (90, 135000, 220),
            "Soman": (90, 135000, 280),
        },
        "Motion": {
            "None": (3, 1000, 0), "Steps": (15, 30000, 15),
            "Steps + balance": (30, 45000, 30), "Steps + balance + gait": (45, 60000, 45),
        },
    }

    # Base features now with REAL costs from Product Design Guide (page 24)
    N14_BASE = {
        "Platform": {
            "Wrists": (90, 135000, 20), "Chest": (15, 3000, 10),
            "Sleeves": (30, 30000, 15), "Stockings": (30, 30000, 15),
        },
        "GPS": {
            "No GPS": (3, 1000, 0), "GPS": (30, 45000, 50),
        },
        "Network": {
            "Bluetooth": (15, 1000, 5), "2.4 GHz": (30, 30000, 10),
            "5 GHz": (45, 36000, 20),
        },
        "Power": {
            "Ni-Cd": (5, 1500, 5), "Ni-Cd pack": (10, 15000, 20),
            "Polymer": (5, 1500, 35), "Polymer pack": (10, 15000, 140),
        },
        "Finish": {
            "Original": (3, 2400, 0), "Blue": (5, 3000, 3), "Red": (5, 3000, 3),
            "Green": (5, 3000, 3), "Black": (5, 3000, 3), "White": (5, 3000, 3),
            "Metallic": (90, 27000, 6), "Geometric": (90, 27000, 6),
            "Camouflage": (20, 27000, 6),
        },
    }

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 1: MARKET SEGMENT ANALYZER (Region-aware, all 16 markets)
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("1. Market Segment Analyzer (Region-Aware, up to 5 markets)")
    st.caption(f"Region: **{N14_REGION}**. Market sizes auto-scaled. Athlete markets use additive WTP.")

    ms_top1, ms_top2 = st.columns([1, 3])
    with ms_top1:
        n14_n_mkts = st.number_input("# Markets", min_value=2, max_value=5,
                                        value=5, step=1, key="n14_n_mkts")
    with ms_top2:
        n14_mkt_materials = st.number_input("Your Materials Cost ($/u)",
                                               value=100, step=10, key="n14_mkt_mat")

    # Filter markets available in this region (exclude zero-size)
    available_markets = {k: v for k, v in N14_MARKETS.items()
                          if v["sizes"].get(N14_REGION, 0) > 0}

    if len(available_markets) < 2:
        st.error(f"Only {len(available_markets)} markets available in {N14_REGION}. Change region.")
    else:
        mkt_keys = list(available_markets.keys())
        max_cols = min(int(n14_n_mkts), len(mkt_keys))
        n14_mkt_cols = st.columns(max_cols)
        n14_mkt_summary = []

        # default selection - pick first 5 different markets
        for i, col in enumerate(n14_mkt_cols):
            with col:
                default_idx = i if i < len(mkt_keys) else 0
                n14_sel_mkt = st.selectbox(f"Market {i+1}", mkt_keys, index=default_idx,
                                             key=f"n14_ms_sel_{i}")
                m = available_markets[n14_sel_mkt]
                m_size = m["sizes"][N14_REGION]

                # Info card
                db_color = "#b22222" if m["dealbreaker"] != "None" else "#2d6a2e"
                st.markdown(f"""
<div style="background:rgba(26,60,94,0.15);border-left:3px solid #1a3c5e;
    border-radius:6px;padding:0.5rem 0.7rem;font-size:0.72rem;margin-bottom:0.3rem;">
<b>{n14_sel_mkt}</b><br>
Feature: {m['core_feature']}<br>
Size @ {N14_REGION}: {m_size:,}<br>
Bass p: {m['p']} q: {m['q']} | DSO: {m['dso']}d<br>
DB: <span style="color:{db_color};">{m['dealbreaker']}</span>
</div>
""", unsafe_allow_html=True)

                # Market size slider (adjustable around default)
                mkt_size_in = st.slider("Market Size",
                                          int(m_size * 0.3), int(m_size * 2.5),
                                          int(m_size), step=max(100, m_size // 50),
                                          key=f"n14_ms_size_{i}")

                # Handle Athlete markets with additive WTP
                if m.get("type") == "athlete" or m.get("type") == "athlete_fad":
                    st.markdown("**Athlete Features (additive WTP)**")
                    a_heart = st.selectbox("Heartbeat", list(N14_ATHLETE_WTP["Heartbeat"].keys()),
                                              index=1, key=f"n14_ath_hb_{i}")
                    a_bv = st.selectbox("Blood Vessel", list(N14_ATHLETE_WTP["Blood vessel"].keys()),
                                          index=0, key=f"n14_ath_bv_{i}")
                    a_dg = st.selectbox("Dissolved Gasses", list(N14_ATHLETE_WTP["Dissolved gasses"].keys()),
                                          index=0, key=f"n14_ath_dg_{i}")
                    a_mo = st.selectbox("Motion", list(N14_ATHLETE_WTP["Motion"].keys()),
                                          index=1, key=f"n14_ath_mo_{i}")
                    a_pl = st.selectbox("Platform", list(N14_ATHLETE_WTP["Platform"].keys()),
                                          index=3, key=f"n14_ath_pl_{i}")
                    additive_wtp = (N14_ATHLETE_WTP["Heartbeat"][a_heart] +
                                     N14_ATHLETE_WTP["Blood vessel"][a_bv] +
                                     N14_ATHLETE_WTP["Dissolved gasses"][a_dg] +
                                     N14_ATHLETE_WTP["Motion"][a_mo] +
                                     N14_ATHLETE_WTP["Platform"][a_pl])
                    st.metric("Summed Max WTP", f"${additive_wtp}")
                    wtp_max_use = additive_wtp
                    wtp_mean_use = additive_wtp * 0.85
                    wtp_std_use = additive_wtp * 0.1
                elif "wtp_tiers" in m:
                    # Tiered WTP: user picks which feature tier applies to their product
                    st.markdown("**Feature tier (determines WTP range)**")
                    tier_labels = [f"{t[0]} (${t[1]}-${t[2]})" for t in m["wtp_tiers"]]
                    tier_idx = st.selectbox("Your product tier", range(len(tier_labels)),
                                              format_func=lambda x: tier_labels[x],
                                              index=len(m["wtp_tiers"]) - 1,  # default to highest
                                              key=f"n14_ms_tier_{i}")
                    tier = m["wtp_tiers"][tier_idx]
                    wtp_low_d, wtp_high_d = tier[1], tier[2]
                    wtp_max_use = st.slider("Max WTP ($)",
                                               int(wtp_low_d), int(wtp_high_d * 1.2),
                                               int(wtp_high_d), step=10, key=f"n14_ms_wtp_{i}")
                    wtp_mean_use = (wtp_low_d + wtp_max_use) / 2
                    wtp_std_use = max(1, (wtp_max_use - wtp_low_d) / 3.464)
                else:
                    # Normal WTP: mid-range from Practice Game doc; treat uniform [wtp_low, wtp_high]
                    wtp_low_d = m["wtp_low"]
                    wtp_high_d = m["wtp_high"]
                    st.caption(f"WTP range: ${wtp_low_d} - ${wtp_high_d} (uniform assumption)")
                    wtp_max_use = st.slider("Max WTP ($)",
                                               int(max(wtp_low_d + 1, 1)), int(max(wtp_high_d * 1.2, wtp_low_d + 10)),
                                               int(max(wtp_high_d, wtp_low_d + 1)), step=10, key=f"n14_ms_wtp_{i}")
                    wtp_mean_use = (wtp_low_d + wtp_max_use) / 2
                    wtp_std_use = max(1, (wtp_max_use - wtp_low_d) / 3.464)  # uniform → std

                # Price slider
                ms_p_min = int(n14_mkt_materials + N14_HANDLING + N14_SHIPPING)
                ms_p_max = int(wtp_max_use * 1.1) if wtp_max_use > 0 else 1000

                # Find optimum via cached function
                opt_p = find_optimal_price_normal(
                    price_min=ms_p_min, price_max=ms_p_max,
                    mean_wtp=float(wtp_mean_use), std_wtp=float(max(1, wtp_std_use)),
                    materials=float(n14_mkt_materials), shipping=float(N14_SHIPPING),
                    handling=float(N14_HANDLING), commission_frac=float(n14_comm_frac),
                    step=5,
                )

                n14_ms_price = st.slider("Your Price ($)",
                                           ms_p_min, ms_p_max, opt_p,
                                           step=10, key=f"n14_ms_price_{i}",
                                           help=f"Default = optimum (${opt_p})")

                # P(buy) via Normal assumption
                p_buy_ms = 1 - _normal_cdf(float(n14_ms_price),
                                              float(wtp_mean_use),
                                              float(max(1, wtp_std_use)))

                # CM
                ms_comm = n14_ms_price * n14_comm_frac
                ms_cm_u = n14_ms_price - ms_comm - N14_HANDLING - n14_mkt_materials - N14_SHIPPING
                ms_cm_arr = ms_cm_u * p_buy_ms

                # Bass peak
                peak_q = mkt_size_in * ((m["p"]+m["q"])**2) / (4*m["q"]) if m["q"] > 0 else 0

                cm_c = "#2d6a2e" if ms_cm_u > 0 else "#b22222"
                st.markdown(f"""
<div style="background:rgba({'45,106,46' if ms_cm_u > 0 else '178,34,34'},0.12);
    border-left:3px solid {cm_c};padding:0.4rem 0.6rem;border-radius:5px;">
<span style="font-size:0.65rem;opacity:0.7;">At ${n14_ms_price}</span><br>
P(buy): <b>{p_buy_ms:.0%}</b> | CM/u: <b style="color:{cm_c};">${ms_cm_u:,.0f}</b><br>
CM/arr: <b style="color:{cm_c};">${ms_cm_arr:,.0f}</b> | Peak: {peak_q * p_buy_ms:,.1f}/d
</div>
""", unsafe_allow_html=True)

                n14_mkt_summary.append({
                    "Market": n14_sel_mkt,
                    "Region": N14_REGION,
                    "Size": f"{mkt_size_in:,}",
                    "WTP max": f"${wtp_max_use:,.0f}",
                    "Price": f"${n14_ms_price}",
                    "P(buy)": f"{p_buy_ms:.0%}",
                    "CM/u": f"${ms_cm_u:,.0f}",
                    "CM/arr": f"${ms_cm_arr:,.0f}",
                    "Peak/d": f"{peak_q * p_buy_ms:,.1f}",
                    "DSO": f"{m['dso']}d",
                })

        st.markdown("**Market Summary**")
        st.dataframe(pd.DataFrame(n14_mkt_summary), use_container_width=True, hide_index=True)

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2: PRODUCT DESIGN ROI with NEW attributes + REAL base feature costs
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("2. Product Design ROI Calculator (Expanded)")
    st.caption("Base features NOW have real costs. 3 NEW detection attributes: Cancer, Neurotoxins, Motion.")

    pd_top1, pd_top2 = st.columns([1, 3])
    with pd_top1:
        n14_n_prods = st.number_input("# Products", min_value=1, max_value=5,
                                        value=5, step=1, key="n14_n_prods")
    with pd_top2:
        st.caption(f"Region: **{N14_REGION}** (shipping = ${N14_SHIPPING}/u mail in-region). All costs from Practice Game Market Research page 23-24.")

    # Updated presets matching the new research doc
    N14_PRESETS = {
        "Heart View (flagship)": {
            "Platform": "Chest", "GPS": "GPS", "Network": "2.4 GHz",
            "Power": "Polymer", "Finish": "Original",
            "Heartbeat": "Temporal", "Blood vessel": "None",
            "Dissolved gasses": "None", "Toxicology": "None",
            "Hormone": "None", "Metabolic": "None",
            "Cancer": "None", "Neurotoxins": "None", "Motion": "None",
            "price": 700, "target": "MD-Heart",
        },
        "Cancer Breast": {
            "Platform": "Chest", "GPS": "GPS", "Network": "2.4 GHz",
            "Power": "Polymer", "Finish": "Original",
            "Heartbeat": "Pulse only", "Blood vessel": "None",
            "Dissolved gasses": "None", "Toxicology": "None",
            "Hormone": "None", "Metabolic": "None",
            "Cancer": "Breast", "Neurotoxins": "None", "Motion": "None",
            "price": 1250, "target": "MD Cancer (Breast)",
        },
        "Law Narcotic": {
            "Platform": "Stockings", "GPS": "GPS", "Network": "5 GHz",
            "Power": "Polymer pack", "Finish": "Black",
            "Heartbeat": "None", "Blood vessel": "None",
            "Dissolved gasses": "None", "Toxicology": "Narcotic",
            "Hormone": "None", "Metabolic": "None",
            "Cancer": "None", "Neurotoxins": "None", "Motion": "None",
            "price": 1350, "target": "Law (Narcotic)",
        },
        "Military Botulinum (Serenity)": {
            "Platform": "Sleeves", "GPS": "GPS", "Network": "2.4 GHz",
            "Power": "Polymer pack", "Finish": "Camouflage",
            "Heartbeat": "None", "Blood vessel": "None",
            "Dissolved gasses": "None", "Toxicology": "None",
            "Hormone": "None", "Metabolic": "None",
            "Cancer": "None", "Neurotoxins": "Botulinum", "Motion": "None",
            "price": 1100, "target": "Military Botulinum",
        },
        "Athlete General": {
            "Platform": "Wrists", "GPS": "No GPS", "Network": "Bluetooth",
            "Power": "Polymer", "Finish": "Blue",
            "Heartbeat": "Pulse only", "Blood vessel": "None",
            "Dissolved gasses": "None", "Toxicology": "None",
            "Hormone": "None", "Metabolic": "None",
            "Cancer": "None", "Neurotoxins": "None", "Motion": "Steps",
            "price": 250, "target": "Athlete General",
        },
    }

    preset_keys = list(N14_PRESETS.keys())
    n14_pd_cols = st.columns(int(n14_n_prods))
    n14_pd_summary = []

    for i, col in enumerate(n14_pd_cols):
        with col:
            default_idx = i if i < len(preset_keys) else 0
            p_sel = st.selectbox(f"Preset P{i+1}", preset_keys, index=default_idx,
                                   key=f"n14_pd_preset_{i}")
            preset = N14_PRESETS[p_sel]

            st.markdown("**Base Features** (now with real costs!)")
            sel_base = {}
            for attr, opts in N14_BASE.items():
                default_feat = preset.get(attr, list(opts.keys())[0])
                labeled = {f"{feat} — {d}d, ${c/1000:.1f}K, ${m}/u": feat
                            for feat, (d, c, m) in opts.items()}
                labels = list(labeled.keys())
                default_label = next((l for l, f in labeled.items() if f == default_feat), labels[0])
                idx = labels.index(default_label)
                chosen = st.selectbox(attr, labels, index=idx,
                                        key=f"n14_pd_base_{attr}_{i}")
                sel_base[attr] = labeled[chosen]

            st.markdown("**Detection Agenda** (9 attributes)")
            sel_det = {}
            for attr, opts in N14_DETECTION.items():
                default_feat = preset[attr]
                labeled = {f"{feat} — {d}d, ${c/1000:.0f}K, ${m}/u": feat
                            for feat, (d, c, m) in opts.items()}
                labels = list(labeled.keys())
                default_label = next((l for l, f in labeled.items() if f == default_feat), labels[0])
                idx = labels.index(default_label)
                chosen = st.selectbox(attr, labels, index=idx,
                                        key=f"n14_pd_det_{attr}_{i}")
                sel_det[attr] = labeled[chosen]

            # Totals (base + detection)
            base_days = max(N14_BASE[a][sel_base[a]][0] for a in N14_BASE)
            base_cost = sum(N14_BASE[a][sel_base[a]][1] for a in N14_BASE)
            base_mat = sum(N14_BASE[a][sel_base[a]][2] for a in N14_BASE)
            det_days = max(N14_DETECTION[a][sel_det[a]][0] for a in N14_DETECTION)
            det_cost = sum(N14_DETECTION[a][sel_det[a]][1] for a in N14_DETECTION)
            det_mat = sum(N14_DETECTION[a][sel_det[a]][2] for a in N14_DETECTION)

            total_days = max(base_days, det_days)
            total_cost = base_cost + det_cost
            total_mat = base_mat + det_mat

            n14_pd_price = st.number_input("Price ($)", value=preset["price"], step=25,
                                              key=f"n14_pd_price_{i}")
            n14_pd_sales = st.number_input("Sales/day", value=5, step=1,
                                              key=f"n14_pd_sales_{i}")

            n14_pd_margin = (n14_pd_price * (1 - n14_comm_frac)
                               - N14_HANDLING - total_mat - N14_SHIPPING)
            n14_pd_be = (total_cost / (n14_pd_margin * n14_pd_sales)
                           if n14_pd_margin > 0 and n14_pd_sales > 0 else float("inf"))

            st.metric("Design Days", f"{total_days}")
            st.metric("Design Cost", f"${total_cost:,}")
            st.metric("Materials/u", f"${total_mat}")
            st.metric("CM/u", f"${n14_pd_margin:,.0f}")
            if n14_pd_be < float("inf"):
                st.metric("Break-even", f"{n14_pd_be:.0f}d ({n14_pd_be/30:.1f} mo)",
                           delta_color="off")
            else:
                st.error("Negative margin")

            # Cannibalization + fit checker with new rules
            target = st.selectbox("Target Market",
                                     ["(select)", "MD-Heart", "MD Cancer (Breast)",
                                      "MD Cancer (Bladder & Kidney)", "MD Cancer (Base)",
                                      "MD-Estrogen", "Law (Narcotic)", "Military Botulinum",
                                      "Military Anatoxin-a", "Athlete General", "Athlete Fad",
                                      "Clinical Cardiovascular", "Clinical Fertility",
                                      "MD Metabolic", "MD Dissolved Gasses"],
                                     index=0, key=f"n14_pd_target_{i}")

            warnings = []
            # Heartbeat cannibalization
            if sel_det["Heartbeat"] == "Temporal" and target not in ["(select)", "MD-Heart"]:
                warnings.append("🔴 Temporal heartbeat outside MD-Heart → cannibalizes Heart View")
            # Narcotic only in Law/Military
            if sel_det["Toxicology"] in ["Narcotic", "Amphetamine", "THC", "Barbiturate"] and target not in ["(select)", "Law (Narcotic)", "Military Botulinum", "Military Anatoxin-a"]:
                warnings.append(f"🟠 {sel_det['Toxicology']} outside Law/Mil → $140/u wasted + cannibalization risk")
            # Cancer only in Cancer markets
            if sel_det["Cancer"] not in ["None"] and "Cancer" not in target:
                warnings.append(f"🔴 Cancer {sel_det['Cancer']} outside Cancer market → $200+/u wasted materials")
            # Neurotoxin only in Military
            if sel_det["Neurotoxins"] not in ["None"] and "Military" not in target:
                warnings.append(f"🔴 Neurotoxin {sel_det['Neurotoxins']} outside Military → $190+/u wasted")
            # Law Narcotic needs GPS + cellular
            if target == "Law (Narcotic)" and (sel_base["GPS"] == "No GPS" or sel_base["Network"] == "Bluetooth"):
                warnings.append("🔴 Law-Narcotic deal breaker — needs GPS + cellular")
            # Military needs GPS + polymer pack
            if "Military" in target and (sel_base["GPS"] == "No GPS" or "Polymer pack" not in sel_base["Power"]):
                warnings.append("🔴 Military deal breaker — needs GPS + polymer battery pack")
            # Fertility avoid bulky battery
            if target in ["Clinical Fertility", "MD-Estrogen"] and "pack" in sel_base["Power"]:
                warnings.append("🟠 Fertility: avoid bulky battery packs")
            # Cardiovascular needs GPS
            if target == "Clinical Cardiovascular" and sel_base["GPS"] == "No GPS":
                warnings.append("🟠 Cardiovascular: lack of GPS reduces perceived value")
            # Platform-market mismatch
            if target in ["Clinical Fertility", "MD-Estrogen"] and sel_base["Platform"] == "Chest":
                warnings.append("🟡 Fertility users prefer wrists over chest")
            # Athlete prefers wrists
            if "Athlete" in target and sel_base["Platform"] == "Chest":
                warnings.append("🟡 Athletes prefer wrists > sleeves > stockings > chest")

            if warnings:
                st.warning("**Flags:**\n\n" + "\n\n".join(warnings))
            elif target != "(select)":
                st.success("✅ No cannibalization/fit flags")

            n14_pd_summary.append({
                "Product": f"P{i+1}: {p_sel}",
                "Target": target,
                "Days": total_days,
                "Design $": f"${total_cost:,}",
                "Mat/u": f"${total_mat}",
                "Price": f"${n14_pd_price}",
                "CM/u": f"${n14_pd_margin:,.0f}",
                "Break-even": f"{n14_pd_be:.0f}d" if n14_pd_be < float("inf") else "N/A",
                "Warnings": len(warnings) if target != "(select)" else "—",
            })

    st.markdown("**Product Comparison Summary**")
    st.dataframe(pd.DataFrame(n14_pd_summary), use_container_width=True, hide_index=True)

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # Quick reference / strategic summary
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("3. Strategic Quick Reference")

    with st.expander("**Region-specific strategic angles**", expanded=True):
        st.markdown(f"""
### You are currently analyzing: **{N14_REGION}**

**Metropolis (2-3× sizes):**
- Athlete General/Fad **220K** (MASSIVE) — target with motion + finish products
- Clinical Cardiovascular **40K**, Clinical Fertility **100K**, all MD **20-60K**
- Best region for volume plays with moderate WTP products

**Serenity (niche + military):**
- **ONLY region with Military markets** (Botulinum 100-140K, Anatoxin-a 50-70K)
- All medical/law markets are TINY (250-2500) — don't bother with non-military here
- Military WTP $800-1,300 with 60-day DSO. Requires GPS + polymer pack + dark finish

**Other Regions (balanced):**
- MD Heart 30K, Cancer 15K, Law 10K, Fertility 50K, Athlete 115K
- Balanced for 2-3 product portfolios
        """)

    with st.expander("**Design cost traps — base features add up fast**", expanded=False):
        st.markdown("""
### Old vs New assumptions

| Feature | Old $ (free) | New Real $ | Δ |
|---|---|---|---|
| GPS | $0 / 0d | **$45K / 30d / $50/u** | MAJOR |
| 5 GHz cellular | $0 / 0d | **$36K / 45d / $20/u** | |
| Wrists platform | $0 / 0d | **$135K / 90d / $20/u** | |
| Polymer pack | $0 / 0d | **$15K / 10d / $140/u** | |
| Metallic finish | $0 / 0d | **$27K / 90d / $6/u** | |

**Implication:** Any product with GPS + 5GHz + wrists + polymer pack + metallic
adds **$258K design cost, 90+ days, $236/u materials** before you even pick
a detection attribute.

**Recommendation:** Start with Chest + No GPS + Bluetooth + Ni-Cd + Original
(total $9K design + minimal materials) for low-WTP markets. Only add premium
base features when the market requires them (Law: GPS+cellular; Military: polymer pack+GPS).
        """)

    with st.expander("**Market x product fit rules of thumb**", expanded=False):
        st.markdown("""
| Target Market | Required Features | Avoid |
|---|---|---|
| **MD-Heart (Temporal)** | Temporal heartbeat + GPS (safety) | — |
| **MD Cancer (specific)** | Matching Cancer attribute | Wrong cancer type (wasted $) |
| **Law Narcotic** | Toxicology=Narcotic + GPS + cellular | Bluetooth |
| **Military Botulinum/Anatoxin-a** | Matching Neurotoxin + GPS + polymer pack + dark finish | Short battery |
| **Clinical Cardiovascular** | Full BP + Full DG + GPS | No GPS |
| **Athlete General/Fad** | Motion + Pulse + Wrists + Finish | Bulky battery, chest platform |
| **Fertility (LH/FSH or Estrogen)** | Matching Hormone + wrist preference | Bulky battery packs |
| **Dissolved Gasses MD** | Full C, N, O | — |
| **Metabolic (Bilirubin)** | Bilirubin metabolic | — |
        """)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 0.55: 14 TRIAL WAR ROOM (Day 3 — Bass Model deep dive + Debt + Scenarios)
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🚀 14 Trial War Room":
    st.markdown('<p class="big-header">14 Trial War Room</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Day 3 Practice Game — April 14 | Normal-WTP Bass Model, Advertising Strategy, Debt Capacity, Scenario Analysis</p>',
                unsafe_allow_html=True)
    st.markdown("")

    # ══════════════════════════════════════════════════════════════════════════
    # WHAT'S NEW + 8-SECTION SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    w14_intro_col1, w14_intro_col2 = st.columns([1, 1])

    with w14_intro_col1:
        st.info("""
**🆕 What's new today (from D3 materials):**

1. **WTP is NORMALLY distributed** — mean + std dev, not uniform. Focus groups reveal mean/max.
2. **Three arrival streams:** Innovators (decay over time) + Imitators (grow over time) + **Advertising-attracted** (same-day)
3. **Advertising decision framework** — when to advertise, when not to, strategic use
4. **Debt issuance is tranche-based** — exhaust Excellent → Good → Poor sequentially
5. **Scenario comparison** — 4-year cumulative contribution under price × advertising combinations

**Today's game:** Practice Game 7-9pm. **Tomorrow (Wed):** Competition begins. Today is last practice.
        """)

    with w14_intro_col2:
        st.success("""
**📋 The 10 Components of this War Room:**

1. **Advanced Bass Model** — Normal WTP, 3 arrival streams, 4-year daily simulation
2. **Scenario Comparison** — 4 price × ad scenarios with trajectory plots
3. **Advertising Decision Framework** — when/when-not checklist + ROI calculator
4. **Debt Capacity & Bond Issuance** — tranche-based (Excellent → Good → Poor)
5. **Normal vs Uniform WTP** — side-by-side comparison tool
6. **Cobb-Douglas + Little's Law + CM Table** — 4 factory types side-by-side
7. **Market Segment Analyzer** (up to 5 markets, **Metropolis toggle** for +4 military markets)
8. **Product Design ROI** (up to 5 products, **cannibalization checker** built in)
9. **🆕 Supply Chain Trade-Offs** — Mail vs Container, Own DC vs Wholesale, New Factory vs Capex
10. **🆕 Cash & Tax Discipline Planner** — quarterly tax projection, cash buffer, waterfall
        """)

    # ══════════════════════════════════════════════════════════════════════════
    # GAME PARAMETERS (shared with 13 War Room)
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("Game Parameters")
    st.caption("Production Game (oligopoly, 8 teams, 4-year horizon starting day 365)")

    r1c1, r1c2, r1c3, r1c4, r1c5 = st.columns(5)
    with r1c1:
        W14_STARTING_CASH = st.number_input("Starting Cash ($)", value=1579530, step=10000, key="w14_cash")
    with r1c2:
        W14_COMMISSION = st.number_input("Sales Commission (%)", value=20.0, step=1.0, key="w14_comm")
    with r1c3:
        W14_HANDLING = st.number_input("Handling ($/unit)", value=10, step=1, key="w14_handling")
    with r1c4:
        W14_SHIPPING = st.number_input("Shipping Mail in-region ($/u)", value=20, step=5, key="w14_ship")
    with r1c5:
        W14_TAX = st.number_input("Tax Rate (%)", value=35.0, step=1.0, key="w14_tax")

    w14_comm_frac = W14_COMMISSION / 100
    w14_tax_frac = W14_TAX / 100

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 1: ADVANCED BASS MODEL with NORMAL WTP + ADVERTISING
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("1. Advanced Bass Model — Normal WTP + Advertising")
    st.caption("Three arrival streams: Innovators (p) + Imitators (q, from cumulative adopters) + Advertising-attracted (same-day)")

    bass_col1, bass_col2 = st.columns([1, 2])
    with bass_col1:
        st.markdown("**Market Parameters** (from focus group)")
        b14_mean = st.number_input("Mean WTP ($)", value=1300, step=50, key="b14_mean",
                                     help="Center of the normal WTP distribution")
        b14_std = st.number_input("Std Dev WTP ($)", value=130, step=10, key="b14_std",
                                    help="Spread of WTP. Typically mean/10.")
        b14_M = st.number_input("Market Size (M)", value=15000, step=1000, key="b14_M")
        b14_p = st.number_input("Innovation coef (p)", value=0.0002, step=0.00005,
                                  format="%.5f", key="b14_p")
        b14_q = st.number_input("Imitation coef (q)", value=0.0035, step=0.0005,
                                  format="%.4f", key="b14_q")

        st.markdown("**Pricing & Costs**")
        b14_price = st.number_input("Retail Price ($)", value=900, step=25, key="b14_price")
        b14_materials = st.number_input("Materials ($/u)", value=375, step=10, key="b14_mat")
        b14_mfg_oh = st.number_input("Mfg Overhead ($/u)", value=80, step=10, key="b14_oh")

        st.markdown("**Advertising**")
        b14_ad_daily = st.number_input("Ad Spend ($/day)", value=0, step=500, key="b14_ad")
        b14_ad_duration = st.number_input("Ad Duration (days)", value=364, step=30, key="b14_ad_days")
        b14_p_ad_per_500 = st.number_input("Incremental p per $500 ad/day",
                                              value=0.0002, step=0.00005, format="%.5f",
                                              key="b14_p_ad")
        b14_sim_days = st.number_input("Simulate Days", value=1460, step=30, key="b14_sim")

    # Simulate the Bass model with three arrival types (CACHED — instant on slider repeat)
    _bass_result = simulate_bass_normal(
        M=int(b14_M), p=float(b14_p), q=float(b14_q),
        p_ad_per_500=float(b14_p_ad_per_500),
        ad_daily=float(b14_ad_daily), ad_duration=int(b14_ad_duration),
        price=float(b14_price), mean_wtp=float(b14_mean), std_wtp=float(b14_std),
        sim_days=int(b14_sim_days),
    )
    p_buy = _bass_result["p_buy"]
    days = _bass_result["days"]
    innovators_list = _bass_result["innovators"]
    imitators_list = _bass_result["imitators"]
    advertising_list = _bass_result["advertising"]
    total_arrivals = _bass_result["total_arrivals"]
    purchases_list = _bass_result["purchases"]
    cumulative_purchases = _bass_result["cumulative"]
    # Kept for local helper below (still used in inline helpers in other sections)
    def normal_cdf(x, mu, sigma):
        return _normal_cdf(float(x), float(mu), float(sigma))

    with bass_col2:
        # Plot 3 arrival streams over time
        fig_arrivals = go.Figure()
        fig_arrivals.add_trace(go.Scatter(x=days, y=innovators_list, name="Innovators (p)",
                                            line=dict(color="#1a3c5e", width=2)))
        fig_arrivals.add_trace(go.Scatter(x=days, y=imitators_list, name="Imitators (q × A/M)",
                                            line=dict(color="#800000", width=2)))
        if b14_ad_daily > 0:
            fig_arrivals.add_trace(go.Scatter(x=days, y=advertising_list,
                                                name=f"Advertising (${b14_ad_daily}/day, {b14_ad_duration}d)",
                                                line=dict(color="#b8860b", width=2)))
        fig_arrivals.add_trace(go.Scatter(x=days, y=total_arrivals, name="Total arrivals",
                                            line=dict(color="#2d6a2e", width=2.5, dash="dash")))
        fig_arrivals.update_layout(
            height=400, xaxis_title="Day",
            yaxis_title="Daily Arrivals",
            title=dict(text=f"Daily Customer Arrivals (P(buy at ${b14_price}) = {p_buy:.1%})",
                         x=0.5, xanchor="center", y=0.97, yanchor="top"),
            margin=dict(l=0, r=0, t=90, b=0),
            legend=dict(orientation="h", yanchor="top", y=1.07,
                         xanchor="center", x=0.5),
        )
        st.plotly_chart(fig_arrivals, use_container_width=True)

        # Key check metrics at specific days (matching exercise)
        st.markdown("**Arrivals at Key Days** (Day 1, 364, 728)")
        check_days = [0, 363, 727]  # 0-indexed: day 1, day 364, day 728
        check_data = []
        for dx in check_days:
            if dx < len(days):
                check_data.append({
                    "Day": days[dx],
                    "Innovators": f"{innovators_list[dx]:.2f}",
                    "Imitators": f"{imitators_list[dx]:.2f}",
                    "Ad-attracted": f"{advertising_list[dx]:.2f}",
                    "Total arrivals": f"{total_arrivals[dx]:.2f}",
                    "Purchases": f"{purchases_list[dx]:.2f}",
                })
        st.dataframe(pd.DataFrame(check_data), use_container_width=True, hide_index=True)

        # Cumulative contribution (4-year) — inside right column, fills whitespace next to advertising inputs
        var_cost = b14_materials + W14_SHIPPING + W14_HANDLING + b14_mfg_oh
        cm_per_unit = b14_price * (1 - w14_comm_frac) - var_cost
        daily_cm = [cm_per_unit * p for p in purchases_list]
        cumulative_cm = []
        cum = 0
        total_ad_spend = 0
        for t, d_cm in zip(days, daily_cm):
            cum += d_cm
            if t <= b14_ad_duration:
                cum -= b14_ad_daily
                total_ad_spend += b14_ad_daily
            cumulative_cm.append(cum)

        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(x=days, y=cumulative_cm, name="Cumulative Contribution",
                                       line=dict(color="#2d6a2e", width=2.5),
                                       fill="tozeroy", fillcolor="rgba(45,106,46,0.1)"))
        fig_cum.add_hline(y=0, line_dash="dot", line_color="gray")
        fig_cum.update_layout(height=300, xaxis_title="Day",
                               yaxis_title="Cumulative Contribution ($)",
                               yaxis_tickformat="$,.0f",
                               title="4-Year Cumulative Contribution (net of advertising)",
                               margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_cum, use_container_width=True)

    # 6 metric boxes in horizontal alignment (full width)
    total_purchases = sum(purchases_list)
    final_cum_cm = cumulative_cm[-1] if cumulative_cm else 0
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("P(buy)", f"{p_buy:.1%}")
    m2.metric("Total Purchases", f"{total_purchases:,.0f} u")
    m3.metric("Market Served", f"{total_purchases/b14_M*100:.1f}%")
    m4.metric("Ad Spend", f"${total_ad_spend:,}")
    m5.metric("Cumulative CM", f"${final_cum_cm:,.0f}",
               delta=f"${final_cum_cm/1000:.0f}K")
    m6.metric("CM / unit", f"${cm_per_unit:.2f}")

    # D3 Exercise verification
    with st.expander("**D3 Exercise Verification** (default params: MD Cancer Bladder/Kidney)", expanded=False):
        st.markdown("""
**D3 Solutions check** (market size 15K, mean WTP $1,300, std $130, materials $375, OH $80, ship $20):

| Scenario | Expected Year 4 Cumulative CM |
|---|---|
| P=$900, no ad | **$3,249K** |
| P=$900, $3,000/day for 364 days | **$2,391K** |
| P=$1,200, no ad | **$4,597K** (BEST) |
| P=$1,200, $3,000/day for 364 days | **$4,287K** |

**Key insight:** At $900 retail, advertising HURTS profit ($2,391K < $3,249K). Why?
- At $900, P(buy) ≈ 99.9% (well above 3σ below mean), so nearly all arrivals buy anyway
- Advertising just accelerates when they arrive, doesn't increase total demand
- $3,000 × 364 = $1.09M in ad spend minus minor value = net loss

**Better to price at $1,200** (P(buy) ≈ 22%) without ad:
- More profit per sale covers slower cumulative adoption
- Ad at $1,200 provides marginal benefit but still loses to no-ad
        """)

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2: SCENARIO COMPARISON (Price × Advertising)
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("2. Scenario Comparison — Price × Advertising")
    st.caption("Compare 4 scenarios side-by-side: 2 price points × (no ad vs with ad)")

    scc1, scc2, scc3 = st.columns([1, 1, 2])
    with scc1:
        sc_p_low = st.number_input("Price Low", value=900, step=25, key="sc_p_low")
        sc_p_high = st.number_input("Price High", value=1200, step=25, key="sc_p_high")
    with scc2:
        sc_ad_amount = st.number_input("Ad Spend ($/day)", value=3000, step=500, key="sc_ad_amount")
        sc_ad_days = st.number_input("Ad Duration (days)", value=364, step=30, key="sc_ad_days")
    with scc3:
        st.caption("Uses market params from Section 1. Adjust mean WTP, std, market size, etc. above to match your target market.")

    # Scenarios — use cached simulate_scenario_traj for 100× speedup on slider repeats
    scenarios = [
        ("A", sc_p_low, 0, "Low price, no ad"),
        ("B", sc_p_low, sc_ad_amount, f"Low price, ${sc_ad_amount}/day ad for {sc_ad_days}d"),
        ("C", sc_p_high, 0, "High price, no ad"),
        ("D", sc_p_high, sc_ad_amount, f"High price, ${sc_ad_amount}/day ad for {sc_ad_days}d"),
    ]

    def _run_scenario(price, ad):
        return simulate_scenario_traj(
            price=float(price), ad_daily=float(ad), ad_duration=int(sc_ad_days),
            M=int(b14_M), p=float(b14_p), q=float(b14_q),
            p_ad_per_500=float(b14_p_ad_per_500),
            mean_wtp=float(b14_mean), std_wtp=float(b14_std),
            materials=float(b14_materials), mfg_oh=float(b14_mfg_oh),
            shipping=float(W14_SHIPPING), handling=float(W14_HANDLING),
            commission_frac=float(w14_comm_frac), days_total=1460,
        )

    # Run each scenario ONCE and reuse result for table + chart (also cached)
    scenario_results = {}
    for label, price, ad, _ in scenarios:
        scenario_results[label] = _run_scenario(price, ad)

    sc_results = []
    for label, price, ad, desc in scenarios:
        r = scenario_results[label]
        sc_results.append({
            "Scenario": f"{label}: {desc}",
            "Price": f"${price:,}",
            "Ad Spend Total": f"${ad * sc_ad_days:,}",
            "P(buy)": f"{r['p_buy']:.1%}",
            "Units Sold (4yr)": f"{r['cum_units']:,.0f}",
            "CM/unit": f"${r['cm_per_unit']:.0f}",
            "Cumulative CM": f"${r['cum_cm_final']:,.0f}",
            "vs Best": "",
        })

    # Identify best
    best_cm = max(r["cum_cm_final"] for r in scenario_results.values())
    for i, r_row in enumerate(sc_results):
        label, _, _, _ = scenarios[i]
        r = scenario_results[label]
        delta = r["cum_cm_final"] - best_cm
        r_row["vs Best"] = f"${delta:,.0f}" if delta < 0 else "🏆 Best"

    st.dataframe(pd.DataFrame(sc_results), use_container_width=True, hide_index=True)

    # Visualize (reuses cached trajectories — zero extra compute)
    fig_sc = go.Figure()
    for label, price, ad, _ in scenarios:
        r = scenario_results[label]
        fig_sc.add_trace(go.Scatter(
            x=list(range(1, 1461)), y=r["trajectory"],
            name=f"{label}: ${price} {'w/ ad' if ad > 0 else ''}",
            mode="lines",
        ))
    fig_sc.add_hline(y=0, line_dash="dot", line_color="gray")
    fig_sc.update_layout(height=450, xaxis_title="Day",
                          yaxis_title="Cumulative CM ($)", yaxis_tickformat="$,.0f",
                          title=dict(text="Cumulative Contribution over 4 Years",
                                       x=0.5, xanchor="center", y=0.97, yanchor="top"),
                          margin=dict(l=0, r=0, t=90, b=0),
                          legend=dict(orientation="h", yanchor="top", y=1.07,
                                        xanchor="center", x=0.5))
    st.plotly_chart(fig_sc, use_container_width=True)

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 3: ADVERTISING DECISION FRAMEWORK
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("3. Advertising Decision Framework")

    adv_col1, adv_col2 = st.columns(2)
    with adv_col1:
        st.markdown("**✅ Advertise when...**")
        st.markdown("""
- **Early in product lifecycle** — most arrivals are innovators, maximum leverage on future imitators
- **Product is profitable** at current price — otherwise advertising amplifies losses
- **You have supply capacity** — customers arrive same day $ is spent; stockouts = lost forever
- **To stave off competitor entry** — signal commitment, build brand loyalty
- **To avoid price wars** — differentiated demand via advertising buys you time
- **Low P(buy)** at current price — advertising creates new arrivals that wouldn't come organically
        """)
    with adv_col2:
        st.markdown("**❌ Don't advertise when...**")
        st.markdown("""
- **Late in product lifecycle** — few customers remain, most arrivals are imitators (already coming)
- **Unprofitable product** — ad spend compounds losses
- **At stockout risk** — you'll turn away paying customers
- **At low prices (high P(buy))** — customers arrive anyway, ad just pulls demand forward
- **When competitors match** — Bertrand-like race to zero
- **Short horizon remaining** — not enough time to recoup ad investment via imitator cascade
        """)

    st.markdown("#### Advertising ROI Calculator")
    ar_col1, ar_col2, ar_col3 = st.columns(3)
    with ar_col1:
        ar_current_price = st.number_input("Current Price ($)", value=1200, step=50, key="ar_price")
        ar_cm_per_unit = st.number_input("CM per Unit ($)", value=500, step=25, key="ar_cm")
    with ar_col2:
        ar_cur_arrivals = st.number_input("Current Arrivals/day (from Bass)", value=5, step=1, key="ar_arr")
        ar_p_buy_cur = st.number_input("Current P(buy)", value=0.22, step=0.05, format="%.2f", key="ar_pbuy")
    with ar_col3:
        ar_ad_spend = st.number_input("Proposed Ad $/day", value=3000, step=500, key="ar_spend")
        ar_ad_incr_p = st.number_input("Incremental customers/day", value=18, step=1, key="ar_incr",
                                          help="Ad customers = (ad/$500) × p_inc × remaining market. Check Bass model above.")

    ar_incremental_daily_cm = ar_ad_incr_p * ar_p_buy_cur * ar_cm_per_unit - ar_ad_spend
    ar_breakeven_incr = ar_ad_spend / (ar_p_buy_cur * ar_cm_per_unit) if ar_p_buy_cur * ar_cm_per_unit > 0 else float("inf")

    if ar_incremental_daily_cm > 0:
        st.success(f"✅ Advertising adds ${ar_incremental_daily_cm:.0f}/day in net CM. "
                    f"Need {ar_breakeven_incr:.1f} incremental customers/day to break even — currently projecting {ar_ad_incr_p}.")
    else:
        st.error(f"❌ Advertising costs ${-ar_incremental_daily_cm:.0f}/day in net CM. "
                  f"Need {ar_breakeven_incr:.1f} incremental customers/day to break even — currently only {ar_ad_incr_p}. "
                  f"Either raise price to increase CM per unit, or skip ads.")

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 4: ENHANCED DEBT MODEL (Tranche-based)
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("4. Debt Capacity & Bond Issuance")
    st.caption("Zero-coupon bonds, $1,000 face, 5-year maturity, semi-annual compounding. Sequential tranche: Excellent → Good → Poor.")

    debt_col1, debt_col2 = st.columns([1, 2])
    with debt_col1:
        d_ebit = st.number_input("Yearly EBIT / Operating Income ($)", value=100000, step=10000, key="d_ebit",
                                   help="Last full quarter × 4")
        d_existing_interest = st.number_input("Existing Interest ($/yr)", value=0, step=500, key="d_exist")

    RATES = {"Excellent": (20, 0.10), "Good": (7, 0.15), "Poor": (2, 0.25)}

    # Calculate debt capacity by tranche
    # Rule: exhaust Excellent first, then Good, then Poor
    # At each rating, max total interest = EBIT / hurdle
    # EAR = (1 + APR/2)^2 - 1
    def ear(apr):
        return (1 + apr/2) ** 2 - 1

    def bond_price(apr, years=5):
        # Zero-coupon price from face $1000
        return 1000 / (1 + apr/2) ** (2 * years)

    tranche_data = []
    used_interest = d_existing_interest
    cum_bonds_face = 0
    cum_bonds_cash = 0
    for rating, (hurdle, apr) in RATES.items():
        max_total_interest_for_this_rating = d_ebit / hurdle if hurdle > 0 else 0
        incremental_interest = max(0, max_total_interest_for_this_rating - used_interest)
        # Each bond face $1000 at APR rate → annual imputed interest ≈ $1000 × EAR
        interest_per_bond = 1000 * ear(apr) / 5  # approx — actually accreting, use simple avg
        # For simplicity, use total interest over 5 years = 1000 - price, then divide by 5
        price = bond_price(apr)
        total_interest_per_bond_5yr = 1000 - price
        annual_interest_per_bond = total_interest_per_bond_5yr / 5
        # But the EAR formula is more accurate
        # Use simple: num bonds × APR × face = annual interest
        # Actually zero-coupon bonds don't pay coupons — imputed interest accretes
        # Game uses: yearly_interest = face × APR (approximation per assignment)
        yearly_interest_per_bond = 1000 * apr  # per game convention (approx)
        num_bonds = incremental_interest / yearly_interest_per_bond if yearly_interest_per_bond > 0 else 0
        face_value = num_bonds * 1000
        cash_received = num_bonds * price

        tranche_data.append({
            "Rating": rating,
            "Coverage Hurdle": f"{hurdle}×",
            "APR": f"{apr*100:.0f}%",
            "EAR": f"{ear(apr)*100:.2f}%",
            "Max Cumulative Interest": f"${max_total_interest_for_this_rating:,.0f}",
            "Incremental Interest": f"${incremental_interest:,.0f}",
            "# Bonds Issuable": f"{num_bonds:.1f}",
            "Face Value": f"${face_value:,.0f}",
            "Cash Received": f"${cash_received:,.0f}",
        })
        used_interest = max_total_interest_for_this_rating
        cum_bonds_face += face_value
        cum_bonds_cash += cash_received

    with debt_col2:
        st.dataframe(pd.DataFrame(tranche_data), use_container_width=True, hide_index=True)

        st.markdown(f"""
<div style="background:rgba(26,60,94,0.15); border-left:4px solid #1a3c5e;
    border-radius:6px; padding:0.8rem 1rem;">
<b>Total Debt Capacity</b><br>
<span style="font-size:1.2em;">Face Value: <b>${cum_bonds_face:,.0f}</b> | Cash Received: <b>${cum_bonds_cash:,.0f}</b></span>
</div>
""", unsafe_allow_html=True)

        st.caption(f"""
Bond price formula: P = $1,000 / (1 + APR/2)^10 (semi-annual compounding, 5 years).
Prices: Excellent={bond_price(0.10):,.2f} | Good={bond_price(0.15):,.2f} | Poor={bond_price(0.25):,.2f}
To go from no debt to maximum: cash received = ${cum_bonds_cash:,.0f}, but you commit to
${cum_bonds_face:,.0f} face value due in 5 years + interest expense reducing future flexibility.
        """)

    with st.expander("**Debt Decision Guide**", expanded=False):
        st.markdown(f"""
### When to Issue Bonds

**Rating-specific guidance:**

**Excellent ({RATES['Excellent'][0]}× coverage, {RATES['Excellent'][1]*100:.0f}% APR):**
- Cheapest debt, lowest risk. Issue aggressively if NPV > 0 at 15% cost of capital.
- Rule: coverage stays ≥ 20× → rating protected

**Good ({RATES['Good'][0]}× coverage, {RATES['Good'][1]*100:.0f}% APR):**
- Same as cost of capital (15%) — neutral NPV threshold
- Only issue if project NPV > 0 AT 15% (i.e., returns > 15%)

**Poor ({RATES['Poor'][0]}× coverage, {RATES['Poor'][1]*100:.0f}% APR):**
- 25% APR > 15% cost of capital → destroys value unless project IRR > 25%
- Usually a bad idea; emergency loans at 40% are even worse

### Strategic Moves

1. **Build to EBIT before issuing** — higher EBIT → bigger Excellent tranche at 10%
2. **Use for growth capex, not operating losses** — NPV-positive projects only
3. **Avoid Poor rating** unless you're confident of a big payoff
4. **Plan for 5-year maturity** — bonds come due at game end (day 1460). Match cash flows.

### Tranche Logic (per D3 model)

The simulation **automatically** fills tranches in order:
1. First bonds go at Excellent rate (cheapest) until 20× coverage breached
2. Next bonds go at Good rate until 7× coverage breached
3. Final bonds at Poor rate until 2× coverage breached
4. Beyond: no more issuance possible

**Current tranche result for your EBIT (${d_ebit:,}):**
- Excellent tranche face: **${float(tranche_data[0]['Face Value'].replace('$','').replace(',','')):,.0f}**
- Good tranche face: **${float(tranche_data[1]['Face Value'].replace('$','').replace(',','')):,.0f}**
- Poor tranche face: **${float(tranche_data[2]['Face Value'].replace('$','').replace(',','')):,.0f}**
        """)

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 5: NORMAL WTP vs UNIFORM WTP COMPARISON
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("5. Normal WTP vs Uniform WTP — Pricing Implications")
    st.caption("Pricing formulas differ by distribution assumption. Know which you're using.")

    with st.expander("**When to use which distribution**", expanded=False):
        st.markdown("""
### Normal Distribution (New per D3)
- Focus group reveals **mean and std dev** (or median and max, where mean ≈ median)
- WTP ~ N(μ, σ²) — most customers cluster near mean, tails on both sides
- P(buy at price P) = 1 − Φ((P − μ) / σ) where Φ is standard normal CDF
- No explicit min/max — theoretically unbounded
- Practical min/max: μ ± 3σ captures 99.7% of customers
- For the MD Cancer Bladder example: μ=$1,300, σ=$130 → practical range [$910, $1,690]

### Uniform Distribution [min, max] (V1 assumption)
- Focus group reveals **min and max** (or derive from median)
- WTP ~ Uniform[a, b] — equal mass everywhere in range
- P(buy at P) = (b − P) / (b − a) for P in (a, b)
- Optimal P = b/2 + var_fixed/1.6 (with 20% commission)

### Which is Right for the Gleacher Game?

The D3 Bass Model Exercise uses **Normal WTP** (mean $1,300, std $130).
The focus group UI screenshot showed **median and max** — which could indicate either:
- Normal: median = mean (for symmetric distribution)
- Uniform: median = (min+max)/2

Given the D3 Exercise uses Normal, **we should assume Normal distribution** going forward.
        """)

        # Side-by-side comparison at same price
        comp_col1, comp_col2, comp_col3 = st.columns(3)
        with comp_col1:
            comp_price = st.number_input("Test Price ($)", value=1200, step=50, key="comp_price")
            comp_mean = st.number_input("Normal: Mean WTP", value=1300, step=50, key="comp_mean")
            comp_std = st.number_input("Normal: Std Dev", value=130, step=10, key="comp_std")
        with comp_col2:
            comp_min = st.number_input("Uniform: Min WTP", value=1000, step=50, key="comp_min")
            comp_max = st.number_input("Uniform: Max WTP", value=1600, step=50, key="comp_max")

        p_buy_normal = 1 - normal_cdf(comp_price, comp_mean, comp_std)
        if comp_price <= comp_min:
            p_buy_unif = 1.0
        elif comp_price >= comp_max:
            p_buy_unif = 0.0
        else:
            p_buy_unif = (comp_max - comp_price) / (comp_max - comp_min)

        with comp_col3:
            st.metric("P(buy) — Normal", f"{p_buy_normal:.1%}")
            st.metric("P(buy) — Uniform", f"{p_buy_unif:.1%}")
            diff = p_buy_normal - p_buy_unif
            st.metric("Difference", f"{diff:+.1%}",
                       help="Positive = Normal predicts higher demand than Uniform")

    st.markdown("---")
    st.success("""
**🎯 Key Takeaways from D3 Practice:**
1. Use **Normal WTP distribution** with mean/std (not uniform)
2. Advertising has **diminishing returns** at high P(buy) — skip at low prices
3. Price higher → fewer sales but more profit per sale. Usually wins over 4 years.
4. Issue debt at **Excellent rate first** (10% APR < 15% cost of capital = NPV positive)
5. Customers attracted by advertising arrive **same day** — don't advertise without inventory
    """)

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 6: COBB-DOUGLAS + LITTLE'S LAW + CONTRIBUTION MARGIN TABLE
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("6. Cobb-Douglas + Little's Law + Contribution Margin")
    st.caption("4 factories side-by-side (Bench, Line, Cell, Custom) with shared K, L, batch inputs")

    # Shared inputs (apply to all 4 factories)
    st.markdown("**Shared Inputs** (same across all 4 factories)")
    w14_sh1, w14_sh2, w14_sh3, w14_sh4 = st.columns(4)
    with w14_sh1:
        w14_K = st.number_input("Capital K ($)", value=100000, step=10000, key="w14_K")
    with w14_sh2:
        w14_l = st.number_input("Daily Labor l ($/day)", value=2500, step=100, key="w14_l")
    with w14_sh3:
        w14_batch = st.number_input("Batch Size", value=100, step=10, key="w14_batch")
    with w14_sh4:
        w14_dpy = st.number_input("Days/year", value=364, step=1, key="w14_dpy")

    # 4 factory configurations (3 presets + 1 custom)
    W14_FACTORIES = [
        {"name": "Bench", "A": 0.009, "alpha": 0.10, "beta": 0.85, "setup": 0.05, "min_K": 0, "color": "#800000"},
        {"name": "Production Line", "A": 0.010, "alpha": 0.30, "beta": 0.75, "setup": 0.50, "min_K": 500000, "color": "#1a3c5e"},
        {"name": "Automated Cell", "A": 0.020, "alpha": 0.80, "beta": 0.30, "setup": 1.00, "min_K": 3000000, "color": "#2d6a2e"},
        {"name": "Custom", "A": 0.009, "alpha": 0.10, "beta": 0.85, "setup": 0.05, "min_K": 0, "color": "#b8860b"},
    ]

    # Custom factory parameters (editable)
    st.markdown("**Custom Factory Parameters** (4th column only)")
    w14_c1, w14_c2, w14_c3, w14_c4 = st.columns(4)
    with w14_c1:
        custom_A = st.number_input("Custom A", value=0.009, step=0.001,
                                     format="%.4f", key="w14_custom_A")
    with w14_c2:
        custom_alpha = st.number_input("Custom α", value=0.10, step=0.05,
                                         format="%.2f", key="w14_custom_alpha")
    with w14_c3:
        custom_beta = st.number_input("Custom β", value=0.85, step=0.05,
                                        format="%.2f", key="w14_custom_beta")
    with w14_c4:
        custom_setup = st.number_input("Custom setup (d)", value=0.05, step=0.05,
                                         format="%.2f", key="w14_custom_setup")

    # Apply custom values to 4th entry
    W14_FACTORIES[3]["A"] = custom_A
    W14_FACTORIES[3]["alpha"] = custom_alpha
    W14_FACTORIES[3]["beta"] = custom_beta
    W14_FACTORIES[3]["setup"] = custom_setup

    W14_DEP_YRS = 15

    # Calculate for each factory
    def calc_factory(f, K, l, batch, dpy):
        if K < f["min_K"]:
            return None
        L_yearly = l * dpy
        Y = f["A"] * (K ** f["alpha"]) * (L_yearly ** f["beta"])
        lambda_raw = Y / dpy
        batch_time = batch / lambda_raw if lambda_raw > 0 else float("inf")
        CT = batch_time + f["setup"]
        lambda_eff = batch / CT if CT > 0 else 0
        WIP = lambda_eff * CT
        daily_dep = K / W14_DEP_YRS / dpy
        daily_cost = l + daily_dep
        mfg_oh = daily_cost / lambda_eff if lambda_eff > 0 else 0
        return {
            "Y": Y, "lambda_raw": lambda_raw, "batch_time": batch_time,
            "CT": CT, "lambda_eff": lambda_eff, "WIP": WIP,
            "mfg_oh": mfg_oh, "daily_cost": daily_cost,
        }

    w14_fac_results = [calc_factory(f, w14_K, w14_l, w14_batch, w14_dpy) for f in W14_FACTORIES]

    # 4 side-by-side factory columns
    st.markdown("---")
    st.markdown("### Factory Comparison (side-by-side)")
    fac_cols = st.columns(4)
    for idx, (col, f, r) in enumerate(zip(fac_cols, W14_FACTORIES, w14_fac_results)):
        with col:
            st.markdown(
                f"<div style='background:{f['color']};color:white;padding:0.5rem 0.8rem;"
                f"border-radius:6px;font-weight:700;text-align:center;'>{f['name']}</div>",
                unsafe_allow_html=True,
            )
            st.caption(f"A={f['A']:.4f} | α={f['alpha']:.2f} | β={f['beta']:.2f} | setup={f['setup']:.2f}d")
            if r is None:
                st.error(f"Min K ${f['min_K']:,} not met")
                continue

            st.metric("Yearly Y", f"{r['Y']:,.0f} u/yr")
            st.metric("Daily λ raw", f"{r['lambda_raw']:.2f} u/d")
            st.metric("Daily λ eff", f"{r['lambda_eff']:.2f} u/d")
            st.metric("Batch Time", f"{r['batch_time']:.3f} d")
            st.metric("Setup", f"{f['setup']:.2f} d")
            st.metric("Total CT", f"{r['CT']:.3f} d")
            st.metric("WIP Inventory", f"{r['WIP']:.1f} units")
            st.metric("Mfg OH/unit", f"${r['mfg_oh']:.2f}")
            rts = f["alpha"] + f["beta"]
            rts_label = "Incr." if rts > 1.02 else ("Decr." if rts < 0.98 else "Const.")
            st.metric("α+β", f"{rts:.2f}", delta=f"{rts_label} returns", delta_color="off")

    # Pick which factory's Mfg OH feeds the CM table below
    st.markdown("---")
    oh_pick_col1, oh_pick_col2 = st.columns([1, 3])
    with oh_pick_col1:
        cm_factory_choice = st.selectbox(
            "CM Table uses overhead from:",
            [f["name"] for f in W14_FACTORIES],
            index=0, key="w14_cm_factory_pick",
        )
    picked_idx = [f["name"] for f in W14_FACTORIES].index(cm_factory_choice)
    picked_result = w14_fac_results[picked_idx]
    if picked_result is None:
        # Fallback to Bench
        picked_result = w14_fac_results[0]
        cm_factory_choice = "Bench"
    with oh_pick_col2:
        st.caption(f"**{cm_factory_choice}** Mfg OH/unit = ${picked_result['mfg_oh']:,.2f}. "
                   f"Effective λ = {picked_result['lambda_eff']:.2f} units/day. "
                   f"Change selector to see CM with different factory overhead.")

    # Export vars for CM table section below (preserves the original flow)
    w14_mfg_oh = picked_result["mfg_oh"]
    w14_lambda_eff = picked_result["lambda_eff"]
    w14_daily_factory_cost = picked_result["daily_cost"]

    # Contribution Margin Table — now with SELLER vs RETAILER cost allocation
    st.markdown("### 💰 Contribution Margin — Cost Allocation")
    st.caption("Per Gleacher Tips: **Retailer pays commission + handling. Wholesaler pays shipping + materials + mfg OH.** "
               "Use this to model wholesale price negotiation below.")

    w14_cm_c1, w14_cm_c2, w14_cm_c3, w14_cm_c4 = st.columns(4)
    with w14_cm_c1:
        w14_cm_price = st.number_input("Retail Price ($)", value=1200, step=25, key="w14_cm_price")
    with w14_cm_c2:
        w14_cm_materials = st.number_input("Materials ($/u)", value=100, step=10, key="w14_cm_mat")
    with w14_cm_c3:
        w14_cm_shipping = st.number_input("Shipping ($/u)", value=20, step=5, key="w14_cm_ship")
    with w14_cm_c4:
        w14_cm_handling = st.number_input("Handling ($/u)", value=10, step=1, key="w14_cm_hand")

    w14_cm_commission = w14_cm_price * w14_comm_frac

    # Cost groupings by who bears them
    SELLER = "#1a3c5e"   # blue for wholesaler/seller-borne
    RETAILER = "#b8860b"  # gold for retailer-borne
    seller_costs = w14_mfg_oh + w14_cm_materials + w14_cm_shipping
    retailer_costs = w14_cm_handling + w14_cm_commission

    # Integrated (own DC + own factory) CM: bear all costs
    w14_cm_total_cost = seller_costs + retailer_costs
    w14_cm_before_tax = w14_cm_price - w14_cm_total_cost
    w14_cm_day = w14_cm_before_tax * w14_lambda_eff

    cm_color = "#2d6a2e" if w14_cm_before_tax > 0 else "#b22222"
    pct_rev = lambda v: f"{v/w14_cm_price*100:.1f}%" if w14_cm_price > 0 else "—"

    st.markdown(f"""
<div style="border:1px solid rgba(128,128,128,0.3); border-radius:8px; padding:1rem;">
<table style="width:100%; border-collapse:collapse;">
<tr style="border-bottom:2px solid rgba(128,128,128,0.5);">
<th style="text-align:left;">Line</th>
<th style="text-align:center;">Borne by</th>
<th style="text-align:right;">Per Unit</th>
<th style="text-align:right;">Per Day (at λ_eff={w14_lambda_eff:.2f})</th>
<th style="text-align:right;">% Rev</th>
</tr>
<tr><td>Revenue</td>
<td style="text-align:center;"><b style="color:{RETAILER};">Retailer collects</b></td>
<td style="text-align:right;"><b>${w14_cm_price:,.2f}</b></td>
<td style="text-align:right;"><b>${w14_cm_price * w14_lambda_eff:,.2f}</b></td>
<td style="text-align:right;">100.0%</td></tr>
<tr style="background:rgba(26,60,94,0.08);">
<td>(−) Manufacturing Overhead</td>
<td style="text-align:center;"><b style="color:{SELLER};">Wholesaler (Seller)</b></td>
<td style="text-align:right;color:#b22222;">$({w14_mfg_oh:,.2f})</td>
<td style="text-align:right;color:#b22222;">$({w14_daily_factory_cost:,.2f})</td>
<td style="text-align:right;">{pct_rev(w14_mfg_oh)}</td></tr>
<tr style="background:rgba(26,60,94,0.08);">
<td>(−) Materials</td>
<td style="text-align:center;"><b style="color:{SELLER};">Wholesaler (Seller)</b></td>
<td style="text-align:right;color:#b22222;">$({w14_cm_materials:,.2f})</td>
<td style="text-align:right;color:#b22222;">$({w14_cm_materials * w14_lambda_eff:,.2f})</td>
<td style="text-align:right;">{pct_rev(w14_cm_materials)}</td></tr>
<tr style="background:rgba(26,60,94,0.08);">
<td>(−) Shipping</td>
<td style="text-align:center;"><b style="color:{SELLER};">Wholesaler (Seller)</b></td>
<td style="text-align:right;color:#b22222;">$({w14_cm_shipping:,.2f})</td>
<td style="text-align:right;color:#b22222;">$({w14_cm_shipping * w14_lambda_eff:,.2f})</td>
<td style="text-align:right;">{pct_rev(w14_cm_shipping)}</td></tr>
<tr style="border-top:1px solid rgba(26,60,94,0.5);background:rgba(26,60,94,0.12);">
<td><b>Subtotal: Wholesaler's COGS</b></td>
<td style="text-align:center;"><b style="color:{SELLER};">Seller bears</b></td>
<td style="text-align:right;color:{SELLER};"><b>$({seller_costs:,.2f})</b></td>
<td style="text-align:right;color:{SELLER};"><b>$({seller_costs * w14_lambda_eff:,.2f})</b></td>
<td style="text-align:right;">{pct_rev(seller_costs)}</td></tr>
<tr style="background:rgba(184,134,11,0.08);">
<td>(−) Handling</td>
<td style="text-align:center;"><b style="color:{RETAILER};">Retailer</b></td>
<td style="text-align:right;color:#b22222;">$({w14_cm_handling:,.2f})</td>
<td style="text-align:right;color:#b22222;">$({w14_cm_handling * w14_lambda_eff:,.2f})</td>
<td style="text-align:right;">{pct_rev(w14_cm_handling)}</td></tr>
<tr style="background:rgba(184,134,11,0.08);">
<td>(−) Commission ({W14_COMMISSION:.0f}%)</td>
<td style="text-align:center;"><b style="color:{RETAILER};">Retailer</b></td>
<td style="text-align:right;color:#b22222;">$({w14_cm_commission:,.2f})</td>
<td style="text-align:right;color:#b22222;">$({w14_cm_commission * w14_lambda_eff:,.2f})</td>
<td style="text-align:right;">{pct_rev(w14_cm_commission)}</td></tr>
<tr style="border-top:1px solid rgba(184,134,11,0.5);background:rgba(184,134,11,0.15);border-bottom:2px solid rgba(128,128,128,0.5);">
<td><b>Subtotal: Retailer's cost of sale</b></td>
<td style="text-align:center;"><b style="color:{RETAILER};">Retailer bears</b></td>
<td style="text-align:right;color:{RETAILER};"><b>$({retailer_costs:,.2f})</b></td>
<td style="text-align:right;color:{RETAILER};"><b>$({retailer_costs * w14_lambda_eff:,.2f})</b></td>
<td style="text-align:right;">{pct_rev(retailer_costs)}</td></tr>
<tr style="background:rgba({'45,106,46' if w14_cm_before_tax > 0 else '178,34,34'},0.2);">
<td><b>= Integrated CM (same party owns DC + factory)</b></td>
<td></td>
<td style="text-align:right;color:{cm_color};font-size:1.15em;"><b>${w14_cm_before_tax:,.2f}</b></td>
<td style="text-align:right;color:{cm_color};font-size:1.15em;"><b>${w14_cm_day:,.2f}</b></td>
<td style="text-align:right;color:{cm_color};"><b>{pct_rev(w14_cm_before_tax)}</b></td></tr>
</table>
</div>
""", unsafe_allow_html=True)

    # Waterfall chart with seller/retailer color grouping
    fig_wf = go.Figure(go.Waterfall(
        name="Per Unit",
        orientation="v",
        measure=["absolute", "relative", "relative", "relative", "relative", "relative", "total"],
        x=["Revenue", "Mfg OH<br>(seller)", "Materials<br>(seller)", "Shipping<br>(seller)",
           "Handling<br>(retailer)", "Commission<br>(retailer)", "CM"],
        y=[w14_cm_price, -w14_mfg_oh, -w14_cm_materials, -w14_cm_shipping,
           -w14_cm_handling, -w14_cm_commission, 0],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        increasing={"marker": {"color": "#2d6a2e"}},
        decreasing={"marker": {"color": "#b22222"}},
        totals={"marker": {"color": "#1a3c5e"}},
    ))
    fig_wf.update_layout(height=350, yaxis_title="$ per unit", yaxis_tickformat="$,.0f",
                          title=f"Waterfall: ${w14_cm_price} price → ${w14_cm_before_tax:,.2f} CM (integrated)",
                          margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_wf, use_container_width=True)

    # ── Wholesale Price Negotiation Tool ─────────────────────────────────────
    st.markdown("### 🤝 Wholesale Price Negotiation Tool")
    st.caption("If you sell to another team (wholesale), split the CM between Seller (you) and Retailer (partner). "
               "Use the slider to find a win-win wholesale price.")

    wpn_col1, wpn_col2 = st.columns([1, 2])
    with wpn_col1:
        # Wholesale price must be:
        # - >= seller's cost (materials + mfg OH + shipping) for seller to want the deal
        # - <= retail - retailer's cost (commission + handling) for retailer to make any margin
        seller_min_ws = seller_costs  # wholesale price = seller_costs → seller makes $0
        retailer_max_ws = w14_cm_price - retailer_costs  # retailer CM = 0 at this wholesale price

        if retailer_max_ws > seller_min_ws:
            default_ws = (seller_min_ws + retailer_max_ws) / 2  # ZOPA midpoint
            ws_slider = st.slider(
                "Wholesale Price ($)",
                int(seller_min_ws), int(retailer_max_ws),
                int(default_ws), step=10, key="w14_ws_slider",
                help=f"ZOPA: ${seller_min_ws:,.0f} (seller breakeven) to ${retailer_max_ws:,.0f} (retailer breakeven)",
            )
            has_zopa = True
        else:
            ws_slider = int(seller_min_ws)
            has_zopa = False

        st.markdown(f"""
**ZOPA Range (Zone of Possible Agreement):**
- Seller min (breakeven): **${seller_min_ws:,.2f}**
- Retailer max (breakeven): **${retailer_max_ws:,.2f}**
- Width: **${retailer_max_ws - seller_min_ws:,.2f}**
        """)
        if not has_zopa:
            st.error("🔴 **NO ZOPA** — seller's cost > retailer's revenue after commission/handling. Deal impossible at this retail price.")

    with wpn_col2:
        seller_cm = ws_slider - seller_costs
        retailer_cm = w14_cm_price - ws_slider - retailer_costs
        total_cm = seller_cm + retailer_cm
        seller_share = seller_cm / total_cm * 100 if total_cm > 0 else 0
        retailer_share = retailer_cm / total_cm * 100 if total_cm > 0 else 0

        # Side-by-side metrics
        sc_a, sc_b, sc_c = st.columns(3)
        with sc_a:
            st.markdown(f"""
<div style="background:{SELLER};color:white;border-radius:8px;padding:0.8rem;text-align:center;">
<b>Seller (Wholesaler)</b><br>
<span style="font-size:0.8em;opacity:0.8;">Receives ${ws_slider:,.0f} wholesale</span><br>
<b style="font-size:1.5em;">${seller_cm:,.0f}</b><br>
<span style="font-size:0.8em;">CM/unit ({seller_share:.0f}% of total)</span>
</div>
""", unsafe_allow_html=True)
        with sc_b:
            st.markdown(f"""
<div style="background:{RETAILER};color:white;border-radius:8px;padding:0.8rem;text-align:center;">
<b>Retailer</b><br>
<span style="font-size:0.8em;opacity:0.8;">Pays ${ws_slider:,.0f}, sells at ${w14_cm_price:,.0f}</span><br>
<b style="font-size:1.5em;">${retailer_cm:,.0f}</b><br>
<span style="font-size:0.8em;">CM/unit ({retailer_share:.0f}% of total)</span>
</div>
""", unsafe_allow_html=True)
        with sc_c:
            status_color = "#2d6a2e" if seller_cm > 0 and retailer_cm > 0 else "#b22222"
            status_text = "✅ Win-win" if seller_cm > 0 and retailer_cm > 0 else "⚠️ One side loses"
            st.markdown(f"""
<div style="background:{status_color};color:white;border-radius:8px;padding:0.8rem;text-align:center;">
<b>Combined</b><br>
<span style="font-size:0.8em;opacity:0.8;">Total CM split</span><br>
<b style="font-size:1.5em;">${total_cm:,.0f}</b><br>
<span style="font-size:0.8em;">{status_text}</span>
</div>
""", unsafe_allow_html=True)

        # Sweep chart: seller CM vs retailer CM across wholesale prices
        ws_range = list(range(int(seller_min_ws), int(retailer_max_ws) + 1, 10)) if retailer_max_ws > seller_min_ws else [int(seller_min_ws)]
        seller_cms = [ws - seller_costs for ws in ws_range]
        retailer_cms = [w14_cm_price - ws - retailer_costs for ws in ws_range]

        fig_ws = go.Figure()
        fig_ws.add_trace(go.Scatter(x=ws_range, y=seller_cms, name="Seller CM",
                                      line=dict(color=SELLER, width=2.5),
                                      fill="tozeroy", fillcolor=f"rgba(26,60,94,0.1)"))
        fig_ws.add_trace(go.Scatter(x=ws_range, y=retailer_cms, name="Retailer CM",
                                      line=dict(color=RETAILER, width=2.5),
                                      fill="tozeroy", fillcolor="rgba(184,134,11,0.1)"))
        fig_ws.add_vline(x=ws_slider, line_dash="dash", line_color="green",
                          annotation_text=f"Your WS: ${ws_slider}")
        fig_ws.add_hline(y=0, line_dash="dot", line_color="gray")
        fig_ws.update_layout(
            height=300, xaxis_title="Wholesale Price ($)",
            yaxis_title="CM per unit ($)", yaxis_tickformat="$,.0f",
            title=dict(text="CM Split Across Wholesale Prices",
                         x=0.5, xanchor="center", y=0.97, yanchor="top"),
            margin=dict(l=0, r=0, t=60, b=0),
            legend=dict(orientation="h", yanchor="top", y=1.07, xanchor="center", x=0.5),
        )
        st.plotly_chart(fig_ws, use_container_width=True)

    st.info(f"""
**Negotiation Strategy:**
- **If you are the seller**, push for wholesale price ABOVE midpoint (${(seller_min_ws + retailer_max_ws)/2:,.0f})
- **If you are the retailer**, push BELOW midpoint
- **Fair split**: wholesale = midpoint → 50/50 CM split
- Include WTP info in your shipping agreement comments (per Gleacher Tips) to accelerate negotiation
- Remember: total CM is FIXED at ${retailer_max_ws - seller_min_ws:,.0f} regardless of wholesale price —
  the split determines who takes how much
    """)

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 7: MARKET SEGMENT ANALYZER (default 5)
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("7. Market Segment Analyzer (Side-by-Side, up to 5)")
    st.caption("Normal WTP distribution (per D3 material). Price slider → P(buy) + CM live.")

    W14_STANDARD_MARKETS = {
        "MD-Heart (Temporal)": {"mean": 732, "std": 73, "size": 30000, "p": 0.0002, "q": 0.0035, "dso": 30,
                                  "feature": "Heartbeat Temporal", "dealbreaker": "GPS affects WTP",
                                  "premium_feature": "Heartbeat=Temporal"},
        "MD-Breast (Cancer)": {"mean": 1250, "std": 125, "size": 15000, "p": 0.0002, "q": 0.0035, "dso": 30,
                                 "feature": "Cancer Breast", "dealbreaker": "None",
                                 "premium_feature": "Blood vessel + Cancer"},
        "MD-Estrogen (Fertility)": {"mean": 770, "std": 77, "size": 15000, "p": 0.0002, "q": 0.0035, "dso": 30,
                                      "feature": "Hormone Estrogen", "dealbreaker": "None",
                                      "premium_feature": "Hormone=Estrogen"},
        "Law-Narcotic": {"mean": 1350, "std": 135, "size": 10000, "p": 0.00025, "q": 0.0025, "dso": 90,
                          "feature": "Toxicology Narcotic", "dealbreaker": "Needs GPS + cellular",
                          "premium_feature": "Toxicology=Narcotic"},
        "Clinic-Fertility (LH/FSH)": {"mean": 315, "std": 32, "size": 50000, "p": 0.00025, "q": 0.004, "dso": 10,
                                        "feature": "Hormone LH/FSH", "dealbreaker": "Bulky battery packs",
                                        "premium_feature": "Hormone=LH/FSH"},
    }

    # Military markets — Metropolis-only, from D2 Case Brief ("4 military segments")
    # Placeholders based on case brief (high WTP, moderate size, dark color preference)
    W14_MILITARY_MARKETS = {
        "Mil-Botulinum (Metropolis only)": {"mean": 1200, "std": 120, "size": 8000, "p": 0.00025, "q": 0.003, "dso": 60,
                                              "feature": "Toxicology Botulinum", "dealbreaker": "Needs GPS + cellular + dark finish",
                                              "premium_feature": "Toxicology=Botulinum"},
        "Mil-Sarin (Metropolis only)": {"mean": 1250, "std": 125, "size": 8000, "p": 0.00025, "q": 0.003, "dso": 60,
                                          "feature": "Toxicology Sarin", "dealbreaker": "Needs GPS + cellular + dark finish",
                                          "premium_feature": "Toxicology=Sarin"},
        "Mil-Cyclosarin (Metropolis only)": {"mean": 1250, "std": 125, "size": 8000, "p": 0.00025, "q": 0.003, "dso": 60,
                                               "feature": "Toxicology Cyclosarin", "dealbreaker": "Needs GPS + cellular + dark finish",
                                               "premium_feature": "Toxicology=Cyclosarin"},
        "Mil-Anatoxin-A (Metropolis only)": {"mean": 1200, "std": 120, "size": 8000, "p": 0.00025, "q": 0.003, "dso": 60,
                                               "feature": "Toxicology Anatoxin-A", "dealbreaker": "Needs GPS + cellular + dark finish",
                                               "premium_feature": "Toxicology=Anatoxin-A"},
    }

    # Region toggle
    w14_region_col1, w14_region_col2 = st.columns([1, 2])
    with w14_region_col1:
        w14_is_metropolis = st.checkbox("Analyzing Metropolis region",
                                           value=False, key="w14_metropolis_toggle",
                                           help="Metropolis has larger non-military markets AND 4 military markets unavailable elsewhere")
    with w14_region_col2:
        if w14_is_metropolis:
            st.info("🏛️ **Metropolis mode:** non-military market sizes boosted ×1.5. 4 military markets unlocked (Metropolis-only). Deal breaker: all military needs GPS + cellular + dark finish.")
        else:
            st.caption("Standard region mode. Military markets hidden (only available in Metropolis per Case Brief).")

    # Build market list based on region
    W14_MARKETS = {}
    for k, v in W14_STANDARD_MARKETS.items():
        v_copy = dict(v)
        if w14_is_metropolis:
            # Non-military markets are LARGER in Metropolis
            v_copy["size"] = int(v_copy["size"] * 1.5)
        W14_MARKETS[k] = v_copy
    if w14_is_metropolis:
        W14_MARKETS.update(W14_MILITARY_MARKETS)

    w14_m_top1, w14_m_top2 = st.columns([1, 3])
    with w14_m_top1:
        max_markets = min(5, len(W14_MARKETS))
        w14_n_markets = st.number_input("# Markets to compare", min_value=2, max_value=max_markets,
                                           value=max_markets, step=1, key="w14_n_markets")
    with w14_m_top2:
        w14_m_materials = st.number_input("Your Materials Cost ($/unit)",
                                             value=100, step=10, key="w14_m_materials")

    market_keys = list(W14_MARKETS.keys())
    w14_mkt_cols = st.columns(int(w14_n_markets))
    w14_mkt_summary = []

    for i, col in enumerate(w14_mkt_cols):
        with col:
            default_idx = i if i < len(market_keys) else 0
            w14_mkt_sel = st.selectbox(f"Market {i+1}", market_keys, index=default_idx,
                                         key=f"w14_mkt_sel_{i}")
            m = W14_MARKETS[w14_mkt_sel]

            st.markdown(f"""
<div style="background:rgba(26,60,94,0.15);border-left:3px solid #1a3c5e;
    border-radius:6px;padding:0.5rem 0.7rem;margin-bottom:0.3rem;font-size:0.75rem;">
<b>{w14_mkt_sel}</b><br>
Feature: {m['feature']}<br>
Size: {m['size']:,} | Bass p: {m['p']} q: {m['q']}<br>
DSO: <b>{m['dso']}d</b> | DB: <span style="color:{'#b22222' if m['dealbreaker'] != 'None' else '#2d6a2e'};">{m['dealbreaker']}</span>
</div>
""", unsafe_allow_html=True)

            w14_mean_in = st.number_input("Mean WTP", value=m["mean"], step=25,
                                            key=f"w14_m_mean_{i}")
            w14_std_in = st.number_input("Std Dev WTP", value=m["std"], step=5,
                                           key=f"w14_m_std_{i}")
            # Market size now adjustable (slider range based on default × 0.5 to × 2)
            w14_mkt_size_in = st.slider("Market Size",
                                          int(m["size"] * 0.3), int(m["size"] * 2.5),
                                          int(m["size"]), step=500,
                                          key=f"w14_m_size_{i}",
                                          help=f"Default from market research: {m['size']:,}. Actual in-game may differ.")

            # P(buy) helper — uses cached _normal_cdf
            def _pbuy(P, mu, sigma):
                return 1 - _normal_cdf(float(P), float(mu), float(sigma))

            # Price slider from cost to mean + 3σ
            w14_p_min_slider = int(w14_m_materials + W14_HANDLING + W14_SHIPPING)
            w14_p_max_slider = int(w14_mean_in + 3 * w14_std_in)

            # Find numerical optimum (CACHED — runs once per unique input set)
            best_p = find_optimal_price_normal(
                price_min=int(w14_p_min_slider), price_max=int(w14_p_max_slider),
                mean_wtp=float(w14_mean_in), std_wtp=float(w14_std_in),
                materials=float(w14_m_materials), shipping=float(W14_SHIPPING),
                handling=float(W14_HANDLING), commission_frac=float(w14_comm_frac),
                step=5,
            )

            w14_m_price = st.slider("Your Price ($)",
                                      w14_p_min_slider, w14_p_max_slider, best_p,
                                      step=10, key=f"w14_m_price_{i}",
                                      help=f"Default = optimum (${best_p})")

            w14_p_buy = _pbuy(w14_m_price, w14_mean_in, w14_std_in)
            w14_m_comm = w14_m_price * w14_comm_frac
            w14_m_cm = w14_m_price - w14_m_comm - W14_HANDLING - w14_m_materials - W14_SHIPPING
            w14_m_cm_per_arr = w14_m_cm * w14_p_buy

            # Bass peak arrivals (uses adjustable market size)
            peak_q_val = w14_mkt_size_in * ((m["p"]+m["q"])**2) / (4*m["q"]) if m["q"] > 0 else 0

            cm_c = "#2d6a2e" if w14_m_cm > 0 else "#b22222"
            st.markdown(f"""
<div style="background:rgba({'45,106,46' if w14_m_cm > 0 else '178,34,34'},0.12);
    border-left:3px solid {cm_c};padding:0.5rem 0.7rem;border-radius:5px;margin-top:0.3rem;">
<span style="opacity:0.7;font-size:0.7rem;">Price ${w14_m_price:,}</span><br>
P(buy): <b>{w14_p_buy:.1%}</b><br>
CM/u: <b style="color:{cm_c};">${w14_m_cm:,.0f}</b><br>
CM/arrival: <b style="color:{cm_c};">${w14_m_cm_per_arr:,.0f}</b><br>
Peak sales: {peak_q_val * w14_p_buy:,.1f}/day
</div>
""", unsafe_allow_html=True)

            w14_mkt_summary.append({
                "Market": w14_mkt_sel,
                "Size": f"{w14_mkt_size_in:,}",
                "Mean/Std WTP": f"${w14_mean_in:,}/${w14_std_in:,}",
                "Price": f"${w14_m_price:,}",
                "P(buy)": f"{w14_p_buy:.0%}",
                "CM/u": f"${w14_m_cm:,.0f}",
                "CM/arrival": f"${w14_m_cm_per_arr:,.0f}",
                "Peak/day": f"{peak_q_val * w14_p_buy:,.1f}",
                "DSO": f"{m['dso']}d",
                "DB": m["dealbreaker"],
            })

    st.markdown("**Market Summary**")
    st.dataframe(pd.DataFrame(w14_mkt_summary), use_container_width=True, hide_index=True)

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 8: PRODUCT DESIGN ROI (default 5)
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("8. Product Design ROI Calculator (up to 5 products)")
    st.caption("Base features + detection agenda. Compare design costs, break-even, and margins side-by-side.")

    W14_BASE_FEATURES = {
        "Platform": {"Wristband": 0, "Chest-worn": 0, "Ankle/leg": 0},
        "GPS": {"No GPS": 0, "GPS enabled": 5},
        "Network": {"Bluetooth": 0, "2.4 GHz cellular": 10, "5 GHz cellular": 15},
        "Power": {"Ni-Cd": 0, "Polymer": 5, "Two-pack Ni-Cd": 10, "Two-pack polymer": 15},
        "Finish": {"Original": 0, "Dyed": 2, "Painted": 3},
    }

    W14_DETECTION = {
        "Heartbeat": {"None": (3, 1000, 0), "Pulse only": (15, 30000, 15), "Temporal": (90, 135000, 25)},
        "Blood vessel": {"None": (3, 1000, 0), "Systolic only": (30, 75000, 10),
                          "Systolic & diastolic": (90, 135000, 15), "Full profile": (120, 180000, 40)},
        "Dissolved gasses": {"None": (3, 1000, 0), "O2 only": (30, 75000, 15),
                              "O2, N2, CO2": (90, 135000, 20), "Full C,N,O": (90, 135000, 40)},
        "Toxicology": {"None": (3, 1000, 0), "Ethanol": (30, 150000, 95),
                        "Amphetamine": (90, 250000, 140), "THC": (90, 250000, 140),
                        "Barbiturate": (90, 250000, 140), "Narcotic": (90, 250000, 140)},
        "Hormone": {"None": (3, 1000, 0), "LH": (30, 45000, 20), "LH and FSH": (60, 75000, 50),
                     "Estrogen": (60, 75000, 60), "Progesterone": (60, 75000, 60),
                     "Testosterone": (60, 75000, 50)},
        "Metabolic": {"None": (3, 1000, 0), "Thyroxine": (90, 90000, 155),
                       "Bilirubin": (90, 90000, 150), "Proteins": (90, 90000, 170),
                       "Uric acid": (90, 90000, 160)},
    }

    W14_PRESETS = {
        "A — Heart View": {"Platform": "Chest-worn", "GPS": "GPS enabled", "Network": "2.4 GHz cellular",
                            "Power": "Polymer", "Finish": "Original",
                            "Heartbeat": "Temporal", "Blood vessel": "None",
                            "Dissolved gasses": "None", "Toxicology": "None",
                            "Hormone": "None", "Metabolic": "None", "price": 700},
        "B — Estrogen (minimal)": {"Platform": "Wristband", "GPS": "No GPS", "Network": "Bluetooth",
                                     "Power": "Polymer", "Finish": "Original",
                                     "Heartbeat": "None", "Blood vessel": "None",
                                     "Dissolved gasses": "None", "Toxicology": "None",
                                     "Hormone": "Estrogen", "Metabolic": "None", "price": 712},
        "C — Narcotic": {"Platform": "Ankle/leg", "GPS": "GPS enabled", "Network": "5 GHz cellular",
                          "Power": "Two-pack polymer", "Finish": "Original",
                          "Heartbeat": "None", "Blood vessel": "None",
                          "Dissolved gasses": "None", "Toxicology": "Narcotic",
                          "Hormone": "None", "Metabolic": "None", "price": 1200},
        "D — Breast Cancer": {"Platform": "Chest-worn", "GPS": "GPS enabled", "Network": "2.4 GHz cellular",
                                "Power": "Polymer", "Finish": "Dyed",
                                "Heartbeat": "Pulse only", "Blood vessel": "Systolic only",
                                "Dissolved gasses": "None", "Toxicology": "None",
                                "Hormone": "None", "Metabolic": "None", "price": 1250},
        "E — Clinic Fertility": {"Platform": "Wristband", "GPS": "No GPS", "Network": "Bluetooth",
                                   "Power": "Polymer", "Finish": "Original",
                                   "Heartbeat": "None", "Blood vessel": "None",
                                   "Dissolved gasses": "None", "Toxicology": "None",
                                   "Hormone": "LH and FSH", "Metabolic": "None", "price": 315},
    }

    w14_pd_t1, w14_pd_t2 = st.columns([1, 3])
    with w14_pd_t1:
        w14_n_products = st.number_input("# Products to design", min_value=1, max_value=5,
                                            value=5, step=1, key="w14_n_products")
    with w14_pd_t2:
        st.caption("Base features: design cost $0/0 days (not published — override if confirmed in-game). "
                   "Detection features: exact values from Design Guide.")

    preset_keys = list(W14_PRESETS.keys())
    w14_pd_cols = st.columns(int(w14_n_products))
    w14_pd_summary = []

    for i, col in enumerate(w14_pd_cols):
        with col:
            default_idx = i if i < len(preset_keys) else 0
            p_sel = st.selectbox(f"Preset P{i+1}", preset_keys, index=default_idx,
                                   key=f"w14_pd_preset_{i}")
            preset = W14_PRESETS[p_sel]

            st.markdown("**Base Features** (materials $/unit shown)")
            w14_sel_base = {}
            for attr, opts in W14_BASE_FEATURES.items():
                default_feat = preset.get(attr, list(opts.keys())[0])
                # Create labeled options: "Feature name — $X/u materials"
                labeled = {f"{feat} — ${mat}/u mat": feat for feat, mat in opts.items()}
                labels = list(labeled.keys())
                default_label = next((l for l, f in labeled.items() if f == default_feat), labels[0])
                idx = labels.index(default_label)
                chosen = st.selectbox(attr, labels, index=idx,
                                        key=f"w14_pd_base_{attr}_{i}")
                w14_sel_base[attr] = labeled[chosen]

            st.markdown("**Detection Agenda** (days / design $ / materials $/u shown)")
            w14_sel_det = {}
            for attr, opts in W14_DETECTION.items():
                default_feat = preset[attr]
                # Create labeled options: "Feature — 60d, $75K, $60/u"
                labeled = {f"{feat} — {d}d, ${c/1000:.0f}K, ${m}/u": feat
                            for feat, (d, c, m) in opts.items()}
                labels = list(labeled.keys())
                default_label = next((l for l, f in labeled.items() if f == default_feat), labels[0])
                idx = labels.index(default_label)
                chosen = st.selectbox(attr, labels, index=idx,
                                        key=f"w14_pd_det_{attr}_{i}")
                w14_sel_det[attr] = labeled[chosen]

            # Totals
            base_mat = sum(W14_BASE_FEATURES[a][w14_sel_base[a]] for a in W14_BASE_FEATURES)
            det_days = max(W14_DETECTION[a][w14_sel_det[a]][0] for a in W14_DETECTION)
            det_cost = sum(W14_DETECTION[a][w14_sel_det[a]][1] for a in W14_DETECTION)
            det_mat = sum(W14_DETECTION[a][w14_sel_det[a]][2] for a in W14_DETECTION)

            total_days = det_days
            total_cost = det_cost
            total_mat = base_mat + det_mat

            w14_pd_price = st.number_input("Price ($)", value=preset["price"], step=25,
                                             key=f"w14_pd_price_{i}")
            w14_pd_sales = st.number_input("Est. Sales/day", value=5, step=1,
                                             key=f"w14_pd_sales_{i}")

            w14_pd_margin = (w14_pd_price * (1 - w14_comm_frac)
                              - W14_HANDLING - total_mat - W14_SHIPPING)
            w14_pd_be = (total_cost / (w14_pd_margin * w14_pd_sales)
                           if w14_pd_margin > 0 and w14_pd_sales > 0 else float("inf"))

            st.metric("Design Days", f"{total_days}")
            st.metric("Design Cost", f"${total_cost:,}")
            st.metric("Materials/u", f"${total_mat}")
            st.metric("CM/u", f"${w14_pd_margin:,.0f}")
            if w14_pd_be < float("inf"):
                st.metric("Break-even", f"{w14_pd_be:.0f} days",
                           delta=f"{w14_pd_be/30:.1f} months", delta_color="off")
            else:
                st.error("Negative margin")

            # ── Cannibalization Checker (per Gleacher Tips: "do not put key features of
            # high-WTP market into low-WTP product") ─────────────────────────
            cann_target_market = st.selectbox(
                "Target Market",
                ["(select)", "MD-Heart", "MD-Breast Cancer", "MD-Estrogen", "Law-Narcotic", "Clinic-Fertility",
                 "Trainers", "Military"],
                index=0, key=f"w14_pd_mkt_{i}",
                help="What market is this product intended for? We'll check for cannibalization risks."
            )

            warnings = []
            # Rule 1: Temporal heartbeat in non-MD-Heart products cannibalizes Heart View
            if w14_sel_det["Heartbeat"] == "Temporal" and cann_target_market not in ["(select)", "MD-Heart"]:
                warnings.append("🔴 **Temporal heartbeat** in non-MD-Heart product — cannibalizes Heart View sales")
            # Rule 2: Pulse heartbeat in non-Heart markets might pull from Heart View (less severe)
            if w14_sel_det["Heartbeat"] == "Pulse only" and cann_target_market not in ["(select)", "MD-Heart", "MD-Breast Cancer"]:
                warnings.append("🟡 **Pulse heartbeat** in non-Heart market may cannibalize Heart View")
            # Rule 3: Narcotic toxicology outside Law/Military wastes money
            if w14_sel_det["Toxicology"] in ["Narcotic", "Amphetamine", "THC", "Barbiturate"] and cann_target_market not in ["(select)", "Law-Narcotic", "Military"]:
                warnings.append(f"🟠 **{w14_sel_det['Toxicology']} toxicology** + non-Law/Mil market — $140/u materials wasted, may cannibalize Law product")
            # Rule 4: Estrogen hormone outside fertility markets
            if w14_sel_det["Hormone"] == "Estrogen" and cann_target_market not in ["(select)", "MD-Estrogen", "Clinic-Fertility"]:
                warnings.append("🟡 **Estrogen** outside fertility markets — paying for materials unlikely to convert to WTP")
            # Rule 5: Law Narcotic products must have GPS + cellular
            if (w14_sel_det["Toxicology"] == "Narcotic" and cann_target_market == "Law-Narcotic"
                    and (w14_sel_base["GPS"] == "No GPS" or w14_sel_base["Network"] == "Bluetooth")):
                warnings.append("🔴 **Law-Narcotic deal breaker** — Needs GPS + cellular (not Bluetooth)")
            # Rule 6: Fertility with bulky battery
            if (w14_sel_det["Hormone"] in ["LH", "LH and FSH", "Estrogen"] and "Two-pack" in w14_sel_base["Power"]):
                warnings.append("🟠 **Fertility deal breaker** — avoid bulky two-battery packs")

            if warnings:
                st.warning("**Cannibalization / Fit Warnings:**\n\n" + "\n\n".join(warnings))
            elif cann_target_market != "(select)":
                st.success("✅ No cannibalization flags. Features fit target market.")

            w14_pd_summary.append({
                "Product": f"P{i+1}: {p_sel}",
                "Target": cann_target_market,
                "Days": total_days,
                "Design $": f"${total_cost:,}",
                "Mat/u": f"${total_mat}",
                "Price": f"${w14_pd_price}",
                "CM/u": f"${w14_pd_margin:,.0f}",
                "Break-even": f"{w14_pd_be:.0f}d" if w14_pd_be < float("inf") else "N/A",
                "Warnings": len(warnings) if cann_target_market != "(select)" else "—",
            })

    st.markdown("**Product Comparison Summary**")
    st.dataframe(pd.DataFrame(w14_pd_summary), use_container_width=True, hide_index=True)

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 9: SUPPLY CHAIN TRADE-OFF CALCULATOR
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("9. Supply Chain Trade-Off Calculator")
    st.caption("Per Gleacher Tips: trade-offs between mail vs container, own DC vs wholesale, new factory vs capex expansion")

    sc_tab1, sc_tab2, sc_tab3 = st.tabs(["Mail vs Container", "Own DC vs Wholesale", "New Factory vs Capex"])

    # ── Tab 1: Mail vs Container breakeven ──────────────────────────────────
    with sc_tab1:
        mc_col1, mc_col2 = st.columns([1, 2])
        with mc_col1:
            mc_region = st.radio("Shipping distance",
                                    ["In-region", "Between regions"],
                                    index=0, key="w14_mc_region")
            mc_qty = st.slider("Order Quantity (units)", 10, 1500, 100, step=10, key="w14_mc_qty")

            if mc_region == "In-region":
                mail_total = (mc_qty / 10) * 200 if mc_qty > 0 else 0
                container_total = 5000  # flat
                mail_per_u = 20
                container_per_u = 5000 / mc_qty if mc_qty > 0 else 0
                mail_days = 1
                container_days = 7
                breakeven = 250  # $5000 / ($20 - $5)
            else:
                mail_total = (mc_qty / 10) * 400 if mc_qty > 0 else 0
                container_total = 10000
                mail_per_u = 40
                container_per_u = 10000 / mc_qty if mc_qty > 0 else 0
                mail_days = 3
                container_days = 21
                breakeven = 250  # $10,000 / ($40 - $10)

            cheaper = "Container" if container_total < mail_total else "Mail"
            faster = "Mail"
            savings = abs(mail_total - container_total)

        with mc_col2:
            st.markdown("**Results**")
            bc1, bc2, bc3 = st.columns(3)
            with bc1:
                st.metric("Mail Total", f"${mail_total:,.0f}", delta=f"{mail_days} day(s)")
                st.metric("Mail per unit", f"${mail_per_u:.2f}")
            with bc2:
                st.metric("Container Total", f"${container_total:,.0f}", delta=f"{container_days} day(s)")
                st.metric("Container per unit", f"${container_per_u:.2f}")
            with bc3:
                st.metric("Breakeven Qty", f"{breakeven} units")
                st.metric("Savings (cheaper)", f"${savings:,.0f}", delta=cheaper)

            # Plot: total cost vs quantity for both modes
            qty_range = list(range(10, 1501, 10))
            if mc_region == "In-region":
                mail_costs = [(q/10) * 200 for q in qty_range]
                container_costs = [5000] * len(qty_range)
            else:
                mail_costs = [(q/10) * 400 for q in qty_range]
                container_costs = [10000] * len(qty_range)

            fig_mc = go.Figure()
            fig_mc.add_trace(go.Scatter(x=qty_range, y=mail_costs, name="Mail", line=dict(color="#800000", width=2)))
            fig_mc.add_trace(go.Scatter(x=qty_range, y=container_costs, name="Container", line=dict(color="#1a3c5e", width=2)))
            fig_mc.add_vline(x=breakeven, line_dash="dash", line_color="gray",
                              annotation_text=f"Breakeven: {breakeven}")
            fig_mc.add_vline(x=mc_qty, line_dash="dot", line_color="green",
                              annotation_text=f"Your qty: {mc_qty}")
            fig_mc.update_layout(height=300, xaxis_title="Order Quantity (units)",
                                  yaxis_title="Total Shipping Cost ($)", yaxis_tickformat="$,.0f",
                                  title=dict(text=f"Mail vs Container ({mc_region.lower()})",
                                               x=0.5, xanchor="center", y=0.97, yanchor="top"),
                                  margin=dict(l=0, r=0, t=70, b=0),
                                  legend=dict(orientation="h", yanchor="top", y=1.07,
                                                xanchor="center", x=0.5))
            st.plotly_chart(fig_mc, use_container_width=True)

            st.info(f"""
**Decision:** At {mc_qty} units {mc_region.lower()}, **{cheaper}** is cheaper by **${savings:,.0f}** total.
Mail is always faster ({mail_days}d vs {container_days}d). If the speed difference matters for stockout risk,
the mail premium ({savings:,.0f} more) may be worth it even below the breakeven quantity.
            """)

    # ── Tab 2: Own DC vs Wholesale ─────────────────────────────────────────
    with sc_tab2:
        st.markdown("Compare: **build your own DC in a region** vs **sell through another team's DC** (wholesale).")
        dc_col1, dc_col2 = st.columns([1, 2])
        with dc_col1:
            dc_price = st.number_input("Retail Price ($/u)", value=1200, step=50, key="w14_dc_price")
            dc_expected_demand = st.number_input("Expected demand (units/day)", value=5, step=1, key="w14_dc_demand")
            dc_days_left = st.number_input("Days remaining", value=1000, step=50, key="w14_dc_days")
            dc_materials = st.number_input("Materials + ship ($/u)", value=140, step=10, key="w14_dc_mat")
            # Build-your-own DC params
            dc_build_cost = 2600000
            dc_build_days = 60
            dc_daily_cost = 2000
            dc_depreciation = dc_build_cost / 15 / 364
            # Wholesale: sell to another team at a discount, they retail
            dc_wholesale_price = st.number_input("Wholesale price to partner ($/u)", value=600, step=25, key="w14_dc_wsp")

        with dc_col2:
            # Own DC: full retail price minus commission minus handling minus materials minus daily DC opex
            own_dc_cm_per_unit = dc_price * (1 - w14_comm_frac) - 10 - dc_materials  # handling $10
            operating_days = max(0, dc_days_left - dc_build_days)
            own_dc_revenue_total = own_dc_cm_per_unit * dc_expected_demand * operating_days
            own_dc_opex_total = (dc_daily_cost + dc_depreciation) * operating_days
            own_dc_net = own_dc_revenue_total - own_dc_opex_total - dc_build_cost

            # Wholesale: partner takes 20% commission + handles everything retail-side
            # We ship to them, they retail
            ws_cm_per_unit = dc_wholesale_price - dc_materials - 20  # materials + shipping
            ws_revenue_total = ws_cm_per_unit * dc_expected_demand * dc_days_left
            ws_net = ws_revenue_total  # no capex, no daily opex from us

            winner = "Own DC" if own_dc_net > ws_net else "Wholesale"
            delta = abs(own_dc_net - ws_net)

            d1, d2, d3 = st.columns(3)
            with d1:
                st.markdown("**Own DC**")
                st.metric("CM/unit", f"${own_dc_cm_per_unit:,.0f}")
                st.metric("Operating days", f"{operating_days:.0f}")
                st.metric("Gross CM", f"${own_dc_revenue_total:,.0f}")
                st.metric("− Opex + capex", f"${own_dc_opex_total + dc_build_cost:,.0f}")
                st.metric("Net", f"${own_dc_net:,.0f}")
            with d2:
                st.markdown("**Wholesale**")
                st.metric("CM/unit", f"${ws_cm_per_unit:,.0f}")
                st.metric("Operating days", f"{dc_days_left}")
                st.metric("Gross CM", f"${ws_revenue_total:,.0f}")
                st.metric("− Opex", "$0 (partner pays)")
                st.metric("Net", f"${ws_net:,.0f}")
            with d3:
                color = "#2d6a2e" if winner == "Own DC" else "#1a3c5e"
                st.markdown(f"""
<div style="background:{color};color:white;border-radius:8px;padding:1rem;text-align:center;">
<span style="font-size:0.85em;opacity:0.8;">Recommendation</span><br>
<b style="font-size:1.6em;">{winner}</b><br>
<span style="font-size:0.9em;">${delta:,.0f} advantage</span>
</div>
""", unsafe_allow_html=True)

            st.info("""
**Key factors:**
- Own DC requires **60-day build** (no revenue during build)
- Own DC has **$2.6M capex + $2K/day opex**
- Wholesale preserves capital but caps price at wholesale level
- **Breakeven** depends on days remaining and demand volume

Per Gleacher Tips: *"Don't be shy about borrowing money and expanding, but make sure investments are positive NPV."*
            """)

    # ── Tab 3: New Factory vs Capex Expansion ──────────────────────────────
    with sc_tab3:
        st.markdown("Compare: **build a new factory** vs **add capital to existing factory**.")
        fx_col1, fx_col2 = st.columns([1, 2])
        with fx_col1:
            fx_current_K = st.number_input("Current factory K ($)", value=100000, step=50000, key="w14_fx_K")
            fx_add_capex = st.number_input("Additional capex to add ($)", value=400000, step=50000, key="w14_fx_add")
            fx_new_K = st.number_input("New factory K ($)", value=500000, step=50000, key="w14_fx_newK")
            fx_daily_l = st.number_input("Daily labor both options ($/day)", value=2500, step=500, key="w14_fx_l")
            fx_days_left = st.number_input("Days remaining", value=1000, step=50, key="w14_fx_days")
            fx_cm_per_unit = st.number_input("CM per unit ($)", value=400, step=50, key="w14_fx_cm")

        with fx_col2:
            # Option A: add capex to existing (30-day lead)
            A_K = fx_current_K + fx_add_capex
            A_lambda = 0.009 * (A_K ** 0.10) * ((fx_daily_l * 364) ** 0.85) / 364
            A_batch_time = 100 / A_lambda if A_lambda > 0 else float("inf")
            A_lambda_eff = 100 / (A_batch_time + 0.05)
            A_lead = 30
            A_operating_days = max(0, fx_days_left - A_lead)
            A_revenue = A_lambda_eff * A_operating_days * fx_cm_per_unit
            A_net = A_revenue - fx_add_capex - (fx_daily_l * (A_operating_days + A_lead))

            # Option B: build new factory (90-day build, separate factory at new_K)
            B_K = fx_new_K
            B_land = 100000
            B_lambda = 0.009 * (B_K ** 0.10) * ((fx_daily_l * 364) ** 0.85) / 364
            B_batch_time = 100 / B_lambda if B_lambda > 0 else float("inf")
            B_lambda_eff = 100 / (B_batch_time + 0.05)
            B_lead = 90
            B_operating_days = max(0, fx_days_left - B_lead)
            # Plus existing factory keeps running at its current throughput during the 90 days
            old_lambda = 0.009 * (fx_current_K ** 0.10) * ((fx_daily_l * 364) ** 0.85) / 364
            old_lambda_eff = 100 / ((100 / old_lambda if old_lambda > 0 else float("inf")) + 0.05)
            # B throughput during build = old only, after build = old + new
            B_units_during_build = old_lambda_eff * B_lead
            B_units_after_build = (old_lambda_eff + B_lambda_eff) * B_operating_days
            B_total_units = B_units_during_build + B_units_after_build
            B_revenue = B_total_units * fx_cm_per_unit
            # Labor for both factories through building + operating
            B_total_labor = fx_daily_l * fx_days_left + fx_daily_l * B_operating_days  # new factory uses labor only when running
            B_net = B_revenue - (fx_new_K + B_land) - B_total_labor

            winner = "Add Capex" if A_net > B_net else "New Factory"
            delta = abs(A_net - B_net)

            f1, f2, f3 = st.columns(3)
            with f1:
                st.markdown("**Option A: Add Capex**")
                st.metric("K after upgrade", f"${A_K:,}")
                st.metric("Effective λ", f"{A_lambda_eff:.2f}/day")
                st.metric("Lead time", f"{A_lead} days")
                st.metric("Operating days", f"{A_operating_days}")
                st.metric("Net CM (after cost)", f"${A_net:,.0f}")
            with f2:
                st.markdown("**Option B: New Factory**")
                st.metric("New factory K", f"${B_K:,}")
                st.metric("New λ", f"{B_lambda_eff:.2f}/day")
                st.metric("Combined λ", f"{old_lambda_eff + B_lambda_eff:.2f}/day")
                st.metric("Lead time", f"{B_lead} days")
                st.metric("Net CM (after cost)", f"${B_net:,.0f}")
            with f3:
                color = "#2d6a2e" if winner == "Add Capex" else "#1a3c5e"
                st.markdown(f"""
<div style="background:{color};color:white;border-radius:8px;padding:1rem;text-align:center;">
<span style="font-size:0.85em;opacity:0.8;">Recommendation</span><br>
<b style="font-size:1.5em;">{winner}</b><br>
<span style="font-size:0.9em;">${delta:,.0f} advantage</span>
</div>
""", unsafe_allow_html=True)

            st.info("""
**Key factors:**
- **Add Capex** = 30-day lead, existing factory keeps running at current rate during
- **New Factory** = 90-day build, adds incremental throughput on TOP of existing
- Cell/Line factories have minimum K requirements ($500K / $3M)
- Each factory has its own daily labor cost — two factories = 2× labor
            """)

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 10: CASH & TAX DISCIPLINE PLANNER
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("10. Cash & Tax Discipline Planner")
    st.caption("Per Gleacher Tips: make sure there's enough cash for quarterly taxes + plan for growth in working capital")

    ct_col1, ct_col2 = st.columns([1, 2])
    with ct_col1:
        st.markdown("**Quarterly Inputs**")
        ct_current_cash = st.number_input("Current Cash ($)", value=1579530, step=10000, key="w14_ct_cash")
        ct_quarter_revenue = st.number_input("This Q Revenue ($)", value=683000, step=10000, key="w14_ct_rev")
        ct_quarter_opex = st.number_input("This Q Opex ($)", value=680000, step=10000, key="w14_ct_opex",
                                             help="COGS + selling + DC opex + depreciation")
        ct_next_q_capex = st.number_input("Planned Capex next Q ($)", value=0, step=50000, key="w14_ct_capex")
        ct_next_q_div = st.number_input("Planned Dividends next Q ($)", value=0, step=10000, key="w14_ct_div")
        ct_next_q_ad = st.number_input("Planned Ad Spend next Q ($)", value=0, step=10000, key="w14_ct_ad")

    with ct_col2:
        # Tax calculation
        ct_quarter_op_income = ct_quarter_revenue - ct_quarter_opex
        ct_tax = max(0, ct_quarter_op_income) * 0.35

        # Cash flow projection for next quarter
        ct_next_q_opex_est = ct_quarter_opex  # assume similar
        ct_next_q_rev_est = ct_quarter_revenue  # assume similar (conservative)
        ct_next_q_operating_cash = ct_next_q_rev_est - ct_next_q_opex_est
        ct_next_q_ending_cash = (ct_current_cash + ct_next_q_operating_cash
                                    - ct_next_q_capex - ct_next_q_div - ct_next_q_ad - ct_tax)

        # Buffer recommendation
        recommended_buffer = ct_tax + ct_quarter_opex / 3  # tax + 1 month opex

        st.markdown("**Cash Flow Projection — Next Quarter**")
        ct_a, ct_b, ct_c = st.columns(3)
        with ct_a:
            st.metric("This Q Op. Income", f"${ct_quarter_op_income:,.0f}")
            st.metric("Tax due (35%)", f"${ct_tax:,.0f}",
                       delta="Paid end of quarter", delta_color="off")
        with ct_b:
            st.metric("Projected Op Cash Flow", f"${ct_next_q_operating_cash:,.0f}")
            st.metric("Projected Ending Cash", f"${ct_next_q_ending_cash:,.0f}",
                       delta=f"${ct_next_q_ending_cash - ct_current_cash:,.0f}")
        with ct_c:
            st.metric("Recommended Buffer", f"${recommended_buffer:,.0f}",
                       help="Tax + 1 month opex")
            buffer_status = "✅ Safe" if ct_next_q_ending_cash > recommended_buffer else "⚠️ Below buffer"
            if ct_next_q_ending_cash < 0:
                buffer_status = "🔴 EMERGENCY LOAN (40% APR!)"
            st.metric("Status", buffer_status)

        # Detailed waterfall for next quarter
        fig_cash = go.Figure(go.Waterfall(
            name="Q+1 cash flow",
            orientation="v",
            measure=["absolute", "relative", "relative", "relative", "relative", "relative", "total"],
            x=["Start Cash", "Operating CF", "− Capex", "− Dividends", "− Ad Spend", "− Tax", "End Cash"],
            y=[ct_current_cash, ct_next_q_operating_cash,
               -ct_next_q_capex, -ct_next_q_div, -ct_next_q_ad, -ct_tax, 0],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": "#2d6a2e"}},
            decreasing={"marker": {"color": "#b22222"}},
            totals={"marker": {"color": "#1a3c5e"}},
        ))
        fig_cash.update_layout(height=300, yaxis_title="Cash ($)", yaxis_tickformat="$,.0f",
                                title=dict(text="Next Quarter Cash Flow Waterfall",
                                             x=0.5, xanchor="center", y=0.97, yanchor="top"),
                                margin=dict(l=0, r=0, t=60, b=0))
        st.plotly_chart(fig_cash, use_container_width=True)

        if ct_next_q_ending_cash < 0:
            st.error(f"""
🔴 **CRITICAL: Emergency loan triggered at 40% APR.**
Gap: ${-ct_next_q_ending_cash:,.0f}. Actions:
1. Defer capex (${ct_next_q_capex:,}) to a later quarter
2. Defer dividends (${ct_next_q_div:,})
3. Cut ad spend (${ct_next_q_ad:,})
4. Issue bonds now (Excellent rate 10% APR << 40% emergency)
            """)
        elif ct_next_q_ending_cash < recommended_buffer:
            st.warning(f"""
⚠️ **Cash below recommended buffer.**
You'll survive this quarter but have no margin for surprises. Consider:
- Delay non-critical capex
- Build cash cushion before expansion
            """)
        else:
            st.success(f"✅ Cash position is healthy. Buffer of ${ct_next_q_ending_cash - recommended_buffer:,.0f} above recommended minimum.")

    # Going concern reminder
    st.info("""
**Going Concern Note (for Final Project valuation):**
Per Gleacher Tips: *"The business remains a going concern after the period of active play."*
Your firm's terminal value (post-day 1460) should be included in valuation. Use a **terminal value**
= next-year cash flow / (discount rate − growth rate) or a 4-year DCF + terminal multiple.
    """)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 0.6: 13 TRIAL WAR ROOM (Production Game / Gleacher Game)
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🏭 13 Trial War Room":
    st.markdown('<p class="big-header">13 Trial War Room</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Production Game (Gleacher Game) — April 13 | Oligopoly with 8 teams, Bass diffusion, Cobb-Douglas, 4-year simulation</p>',
                unsafe_allow_html=True)
    st.markdown("")

    # ══════════════════════════════════════════════════════════════════════════
    # CRITICAL WARNING BANNER (Class 2 updates)
    # ══════════════════════════════════════════════════════════════════════════
    st.error("""
**⚠ CRITICAL: This is NOT a monopoly. Per Class 2 lecture:**
- **DO NOT use monopoly pricing** `(WTP+MC)/2` — that was only for the Monopoly & Trading games
- Daily production is **NOT 40/day** — calculate from Cobb-Douglas for your factory
- Lead time is **NOT 3.5 days** — depends on throughput, batch size, and shipping mode
- Production cost is **NOT $100** — you pay for materials + shipping + overhead + handling + commission
- Starting cash is **$1,579,530** (not $4M) — firm has been operating at a loss for Year 1
- Only **5 markets** exist: MD-Heart, MD-Breast, MD-Estrogen, Law-Narcotic, Clinic-Fertility
- **No new factories until Tuesday** (per instructor)
""")

    # ══════════════════════════════════════════════════════════════════════════
    # GAME CONSTANTS (from Gleacher Game Case Brief)
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("Game Parameters")
    st.caption("Adjust to match your current scenario — all calculators below recalculate")

    pg_col1, pg_col2, pg_col3, pg_col4, pg_col5 = st.columns(5)
    with pg_col1:
        PG_STARTING_CASH = st.number_input("Starting Cash (Day 365) ($)", value=1579530, step=10000, key="pg_cash",
                                             help="Actual cash at end of Q4. Firm's contributed capital is $5M but retained earnings are -$342K after Year 1 losses, factory ($193K net), DC ($2.33M net).")
    with pg_col2:
        PG_SALES_COMMISSION = st.number_input("Sales Commission (%)", value=20.0, step=1.0, key="pg_comm")
    with pg_col3:
        PG_HANDLING_COST = st.number_input("Handling Cost ($/unit)", value=10, step=1, key="pg_handling")
    with pg_col4:
        PG_FACTORY_LAND = st.number_input("Factory Land ($)", value=100000, step=10000, key="pg_fland")
    with pg_col5:
        PG_DC_LAND = st.number_input("DC Land ($)", value=100000, step=10000, key="pg_dcland")

    pg2_col1, pg2_col2, pg2_col3, pg2_col4, pg2_col5 = st.columns(5)
    with pg2_col1:
        PG_DC_CAPITAL = st.number_input("DC Capital ($)", value=2500000, step=100000, key="pg_dccap")
    with pg2_col2:
        PG_DC_DAILY = st.number_input("DC Daily Cost ($)", value=2000, step=100, key="pg_dcdaily")
    with pg2_col3:
        PG_DC_BUILD = st.number_input("DC Build (days)", value=60, step=5, key="pg_dcbuild")
    with pg2_col4:
        PG_FACTORY_BUILD = st.number_input("Factory Build (days)", value=90, step=5, key="pg_fbuild")
    with pg2_col5:
        PG_DEPRECIATION_YRS = st.number_input("Depreciation (yrs)", value=15, step=1, key="pg_dep")

    # ── Key Insight Banner ────────────────────────────────────────────────────
    st.markdown(f"""
<div style="background:linear-gradient(135deg,#0e7c7b,#1a3c5e);color:white;
    border-radius:12px;padding:1.2rem 1.5rem;margin-bottom:1rem;">
<h4 style="color:#ffd700;margin:0 0 0.5rem 0;">Goal: Maximize Current Shareholder Value | 8-Team Oligopoly | Bass Diffusion | 4-year Horizon (day 365 → 1460)</h4>
<div style="display:flex;gap:2rem;flex-wrap:wrap;">
<div><span style="opacity:0.7;">Starting Cash (Day 365)</span><br><b style="font-size:1.3rem;">${PG_STARTING_CASH:,}</b><br><span style="font-size:0.75rem;opacity:0.6;">actual after Y1 losses</span></div>
<div><span style="opacity:0.7;">Sales Commission</span><br><b style="font-size:1.3rem;">{PG_SALES_COMMISSION:.0f}%</b><br><span style="font-size:0.75rem;opacity:0.6;">of retail price</span></div>
<div><span style="opacity:0.7;">Handling Cost</span><br><b style="font-size:1.3rem;">${PG_HANDLING_COST}/unit</b></div>
<div><span style="opacity:0.7;">Raw Materials Payable</span><br><b style="font-size:1.3rem;">30 days</b></div>
<div><span style="opacity:0.7;">Other Payables</span><br><b style="font-size:1.3rem;">15 days</b></div>
<div><span style="opacity:0.7;">Tax Rate</span><br><b style="font-size:1.3rem;">35%</b></div>
<div><span style="opacity:0.7;">Cost of Capital</span><br><b style="font-size:1.3rem;">15% APR</b></div>
<div><span style="opacity:0.7;">Emergency Loan</span><br><b style="font-size:1.3rem;color:#ff6b6b;">40% APR</b></div>
</div>
</div>
""", unsafe_allow_html=True)

    # Starting Financial Snapshot
    with st.expander("**Starting Financial Snapshot (Day 364 — all 8 teams identical)**", expanded=False):
        sp_col1, sp_col2, sp_col3 = st.columns(3)
        with sp_col1:
            st.markdown("**Income Statement Q4**")
            st.markdown("""
| Line | Amount |
|---|---|
| Sales revenue | $682,500 |
| COGS | $(417,743) |
| **Gross profit** | **$264,757** |
| Selling expense (commission) | $(146,250) |
| DC operating expense | $(182,000) |
| DC depreciation | $(41,667) |
| **Operating income** | **$(105,160)** |
| Interest revenue | $11,913 |
| Tax benefit | $32,636 |
| **Net income** | **$(60,611)** |
""")
        with sp_col2:
            st.markdown("**Balance Sheet**")
            st.markdown("""
| Line | Amount |
|---|---|
| Cash | $1,579,530 |
| Accounts receivable | $240,100 |
| WIP inventory | $33,157 |
| Finished goods | $112,683 |
| Tax benefit | $184,365 |
| Land | $200,000 |
| Net PP&E (factory) | $93,333 |
| Net PP&E (DC) | $2,333,333 |
| **Total assets** | **$4,776,502** |
| Raw materials payable | $(20,000) |
| Other payables | $(98,900) |
| Contributed capital | $5,000,000 |
| Retained earnings | $(342,398) |
""")
        with sp_col3:
            st.markdown("**Key Metrics**")
            st.markdown("""
| Metric | Value |
|---|---|
| Current ratio | 18.08× |
| Non-cash current ratio | 4.80× |
| Quick ratio | 15.30× |
| Working capital | $2,030,936 |
| Non-cash WC | $451,406 |
| Return to Investors | $1,579,530 |

**Return to Investors Formula:**
Cash Balance − Unsecured Debt + Paid Dividends + Subsequent Returns
""")
        st.info(
            "**Starting position:** Q4 operating loss of $105K, Y1 retained earnings −$342K. "
            "Total selling expense ($328K) exceeds gross profit contribution — this business is "
            "losing money at current scale. Your job: turn it around over the next ~3 years."
        )

    # Game Timeline
    with st.expander("**Game Timeline & Play Schedule**", expanded=False):
        st.markdown("""
| Day Range | Duration | What Happens |
|---|---|---|
| **Day 365 (Q4 end)** | Start | Team takes over from engineer |
| **Day 365 → 546 (Q6 end)** | 1 hour in class | Initial moves |
| **Day 546 → 910 (Q10 end)** | 2 hours tonight | Most strategy executed here |
| **Day 910 → 1092** | Autonomous | Game continues without you — see what would happen |
| **Speed** | 3 game days per minute | |

**Teams (8):** B612, Dune, Globex, Gotham, **Panem (us)**, Vulcan, Westeros, Zion

**Markets (only 5):** MD-Heart, MD-Breast, MD-Estrogen, Law-Narcotic, Clinic-Fertility

**Restrictions:** No new factories until Tuesday. Catherine and Kathleen will not trade with us tonight.
        """)

    # Y1 Quarterly Progression (shows business is turning corner)
    with st.expander("**Year 1 Quarterly Progression — the Bass curve is kicking in**", expanded=False):
        st.markdown("""
| Quarter | Sales Revenue | COGS | Gross Profit | Op Income | Net Loss | Cash EOP |
|---|---|---|---|---|---|---|
| **Q1** | $303,800 | $185,949 | $117,851 | $(170,916) | $(101,057) | $1,909,244 |
| **Q2** | $456,400 | $279,352 | $177,048 | $(158,788) | $(94,402) | $1,723,448 |
| **Q3** | $543,200 | $332,481 | $210,719 | $(145,279) | $(86,327) | $1,613,017 |
| **Q4** | $682,500 | $417,743 | $264,757 | $(105,160) | $(60,611) | **$1,579,530** |

**Growth rate:** Revenue doubling from Q1 to Q4. Losses shrinking by ~40%.
The Bass curve is ramping — Y2 should be closer to breakeven.
Sales growth is Bass-driven (innovators → imitators), not from price changes.
Accounts receivable growing from $0 → $240K — customers pay after some DSO.
        """)

    # Product Development Workflow
    with st.expander("**Building New Products — Mandatory Workflow (per Class 2)**", expanded=False):
        st.markdown("""
### 6-Step Workflow

1. **Research** — Study Market Research PDF, decide which market to target
2. **Design** — Pick features in Product Design panel, save the design
3. **🔴 Run a Focus Group** — $20K, 10 participants, 7 days. **CRITICAL: validates WTP before committing to development cost**
4. **Start Development** — Incurs development cost and design days
5. **Determine Price** — Based on focus group + competitor pricing
6. **Set Up Shipping Agreements** — Factory → DC (internal), or Factory → other teams' DCs (external trade)

### Today's 5 Markets
| Market | WTP Range | Size/Region | Bass p | Notes |
|---|---|---|---|---|
| **MD Heart** | $600-865 | 20-40K | 0.0002 | ⚠ **Heart View already covers this — do NOT build redundant product** |
| **Clinical Fertility (LH/FSH)** | $230-400 | 40-60K | 0.00025 | Largest market, lowest WTP |
| **MD Fertility (Estrogen)** | $575-965 | 10-20K | 0.0002 | Mid WTP, mid size |
| **MD Cancer (Breast)** | $900-1,600 | 10-20K | 0.0002 | Premium WTP |
| **Law (Narcotic)** | $1,100-1,600 | 5-15K | 0.00025 | Highest WTP, smallest size, **needs GPS + cellular** |

**Strategic implication:** Pick 1-2 markets to pursue beyond MD Heart. Premium markets (Cancer, Narcotic) offer higher margins; volume markets (Clinical Fertility) offer scale.
        """)

    # Bench Factory Reference Table (instructor-provided check figures)
    with st.expander("**Bench Factory Reference Table — Check Figures from Class 2**", expanded=False):
        st.markdown("Per the lecture, these are the expected throughput and overhead values for the Bench factory. Use to verify your Assignment 3 spreadsheet calculations.")

        st.markdown("**Daily Throughput (units/day, including setup time)**")
        st.markdown("""
| K ↓ / L → | $500 | $1,000 | $2,500 | $3,000 | $4,000 | $8,000 | $16,000 |
|---|---|---|---|---|---|---|---|
| **$100,000** | 2.310 | 4.461 | 9.044 | 10.552 | 13.455 | 24.123 | 43.065 |
| **$200,000** | 2.476 | 4.459 | 9.690 | 11.305 | 14.414 | 25.832 | 46.085 |
| **$500,000** | 2.713 | 4.886 | 10.615 | 12.383 | 15.786 | 28.276 | 50.396 |
| **$1,000,000** | 2.908 | 5.235 | 11.372 | 13.266 | 16.910 | 30.275 | 53.915 |
| **$3,000,000** | 3.245 | 5.841 | 12.685 | 14.795 | 18.855 | 33.731 | 59.998 |
| **$5,000,000** | 3.415 | 6.147 | 13.345 | 15.564 | 19.833 | 35.467 | 63.033 |
        """)

        st.markdown("**Manufacturing Overhead per Unit ($)**")
        st.markdown("""
| K ↓ / L → | $500 | $1,000 | $2,500 | $3,000 | $4,000 | $8,000 | $16,000 |
|---|---|---|---|---|---|---|---|
| **$100,000** | $224.33 | $244.74 | $278.45 | $286.04 | $298.64 | $332.39 | $371.96 |
| **$200,000** | $216.73 | $232.50 | $261.78 | $268.61 | $280.05 | $311.11 | $347.98 |
| **$500,000** | $218.02 | $223.43 | $244.15 | $249.66 | $259.18 | $286.17 | $319.31 |
| **$1,000,000** | $234.93 | $226.00 | $235.94 | $239.95 | $247.38 | $270.30 | $300.16 |
| **$3,000,000** | $323.41 | $265.25 | $240.41 | $239.91 | $241.29 | $253.46 | $275.88 |
| **$5,000,000** | $414.60 | $311.68 | $255.96 | $251.58 | $247.85 | $251.38 | $268.36 |

**Key insight:** Lowest overhead/unit on Bench factory is at K=$500K-$1M, L=$2,500-$4,000 — around $220-245/unit. Going too high on K without matching labor wastes depreciation. Going too high on L alone hits diminishing returns.
        """)

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 1: BASS DIFFUSION MODEL CALCULATOR
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("1. Bass Diffusion Demand Simulator")
    st.caption("Q_t = [p + q × (A/M)] × (M - A) — innovation + imitation drives arrivals")

    bass_col1, bass_col2 = st.columns([1, 2])
    with bass_col1:
        bass_M = st.number_input("Market Size (M)", value=30000, step=1000, key="bass_m",
                                   help="MD-Heart: 20K-40K per region (Heart View's market). Narcotic: 5K-15K. Fertility: 40K-60K. Breast/Estrogen: 10K-20K.")
        bass_p = st.number_input("Innovation Coefficient (p)", value=0.0002, step=0.00005,
                                  format="%.5f", key="bass_p",
                                  help="Typical: 0.0002-0.00025. Higher = faster early adoption.")
        bass_q = st.number_input("Imitation Coefficient (q)", value=0.0035, step=0.0005,
                                  format="%.4f", key="bass_q",
                                  help="Typical: 0.0025-0.004. Higher = faster word-of-mouth.")
        bass_p_inc = st.number_input("Adv. incremental p per $500/day", value=0.0002,
                                      step=0.00005, format="%.5f", key="bass_p_inc",
                                      help="Market-specific. From market research tables. Typically equals base p for medical/clinical markets.")
        bass_adv = st.number_input("Advertising $/day", value=0, step=100, key="bass_adv",
                                    help=f"Adds incremental_p = (adv/$500) × {bass_p_inc:.5f} to base p")
        bass_price = st.number_input("Retail Price ($)", value=600, step=25, key="bass_price",
                                      help="Actual selling price — affects P(buy) for arrivals")
        bass_max_wtp = st.number_input("Max WTP ($)", value=965, step=50, key="bass_max_wtp",
                                         help="Upper bound of WTP range. Used to calculate P(buy).")
        bass_days = st.number_input("Simulate Days", value=1095, step=30, key="bass_days",
                                     help="Day 365 start → day 1460 end = 1095 days remaining for us. (Full 4-year horizon = 1460 days).")

        # Peak sales calculation (theoretical, assuming 100% conversion)
        peak_time = (1 / (bass_p + bass_q)) * np.log(bass_q / bass_p) if bass_p > 0 and bass_q > bass_p else 0
        peak_sales = bass_M * ((bass_p + bass_q) ** 2) / (4 * bass_q) if bass_q > 0 else 0

        st.markdown("---")
        st.metric("Time to Peak Arrivals", f"{peak_time:.0f} days ({peak_time/365:.1f} yrs)")
        st.metric("Peak Daily Arrivals (theoretical)", f"{peak_sales:.1f} /day")

        # P(buy) at given price
        p_buy = max(0, min(1, (bass_max_wtp - bass_price) / bass_max_wtp)) if bass_max_wtp > 0 else 0
        st.metric("P(buy) at this price", f"{p_buy:.1%}")

    with bass_col2:
        # Simulate Bass model with return-to-pool dynamics
        adv_p_boost = (bass_adv / 500) * bass_p_inc if bass_adv > 0 else 0
        effective_p = bass_p + adv_p_boost

        # Key insight: customers who don't buy RETURN to pool (per Case Brief).
        # So only PURCHASES deplete the unserved pool M - A, not arrivals.
        A = 0  # Cumulative purchases (not arrivals)
        days = np.arange(int(bass_days))
        arrivals = []  # Daily arrivals (visits to DC)
        purchases = []  # Daily actual sales
        cumulative = []  # Cumulative purchases
        for t in days:
            if bass_M - A > 0:
                q_t = (effective_p + bass_q * A / bass_M) * (bass_M - A)  # arrivals
            else:
                q_t = 0
            buys = q_t * p_buy  # Only those with positive surplus purchase
            arrivals.append(q_t)
            purchases.append(buys)
            A += buys  # Only purchases deplete pool (non-buyers return)
            cumulative.append(A)

        fig_bass = go.Figure()
        fig_bass.add_trace(go.Scatter(x=days, y=arrivals, name="Daily Arrivals",
                                       line=dict(color="#b8860b", width=1.5, dash="dot")))
        fig_bass.add_trace(go.Scatter(x=days, y=purchases, name=f"Daily Purchases (P(buy)={p_buy:.0%})",
                                       line=dict(color="#800000", width=2.5)))
        fig_bass.add_trace(go.Scatter(x=days, y=cumulative, name="Cumulative Purchases",
                                       line=dict(color="#1a3c5e", width=2, dash="dash"),
                                       yaxis="y2"))
        fig_bass.update_layout(
            height=400, xaxis_title="Days",
            yaxis=dict(title="Daily Count (arrivals/purchases)", tickfont=dict(color="#800000")),
            yaxis2=dict(title="Cumulative Purchases", overlaying="y", side="right",
                        tickfont=dict(color="#1a3c5e")),
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_bass, use_container_width=True)

        if bass_adv > 0:
            st.info(f"Advertising boost: base p = {bass_p:.5f} → effective p = {effective_p:.5f} (+{adv_p_boost/bass_p*100:.1f}%)")

        # Summary stats
        total_purchases = int(sum(purchases))
        total_arrivals = int(sum(arrivals))
        st.caption(f"Over {int(bass_days)} days: {total_arrivals:,} arrivals → {total_purchases:,} purchases ({total_purchases/total_arrivals*100:.1f}% conversion). "
                    f"Non-buyers return to pool per Case Brief — so only purchases deplete M−A.")

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2: COBB-DOUGLAS PRODUCTION CALCULATOR (custom α,β + Little's Law)
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("2. Cobb-Douglas Production + Little's Law Calculator")
    st.caption("Custom α, β, K, L inputs → yearly & daily throughput → cycle time → inventory via Little's Law")

    PROD_TECH = {
        "Custom": {"A": 0.009, "alpha": 0.10, "beta": 0.85, "setup": 0.05, "min_K": 0,
                    "desc": "Set your own A, α, β, setup time."},
        "Benches": {"A": 0.009, "alpha": 0.10, "beta": 0.85, "setup": 0.05, "min_K": 0,
                    "desc": "General-purpose, skilled workers. Labor-intensive."},
        "Production Line": {"A": 0.010, "alpha": 0.30, "beta": 0.75, "setup": 0.50, "min_K": 500000,
                            "desc": "Specialized, unskilled workers. Balanced."},
        "Automated Cell": {"A": 0.020, "alpha": 0.80, "beta": 0.30, "setup": 1.00, "min_K": 3000000,
                            "desc": "Capital-intensive automation. Robots over humans."},
    }

    # ── Preset selector + custom overrides ───────────────────────────────────
    preset_col1, preset_col2 = st.columns([1, 3])
    with preset_col1:
        cd_preset = st.selectbox("Technology Preset", list(PROD_TECH.keys()),
                                   index=1, key="cd_preset")
    with preset_col2:
        st.caption(f"**{PROD_TECH[cd_preset]['desc']}** — You can override A, α, β, setup below to model a custom technology.")

    preset = PROD_TECH[cd_preset]

    # ── Custom parameter inputs ──────────────────────────────────────────────
    st.markdown("**Cobb-Douglas Parameters** (Y = A × K^α × L^β)")
    cd_p_col1, cd_p_col2, cd_p_col3, cd_p_col4 = st.columns(4)
    with cd_p_col1:
        cd_A = st.number_input("A (total factor productivity)",
                                 value=float(preset["A"]), step=0.001, format="%.4f", key="cd_A")
    with cd_p_col2:
        cd_alpha = st.number_input("α (capital exponent)",
                                     value=float(preset["alpha"]), step=0.05, format="%.2f", key="cd_alpha",
                                     help="Sensitivity of output to capital. 0=none, 1=linear.")
    with cd_p_col3:
        cd_beta = st.number_input("β (labor exponent)",
                                    value=float(preset["beta"]), step=0.05, format="%.2f", key="cd_beta",
                                    help="Sensitivity of output to labor. 0=none, 1=linear.")
    with cd_p_col4:
        cd_setup = st.number_input("Setup time (days)",
                                     value=float(preset["setup"]), step=0.05, format="%.2f", key="cd_setup")

    # ── Inputs ────────────────────────────────────────────────────────────────
    st.markdown("**Factory Inputs**")
    cd_col1, cd_col2, cd_col3, cd_col4 = st.columns(4)
    with cd_col1:
        cd_K = st.number_input("K — Capital ($)", value=100000, step=10000, key="cd_K")
    with cd_col2:
        cd_l = st.number_input("l — Daily Labor ($/day)", value=2500, step=100, key="cd_l")
    with cd_col3:
        cd_batch = st.number_input("Batch Size (units)", value=100, step=10, key="cd_batch")
    with cd_col4:
        cd_days_per_year = st.number_input("Days per Year", value=364, step=1, key="cd_dpy")

    # ── Core Cobb-Douglas calculations ───────────────────────────────────────
    cd_L_yearly = cd_l * cd_days_per_year  # L = l × 364 (yearly labor)
    cd_Y_yearly = cd_A * (cd_K ** cd_alpha) * (cd_L_yearly ** cd_beta)  # yearly output
    cd_lambda_raw = cd_Y_yearly / cd_days_per_year  # daily throughput before setup

    # Little's Law + setup time
    cd_batch_time = cd_batch / cd_lambda_raw if cd_lambda_raw > 0 else float("inf")
    cd_CT = cd_batch_time + cd_setup  # cycle time = batch time + setup
    cd_lambda_eff = cd_batch / cd_CT if cd_CT > 0 else 0  # daily throughput with setup
    cd_WIP = cd_lambda_eff * cd_CT  # Little's Law: inventory = throughput × cycle time

    # Returns to scale
    cd_alpha_plus_beta = cd_alpha + cd_beta

    # ── Results display ──────────────────────────────────────────────────────
    st.markdown("---")
    result_col1, result_col2, result_col3 = st.columns(3)

    with result_col1:
        st.markdown("### 📐 Cobb-Douglas Output")
        st.metric("Yearly Production (Y)", f"{cd_Y_yearly:,.0f} units/yr",
                   help="Y = A × K^α × L^β where L = daily labor × 364")
        st.metric("Daily Throughput Raw (λ)", f"{cd_lambda_raw:.2f} units/day",
                   help="λ = Y / 364 — before setup time adjustment")
        st.caption(f"Formula: Y = {cd_A:.4f} × {cd_K:,}^{cd_alpha:.2f} × ({cd_l:,}×{cd_days_per_year})^{cd_beta:.2f}")
        st.caption(f"= {cd_A:.4f} × {cd_K**cd_alpha:,.2f} × {cd_L_yearly**cd_beta:,.0f}")
        st.caption(f"= {cd_Y_yearly:,.0f} units/year")

        # Returns to scale interpretation
        if cd_alpha_plus_beta > 1.02:
            rts_color = "#2d6a2e"
            rts_label = "Increasing returns"
        elif cd_alpha_plus_beta < 0.98:
            rts_color = "#b22222"
            rts_label = "Decreasing returns"
        else:
            rts_color = "#b8860b"
            rts_label = "Constant returns"
        st.markdown(f'<div style="color:{rts_color};font-weight:bold;">α + β = {cd_alpha_plus_beta:.2f} → {rts_label}</div>',
                     unsafe_allow_html=True)

    with result_col2:
        st.markdown("### ⏱️ Cycle Time")
        st.metric("Batch Production Time", f"{cd_batch_time:.3f} days",
                   help=f"batch / λ_raw = {cd_batch} / {cd_lambda_raw:.2f}")
        st.metric("Setup Time", f"{cd_setup:.3f} days")
        st.metric("Cycle Time (CT)", f"{cd_CT:.3f} days",
                   help="CT = batch time + setup time")
        st.caption("The total time from when one batch starts production until the next batch can start.")

    with result_col3:
        st.markdown("### 📦 Little's Law: INV = λ × CT")
        st.metric("Daily Throughput w/ Setup (λ)", f"{cd_lambda_eff:.2f} units/day",
                   help="λ_effective = batch / CT")
        st.metric("Average WIP Inventory", f"{cd_WIP:.1f} units",
                   help="INV = λ × CT — average units in the production system")
        # Compare to batch size
        if cd_WIP < cd_batch:
            st.caption(f"WIP ({cd_WIP:.1f}) < batch ({cd_batch}) → factory doesn't accumulate backlog")
        else:
            st.caption(f"WIP ({cd_WIP:.1f}) ≈ batch ({cd_batch}) → factory runs continuously")

    # ── Step-by-step formula trace ───────────────────────────────────────────
    with st.expander("**Step-by-step formula trace**", expanded=False):
        st.markdown(f"""
### Cobb-Douglas Production Function

**Yearly production** (the textbook form):
$$Y = A \\cdot K^{{\\alpha}} \\cdot L^{{\\beta}}$$

where:
- Y = yearly output (units/year)
- A = {cd_A:.4f} (total factor productivity)
- K = ${cd_K:,} (capital invested in factory, excludes land)
- α = {cd_alpha:.2f} (capital exponent)
- L = ${cd_L_yearly:,} (yearly labor = daily labor × {cd_days_per_year})
- β = {cd_beta:.2f} (labor exponent)

**Calculation:**
Y = {cd_A:.4f} × ({cd_K:,})^{cd_alpha:.2f} × ({cd_L_yearly:,})^{cd_beta:.2f}
Y = {cd_A:.4f} × {cd_K**cd_alpha:,.2f} × {cd_L_yearly**cd_beta:,.2f}
Y = **{cd_Y_yearly:,.0f} units/year**

**Daily throughput (raw, without setup):**
$$\\lambda_{{raw}} = \\frac{{Y}}{{364}} = \\frac{{{cd_Y_yearly:,.0f}}}{{{cd_days_per_year}}} = {cd_lambda_raw:.4f} \\text{{ units/day}}$$

---

### Little's Law: INV = λ × CT

**Cycle Time** (time to produce one batch + setup):
$$CT = \\frac{{batch}}{{\\lambda_{{raw}}}} + setup = \\frac{{{cd_batch}}}{{{cd_lambda_raw:.2f}}} + {cd_setup:.2f} = {cd_CT:.3f} \\text{{ days}}$$

**Effective daily throughput** (accounting for setup):
$$\\lambda_{{eff}} = \\frac{{batch}}{{CT}} = \\frac{{{cd_batch}}}{{{cd_CT:.3f}}} = {cd_lambda_eff:.2f} \\text{{ units/day}}$$

**Average WIP Inventory** (Little's Law):
$$INV = \\lambda_{{eff}} \\times CT = {cd_lambda_eff:.2f} \\times {cd_CT:.3f} = {cd_WIP:.1f} \\text{{ units}}$$

*Note: INV ≈ batch size by construction when factory runs continuously with this model.*
        """)

    # ── Sensitivity: Daily throughput and WIP over K and L ───────────────────
    sens_col1, sens_col2 = st.columns(2)
    with sens_col1:
        # Throughput vs Labor
        labor_range = np.linspace(max(500, cd_l * 0.2), cd_l * 4, 30)
        thp_vs_l = [(cd_A * (cd_K ** cd_alpha) * ((l * cd_days_per_year) ** cd_beta) / cd_days_per_year)
                     for l in labor_range]
        # Apply setup correction
        eff_thp_vs_l = [(cd_batch / (cd_batch / t + cd_setup)) if t > 0 else 0 for t in thp_vs_l]

        fig_l = go.Figure()
        fig_l.add_trace(go.Scatter(x=labor_range, y=thp_vs_l, name="Raw λ",
                                     line=dict(color="#1a3c5e", width=2, dash="dash")))
        fig_l.add_trace(go.Scatter(x=labor_range, y=eff_thp_vs_l, name="Effective λ (w/ setup)",
                                     line=dict(color="#800000", width=2.5)))
        fig_l.add_vline(x=cd_l, line_dash="dot", line_color="gray",
                         annotation_text=f"Current L=${cd_l}")
        fig_l.update_layout(height=300, xaxis_title="Daily Labor ($)",
                             yaxis_title="Daily Throughput (units)",
                             title="Throughput vs Labor",
                             margin=dict(l=0, r=0, t=40, b=0),
                             legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig_l, use_container_width=True)

    with sens_col2:
        # Throughput vs Capital
        k_range = np.linspace(max(50000, cd_K * 0.2), cd_K * 4, 30)
        thp_vs_k = [(cd_A * (k ** cd_alpha) * ((cd_l * cd_days_per_year) ** cd_beta) / cd_days_per_year)
                     for k in k_range]
        eff_thp_vs_k = [(cd_batch / (cd_batch / t + cd_setup)) if t > 0 else 0 for t in thp_vs_k]

        fig_k = go.Figure()
        fig_k.add_trace(go.Scatter(x=k_range, y=thp_vs_k, name="Raw λ",
                                     line=dict(color="#1a3c5e", width=2, dash="dash")))
        fig_k.add_trace(go.Scatter(x=k_range, y=eff_thp_vs_k, name="Effective λ (w/ setup)",
                                     line=dict(color="#800000", width=2.5)))
        fig_k.add_vline(x=cd_K, line_dash="dot", line_color="gray",
                         annotation_text=f"Current K=${cd_K:,}")
        fig_k.update_layout(height=300, xaxis_title="Capital ($)",
                             yaxis_title="Daily Throughput (units)",
                             title="Throughput vs Capital",
                             margin=dict(l=0, r=0, t=40, b=0),
                             legend=dict(orientation="h", yanchor="bottom", y=1.02),
                             xaxis_tickformat="$,.0f")
        st.plotly_chart(fig_k, use_container_width=True)

    st.info(f"""
**Interpretation for current inputs:**
- **α + β = {cd_alpha_plus_beta:.2f}** → {"Increasing" if cd_alpha_plus_beta > 1.02 else ("Decreasing" if cd_alpha_plus_beta < 0.98 else "Constant")} returns to scale
- **α/β = {cd_alpha/cd_beta if cd_beta > 0 else 0:.2f}** → {"Capital-dominant" if cd_alpha > cd_beta else "Labor-dominant"} technology. Invest more in {"capital" if cd_alpha > cd_beta else "labor"}.
- **Setup drag:** {cd_setup / cd_CT * 100:.1f}% of cycle time is setup. To reduce drag, increase batch size.
- **WIP = {cd_WIP:.1f} units** — this is what you'll see as "Inventory in Process" in your HQ panel.
    """)

    # ── Manufacturing Overhead per Unit (feeds into Contribution Margin) ─────
    # Total daily factory cost = daily labor + daily depreciation
    # Assumes 15-year SL depreciation, land not included in K (K is equipment only)
    cd_daily_dep = cd_K / PG_DEPRECIATION_YRS / cd_days_per_year
    cd_daily_factory_cost = cd_l + cd_daily_dep
    cd_mfg_overhead_per_unit = cd_daily_factory_cost / cd_lambda_eff if cd_lambda_eff > 0 else 0

    st.markdown("")
    st.markdown("### 💰 Contribution Margin Table")
    st.caption("Set a retail price → see contribution margin after all variable costs (uses the manufacturing overhead calculated above)")

    # ── CM inputs ─────────────────────────────────────────────────────────────
    cm_col1, cm_col2, cm_col3, cm_col4, cm_col5 = st.columns(5)
    with cm_col1:
        cm_price = st.number_input("Retail Price ($/unit)", value=700, step=25, key="cm_price")
    with cm_col2:
        cm_materials = st.number_input("Materials Cost ($/unit)", value=100, step=10, key="cm_materials",
                                          help="Sum of materials costs from product features")
    with cm_col3:
        cm_shipping = st.number_input("Shipping Cost ($/unit)", value=20, step=5, key="cm_shipping",
                                        help="Production Game rates: Mail in region $20/u (1 day) | Mail between regions $40/u (3 days) | Container in region $5/u min 1000u (7 days) | Container between regions $10/u min 1000u (21 days). Default $20 = mail in-region.")
    with cm_col4:
        cm_commission_pct = st.number_input("Commission (%)", value=PG_SALES_COMMISSION,
                                              step=1.0, key="cm_comm",
                                              help="Per Class 2: 20% of revenue")
    with cm_col5:
        cm_handling = st.number_input("Handling ($/unit)", value=PG_HANDLING_COST,
                                         step=1, key="cm_handling")

    # ── Calculations ──────────────────────────────────────────────────────────
    cm_commission_per_unit = cm_price * (cm_commission_pct / 100)
    cm_total_var_cost = (cd_mfg_overhead_per_unit + cm_materials + cm_shipping
                          + cm_handling + cm_commission_per_unit)
    cm_before_tax = cm_price - cm_total_var_cost
    cm_tax = max(0, cm_before_tax) * 0.35
    cm_after_tax = cm_before_tax - cm_tax

    # Per-day aggregates (using effective throughput)
    cm_day_revenue = cm_price * cd_lambda_eff
    cm_day_mfg = cd_daily_factory_cost  # already a daily value
    cm_day_materials = cm_materials * cd_lambda_eff
    cm_day_shipping = cm_shipping * cd_lambda_eff
    cm_day_handling = cm_handling * cd_lambda_eff
    cm_day_commission = cm_commission_per_unit * cd_lambda_eff
    cm_day_before_tax = cm_day_revenue - cm_day_mfg - cm_day_materials - cm_day_shipping - cm_day_handling - cm_day_commission
    cm_day_tax = max(0, cm_day_before_tax) * 0.35
    cm_day_after_tax = cm_day_before_tax - cm_day_tax

    # ── Table display ────────────────────────────────────────────────────────
    cm_table_col, cm_chart_col = st.columns([1, 1])

    with cm_table_col:
        cm_pct_of_rev = lambda v: f"{v/cm_price*100:.1f}%" if cm_price > 0 else "—"

        color_pre = "#2d6a2e" if cm_before_tax > 0 else "#b22222"

        st.markdown(f"""
<div style="border:1px solid rgba(128,128,128,0.3); border-radius:8px; padding:1rem;">
<table style="width:100%; border-collapse:collapse;">
<tr style="border-bottom:2px solid rgba(128,128,128,0.5);">
<th style="text-align:left;">Line</th>
<th style="text-align:right;">Per Unit</th>
<th style="text-align:right;">Per Day</th>
<th style="text-align:right;">% Rev</th>
</tr>
<tr>
<td>Revenue</td>
<td style="text-align:right;"><b>${cm_price:,.2f}</b></td>
<td style="text-align:right;"><b>${cm_day_revenue:,.2f}</b></td>
<td style="text-align:right;">100.0%</td>
</tr>
<tr>
<td>(−) Manufacturing Overhead</td>
<td style="text-align:right;color:#b22222;">$({cd_mfg_overhead_per_unit:,.2f})</td>
<td style="text-align:right;color:#b22222;">$({cm_day_mfg:,.2f})</td>
<td style="text-align:right;">{cm_pct_of_rev(cd_mfg_overhead_per_unit)}</td>
</tr>
<tr>
<td>(−) Materials (Marginal Cost)</td>
<td style="text-align:right;color:#b22222;">$({cm_materials:,.2f})</td>
<td style="text-align:right;color:#b22222;">$({cm_day_materials:,.2f})</td>
<td style="text-align:right;">{cm_pct_of_rev(cm_materials)}</td>
</tr>
<tr>
<td>(−) Handling</td>
<td style="text-align:right;color:#b22222;">$({cm_handling:,.2f})</td>
<td style="text-align:right;color:#b22222;">$({cm_day_handling:,.2f})</td>
<td style="text-align:right;">{cm_pct_of_rev(cm_handling)}</td>
</tr>
<tr>
<td>(−) Commission ({cm_commission_pct:.0f}%)</td>
<td style="text-align:right;color:#b22222;">$({cm_commission_per_unit:,.2f})</td>
<td style="text-align:right;color:#b22222;">$({cm_day_commission:,.2f})</td>
<td style="text-align:right;">{cm_pct_of_rev(cm_commission_per_unit)}</td>
</tr>
<tr style="border-bottom:2px solid rgba(128,128,128,0.5);">
<td>(−) Shipping</td>
<td style="text-align:right;color:#b22222;">$({cm_shipping:,.2f})</td>
<td style="text-align:right;color:#b22222;">$({cm_day_shipping:,.2f})</td>
<td style="text-align:right;">{cm_pct_of_rev(cm_shipping)}</td>
</tr>
<tr style="background:rgba({'45,106,46' if cm_before_tax > 0 else '178,34,34'},0.2);">
<td><b>= Contribution Margin (before tax)</b></td>
<td style="text-align:right;color:{color_pre};font-size:1.15em;"><b>${cm_before_tax:,.2f}</b></td>
<td style="text-align:right;color:{color_pre};font-size:1.15em;"><b>${cm_day_before_tax:,.2f}</b></td>
<td style="text-align:right;color:{color_pre};"><b>{cm_pct_of_rev(cm_before_tax)}</b></td>
</tr>
</table>
</div>
""", unsafe_allow_html=True)
        st.caption("Focus is on CM before tax. Tax (35%) applies at entity level on aggregate income, not on CM per unit — it's not a decision-relevant cost when comparing pricing/product options.")

    with cm_chart_col:
        # Waterfall chart showing the margin decomposition (pre-tax focus)
        fig_cm = go.Figure(go.Waterfall(
            name="Per Unit",
            orientation="v",
            measure=["absolute", "relative", "relative", "relative", "relative", "relative", "total"],
            x=["Revenue", "Mfg OH", "Materials", "Handling", "Commission", "Shipping", "CM"],
            y=[cm_price, -cd_mfg_overhead_per_unit, -cm_materials, -cm_handling,
               -cm_commission_per_unit, -cm_shipping, 0],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": "#2d6a2e"}},
            decreasing={"marker": {"color": "#b22222"}},
            totals={"marker": {"color": "#1a3c5e"}},
        ))
        fig_cm.update_layout(height=400, yaxis_title="$ per unit",
                              yaxis_tickformat="$,.0f",
                              margin=dict(l=0, r=0, t=30, b=0),
                              title=f"Waterfall: ${cm_price} price → ${cm_before_tax:,.2f} CM before tax")
        st.plotly_chart(fig_cm, use_container_width=True)

    # Break-even analysis (all pre-tax)
    st.markdown("---")
    be_col1, be_col2, be_col3 = st.columns(3)
    with be_col1:
        # Break-even price given current costs (where CM before tax = 0)
        be_fixed_per_unit = cd_mfg_overhead_per_unit + cm_materials + cm_shipping + cm_handling
        be_price = be_fixed_per_unit / (1 - cm_commission_pct / 100) if cm_commission_pct < 100 else 0
        st.metric("Break-even Price", f"${be_price:,.2f}",
                   help="Minimum price where CM before tax = $0")

    with be_col2:
        st.metric("Unit CM (before tax)", f"${cm_before_tax:,.2f}",
                   delta=f"{cm_before_tax/cm_price*100:.1f}% of price" if cm_price > 0 else "—",
                   delta_color="normal" if cm_before_tax > 0 else "inverse")

    with be_col3:
        st.metric("Daily CM (before tax)", f"${cm_day_before_tax:,.2f}",
                   help=f"At λ_eff = {cd_lambda_eff:.2f} units/day")

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 3: MARKET SEGMENT SELECTOR (from D2 market research)
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("3. Market Segment Analyzer")
    st.caption("⚠ Per Class 2: This is OLIGOPOLY pricing, not monopoly. Optimal price formula shown assumes you're alone — actual price depends on competitor moves. Use as starting reference only.")

    # Only 5 markets exist in the Production Game per Class 2 slides
    MARKETS = {
        "MD-Heart (Temporal)": {"wtp_low": 600, "wtp_high": 865, "size_low": 20000, "size_high": 40000,
                                 "p": 0.0002, "q": 0.0035, "dso": 30,
                                 "feature": "Heartbeat Temporal (Heart View's base)",
                                 "dealbreaker": "GPS significantly affects WTP"},
        "MD-Breast (Cancer)": {"wtp_low": 900, "wtp_high": 1600, "size_low": 10000, "size_high": 20000,
                                "p": 0.0002, "q": 0.0035, "dso": 30,
                                "feature": "Cancer Breast panel", "dealbreaker": "None"},
        "MD-Estrogen (Fertility)": {"wtp_low": 575, "wtp_high": 965, "size_low": 10000, "size_high": 20000,
                                     "p": 0.0002, "q": 0.0035, "dso": 30,
                                     "feature": "Hormone Estrogen", "dealbreaker": "None (slight wrist preference)"},
        "Law-Narcotic": {"wtp_low": 1100, "wtp_high": 1600, "size_low": 5000, "size_high": 15000,
                          "p": 0.00025, "q": 0.0025, "dso": 90,
                          "feature": "Toxicology Narcotic",
                          "dealbreaker": "MUST have GPS + cellular (deal breaker)"},
        "Clinic-Fertility (LH/FSH)": {"wtp_low": 230, "wtp_high": 400, "size_low": 40000, "size_high": 60000,
                                       "p": 0.00025, "q": 0.004, "dso": 10,
                                       "feature": "Hormone LH/FSH",
                                       "dealbreaker": "Bulky battery packs (deal breaker)"},
    }

    # Global settings row
    mkt_config_col1, mkt_config_col2, mkt_config_col3 = st.columns([1, 1, 2])
    with mkt_config_col1:
        mkt_n = st.number_input("Markets to compare", min_value=2, max_value=4,
                                 value=2, step=1, key="mkt_n")
    with mkt_config_col2:
        mkt_mc_est = st.number_input("Your Materials Cost ($/unit)",
                                      value=100, step=10, key="mkt_mc",
                                      help="Sum of materials costs from product design features")
    with mkt_config_col3:
        st.caption("Side-by-side comparison. Each column is an independent market selector "
                   "with its own WTP and market size sliders.")

    comm_frac = PG_SALES_COMMISSION / 100
    effective_mc = mkt_mc_est + PG_HANDLING_COST
    effective_mc_commissioned = effective_mc / (1 - comm_frac)

    market_keys = list(MARKETS.keys())
    # Defaults: MD-Heart (our current), MD-Breast, Law-Narcotic, Clinic-Fertility
    default_indices = [0, 1, 3, 4]

    st.markdown("---")
    mkt_cols = st.columns(int(mkt_n))
    mkt_results = []  # collect for summary comparison
    for i, col in enumerate(mkt_cols):
        with col:
            default_idx = default_indices[i] if i < len(default_indices) else i
            mkt_sel = st.selectbox(f"Market {i+1}", market_keys, index=default_idx, key=f"mkt_sel_{i}")
            mkt = MARKETS[mkt_sel]

            # Header card with discipline info
            dealbreaker_color = "#b22222" if mkt['dealbreaker'] != "None" else "#2d6a2e"
            st.markdown(f"""
<div style="background:rgba(26,60,94,0.2);border-left:4px solid #1a3c5e;
    border-radius:6px;padding:0.6rem 0.8rem;margin-bottom:0.5rem;font-size:0.85rem;">
<b>{mkt_sel}</b><br>
<span style="opacity:0.7;">Core feature:</span> {mkt['feature']}<br>
<span style="opacity:0.7;">WTP range:</span> ${mkt['wtp_low']:,} – ${mkt['wtp_high']:,}<br>
<span style="opacity:0.7;">Market size / region:</span> {mkt['size_low']:,} – {mkt['size_high']:,}<br>
<span style="opacity:0.7;">Bass p:</span> {mkt['p']} &nbsp;|&nbsp; <span style="opacity:0.7;">q:</span> {mkt['q']}<br>
<span style="opacity:0.7;">DSO:</span> <b>{mkt['dso']} days</b><br>
<span style="opacity:0.7;">Deal breaker:</span> <b style="color:{dealbreaker_color};">{mkt['dealbreaker']}</b>
</div>
""", unsafe_allow_html=True)

            # ── Focus Group inputs: median AND max WTP ──────────────────────
            st.caption("**From focus group ($20K/10 participants/7 days): enter median and max WTP**")
            mkt_wtp_max = st.slider("Max WTP ($) — from focus group",
                                      int(mkt["wtp_low"]), int(mkt["wtp_high"] * 1.5),
                                      int(mkt["wtp_high"]), step=10, key=f"mkt_wtp_max_{i}",
                                      help="100th percentile of WTP distribution across customers")
            # Default median: geometric approach — 85% of max (typical pattern from focus group screenshots)
            default_median = int(mkt_wtp_max * 0.85)
            mkt_wtp_median = st.slider("Median WTP ($) — from focus group",
                                        int(mkt_wtp_max * 0.3), int(mkt_wtp_max),
                                        default_median, step=10, key=f"mkt_wtp_med_{i}",
                                        help="50th percentile. If uniform distribution, median = (min + max) / 2, so min = 2×median − max.")
            mkt_size_est = st.slider("Market Size Estimate",
                                      int(mkt["size_low"]), int(mkt["size_high"]),
                                      int((mkt["size_low"] + mkt["size_high"]) / 2), step=1000,
                                      key=f"mkt_size_{i}")

            # Implied min WTP under uniform [min, max] assumption
            mkt_wtp_min = max(0, 2 * mkt_wtp_median - mkt_wtp_max)

            # Optimal price calculation (uniform [min, max] + 20% commission)
            # Interior: P* = max/2 + var_fixed / (2 * (1 - c))
            # where var_fixed = materials + handling (shipping handled separately if relevant)
            var_fixed = mkt_mc_est + PG_HANDLING_COST  # excludes shipping in this simplified view
            interior_P = mkt_wtp_max / 2 + var_fixed / (2 * (1 - comm_frac))

            if interior_P < mkt_wtp_min:
                optimal_mkt_price = mkt_wtp_min
                price_regime = f"Corner at min WTP (interior P*=${interior_P:,.0f} < min ${mkt_wtp_min:,.0f})"
                p_buy_at_opt = 1.0
            elif interior_P > mkt_wtp_max:
                optimal_mkt_price = mkt_wtp_max
                price_regime = f"At max WTP (demand → 0)"
                p_buy_at_opt = 0
            else:
                optimal_mkt_price = interior_P
                price_regime = "Interior optimum (between min and max)"
                denom = mkt_wtp_max - mkt_wtp_min if mkt_wtp_max > mkt_wtp_min else 1
                p_buy_at_opt = (mkt_wtp_max - interior_P) / denom

            commission_at_opt = optimal_mkt_price * comm_frac
            gross_profit_per_unit = optimal_mkt_price - commission_at_opt - PG_HANDLING_COST - mkt_mc_est
            margin_pct = (gross_profit_per_unit / optimal_mkt_price * 100) if optimal_mkt_price > 0 else 0

            # Peak sales calculation
            peak_t = (1/(mkt["p"]+mkt["q"])) * np.log(mkt["q"]/mkt["p"]) if mkt["p"] > 0 and mkt["q"] > mkt["p"] else 0
            peak_q = mkt_size_est * ((mkt["p"]+mkt["q"])**2) / (4*mkt["q"]) if mkt["q"] > 0 else 0

            # Display implied min + show theoretical optimal as reference
            st.caption(f"Implied min WTP: **${mkt_wtp_min:,.0f}** (= 2 × {mkt_wtp_median} − {mkt_wtp_max}). "
                        f"Theoretical optimal: **${optimal_mkt_price:,.0f}** — {price_regime}")

            # ── Price slider: let user override the optimal ────────────────
            mkt_price_slider_min = int(mkt_mc_est + PG_HANDLING_COST)  # floor = unit cost
            mkt_price_slider_max = int(mkt_wtp_max * 1.2)
            mkt_price = st.slider("Your Retail Price ($)",
                                    mkt_price_slider_min, mkt_price_slider_max,
                                    int(optimal_mkt_price), step=10,
                                    key=f"mkt_price_{i}",
                                    help=f"Defaults to theoretical optimal ${optimal_mkt_price:,.0f}. Drag to see P(buy) and CM at any price.")

            # Compute P(buy) at slider price
            if mkt_price <= mkt_wtp_min:
                p_buy_at_price = 1.0
            elif mkt_price >= mkt_wtp_max:
                p_buy_at_price = 0.0
            else:
                p_buy_at_price = (mkt_wtp_max - mkt_price) / (mkt_wtp_max - mkt_wtp_min)

            # CM calculation at slider price (no tax, shipping included)
            cm_shipping_mkt = 20  # mail in-region default
            commission_at_price = mkt_price * comm_frac
            cm_per_unit = (mkt_price - commission_at_price - PG_HANDLING_COST
                            - mkt_mc_est - cm_shipping_mkt)
            cm_per_arrival = cm_per_unit * p_buy_at_price

            cm_color = "#2d6a2e" if cm_per_unit > 0 else "#b22222"

            st.markdown(f"""
<div style="background:rgba({'45,106,46' if cm_per_unit > 0 else '178,34,34'},0.12);
    border-left:4px solid {cm_color}; border-radius:6px; padding:0.6rem 0.8rem; margin:0.3rem 0;">
<div style="font-size:0.8rem;opacity:0.7;">At price ${mkt_price:,}</div>
<div style="display:flex;gap:1.2rem;flex-wrap:wrap;margin-top:0.3rem;">
<div><span style="opacity:0.7;font-size:0.75rem;">P(buy)</span><br><b style="font-size:1.1rem;">{p_buy_at_price:.0%}</b></div>
<div><span style="opacity:0.7;font-size:0.75rem;">CM / unit</span><br><b style="font-size:1.1rem;color:{cm_color};">${cm_per_unit:,.0f}</b></div>
<div><span style="opacity:0.7;font-size:0.75rem;">CM / arrival</span><br><b style="font-size:1.1rem;color:{cm_color};">${cm_per_arrival:,.0f}</b></div>
</div>
</div>
""", unsafe_allow_html=True)

            st.metric("Bass Peak Sales (market, 100% buy)", f"{peak_q:,.1f}/day",
                       help="Bass peak assuming 100% conversion. Adjust by current P(buy).")

            # Unit economics detail
            with st.expander("Unit economics & pricing logic", expanded=False):
                st.markdown(f"""
**Focus group inputs:**
- Max WTP: ${mkt_wtp_max:,.0f} (100th pct)
- Median WTP: ${mkt_wtp_median:,.0f} (50th pct)
- Implied min WTP: ${mkt_wtp_min:,.0f} (uniform assumption)

**Theoretical optimal regime:** {price_regime}

**Formula:**
Interior P* = max/2 + var_fixed / (2 × (1 − commission))
           = ${mkt_wtp_max:,.0f}/2 + ${var_fixed:,.0f} / 1.6
           = ${interior_P:,.0f}

**Your chosen price: ${mkt_price:,}**

**Demand:**
- P(buy) = (max − P) / (max − min) = ({mkt_wtp_max:,} − {mkt_price:,}) / ({mkt_wtp_max:,} − {mkt_wtp_min:,}) = **{p_buy_at_price:.1%}**

**Contribution Margin per unit (before tax):**
- Revenue: ${mkt_price:,}
- (−) Commission ({PG_SALES_COMMISSION:.0f}%): -${commission_at_price:,.0f}
- (−) Handling: -${PG_HANDLING_COST}
- (−) Materials: -${mkt_mc_est}
- (−) Shipping (mail in-region): -${cm_shipping_mkt}
- **= CM per unit: ${cm_per_unit:,.0f}**

**Expected value per arriving customer:**
- CM × P(buy) = ${cm_per_unit:,.0f} × {p_buy_at_price:.0%} = **${cm_per_arrival:,.0f}**

**Bass implications:**
- Peak arrivals: {peak_q:,.1f}/day at market peak
- Peak purchases at this price: {peak_q * p_buy_at_price:,.1f}/day
- Daily CM at peak: ${peak_q * p_buy_at_price * cm_per_unit:,.0f}

**Working capital:**
- DSO: {mkt['dso']} days → ${cm_per_unit * mkt['dso']:,.0f} WC tied per unit sold
""")

            mkt_results.append({
                "Market": mkt_sel,
                "Median WTP": f"${mkt_wtp_median:,}",
                "Max WTP": f"${mkt_wtp_max:,}",
                "Your Price": f"${mkt_price:,}",
                "P(buy)": f"{p_buy_at_price:.0%}",
                "CM/unit": f"${cm_per_unit:,.0f}",
                "CM/arrival": f"${cm_per_arrival:,.0f}",
                "Peak Sales/day (at your price)": f"{peak_q * p_buy_at_price:,.1f}",
                "Daily CM at peak": f"${peak_q * p_buy_at_price * cm_per_unit:,.0f}",
                "DSO (d)": mkt['dso'],
            })

    # Comparison summary table
    st.markdown("---")
    st.markdown("**Side-by-Side Summary**")
    st.dataframe(pd.DataFrame(mkt_results), use_container_width=True, hide_index=True)

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 4: PRODUCT DESIGN ROI CALCULATOR
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("4. Product Design ROI Calculator")
    st.caption("Step 1: Base features (platform, GPS, network, power, finish). Step 2: Detection agenda. Together they determine WTP, design cost, time, and materials.")

    # ── Feature descriptions reference (from Production Game Market Research) ─
    with st.expander("**📘 Base Feature Descriptions** — what each attribute means", expanded=False):
        st.markdown("""
**Platform** — Form factor (how the monitor is worn)
- Smaller wearables (wristbands) preferred by general pop, fashion-conscious consumers
- Chest-worn for clinical/medical applications where larger devices are OK
- Market preferences: MD Estrogen & Clinical Fertility slightly prefer wrist; Heart View is chest-band

**GPS** — Location correlation
- Imperative for **Law (Narcotic)** — deal breaker if missing
- Drives perceived safety in **MD Heart** — affects WTP
- Nice-to-have for others

**Network** — Data transmission mode
- **Bluetooth** (to smartphone) — lowest bandwidth, short range
- **2.4 GHz cellular** — medium range, cheaper
- **5 GHz cellular** — higher bandwidth (slightly more for Law market)
- **Law (Narcotic)** requires cellular (Bluetooth insufficient) — deal breaker

**Power** — Battery type
- **Ni-Cd battery** — small/basic, few hours on full charge
- **Polymer battery** — similar size to Ni-Cd, lasts several days
- **Two-battery pack (Ni-Cd or polymer)** — worn around waist, ~10× battery life but bulky
- **Clinical Fertility** & similar markets hate bulky packs (deal breaker)

**Finish** — Material appearance
- **Utilitarian plastic** (original) — functional but bland
- **Dyed/painted plastic** — matches platform garment
- Small WTP premium for athletic/fashion markets

---

**Detection Agenda (Core Features)** — what the monitor actually measures:
- Heartbeat, Blood vessel, Dissolved gasses, Toxicology, Hormone, Metabolic
- Market match: Heart View → Heartbeat; Fertility → Hormone; Law → Toxicology; Cancer → specific biomarkers
""")

    # ── Base features (Platform, GPS, Network, Power, Finish) ────────────────
    # Costs for base features aren't published in the design guide; user can override.
    BASE_FEATURES = {
        "Platform": {
            "Wristband (small)": (0, 0, 0, "General pop / fashion / fertility prefer this"),
            "Chest-worn band": (0, 0, 0, "Heart View default. OK for clinical."),
            "Ankle/leg band": (0, 0, 0, "Law Narcotic — monitors parolees"),
        },
        "GPS": {
            "No GPS": (0, 0, 0, "Fine for Fertility, Clinical. DEAL BREAKER for Law"),
            "GPS enabled": (0, 0, 5, "Required for Law Narcotic. Raises WTP for MD Heart"),
        },
        "Network": {
            "Bluetooth": (0, 0, 0, "Phone-connected only. NOT enough for Law"),
            "2.4 GHz cellular": (0, 0, 10, "Standard for medical. Good for all markets."),
            "5 GHz cellular": (0, 0, 15, "Higher bandwidth. Slight edge in Law."),
        },
        "Power": {
            "Ni-Cd battery": (0, 0, 0, "Few hours — basic"),
            "Polymer battery": (0, 0, 5, "Several days — general recommendation"),
            "Two-battery pack (Ni-Cd)": (0, 0, 10, "10× life BUT bulky — DEAL BREAKER for Fertility"),
            "Two-battery pack (polymer)": (0, 0, 15, "10× life + lighter but still bulky"),
        },
        "Finish": {
            "Original plastic": (0, 0, 0, "Utilitarian"),
            "Dyed plastic": (0, 0, 2, "Color-matched to platform"),
            "Painted plastic": (0, 0, 3, "Premium look — small WTP boost for athletic/fashion"),
        },
    }

    # ── Detection features (with published costs from Design Guide) ──────────
    DETECTION_FEATURES = {
        "Heartbeat": {"None": (3, 1000, 0), "Pulse only": (15, 30000, 15), "Temporal": (90, 135000, 25)},
        "Blood vessel": {"None": (3, 1000, 0), "Systolic only": (30, 75000, 10),
                          "Systolic & diastolic": (90, 135000, 15), "Full profile": (120, 180000, 40)},
        "Dissolved gasses": {"None": (3, 1000, 0), "O2 only": (30, 75000, 15),
                              "O2, N2, CO2": (90, 135000, 20), "Full C,N,O": (90, 135000, 40)},
        "Toxicology": {"None": (3, 1000, 0), "Ethanol": (30, 150000, 95),
                        "Amphetamine": (90, 250000, 140), "THC": (90, 250000, 140),
                        "Barbiturate": (90, 250000, 140), "Narcotic": (90, 250000, 140)},
        "Hormone": {"None": (3, 1000, 0), "LH": (30, 45000, 20), "LH and FSH": (60, 75000, 50),
                     "Estrogen": (60, 75000, 60), "Progesterone": (60, 75000, 60),
                     "Testosterone": (60, 75000, 50)},
        "Metabolic": {"None": (3, 1000, 0), "Thyroxine": (90, 90000, 155),
                       "Bilirubin": (90, 90000, 150), "Proteins": (90, 90000, 170),
                       "Uric acid": (90, 90000, 160)},
    }

    # Global settings row
    pd_config_col1, pd_config_col2 = st.columns([1, 3])
    with pd_config_col1:
        pd_n = st.number_input("Products to design", min_value=1, max_value=4,
                                value=2, step=1, key="pd_n")
    with pd_config_col2:
        st.caption("⚠ Base feature development costs aren't published in the Design Guide — defaults are $0/0 days. "
                   "Materials $/unit are our estimates. Verify in-game and override if needed.")

    # Default presets — now include base features + detection
    PRESETS = {
        "A — Heart View (flagship)": {
            "Platform": "Chest-worn band", "GPS": "GPS enabled", "Network": "2.4 GHz cellular",
            "Power": "Polymer battery", "Finish": "Original plastic",
            "Heartbeat": "Temporal", "Blood vessel": "None",
            "Dissolved gasses": "None", "Toxicology": "None",
            "Hormone": "None", "Metabolic": "None",
            "price": 700,
        },
        "B — Estrogen monitor (wrist)": {
            "Platform": "Wristband (small)", "GPS": "No GPS", "Network": "Bluetooth",
            "Power": "Polymer battery", "Finish": "Original plastic",
            "Heartbeat": "None", "Blood vessel": "None",
            "Dissolved gasses": "None", "Toxicology": "None",
            "Hormone": "Estrogen", "Metabolic": "None",
            "price": 731,
        },
        "C — Narcotic law device": {
            "Platform": "Ankle/leg band", "GPS": "GPS enabled", "Network": "5 GHz cellular",
            "Power": "Two-battery pack (polymer)", "Finish": "Original plastic",
            "Heartbeat": "None", "Blood vessel": "None",
            "Dissolved gasses": "None", "Toxicology": "Narcotic",
            "Hormone": "None", "Metabolic": "None",
            "price": 1350,
        },
        "D — Cancer breast premium": {
            "Platform": "Chest-worn band", "GPS": "GPS enabled", "Network": "2.4 GHz cellular",
            "Power": "Polymer battery", "Finish": "Dyed plastic",
            "Heartbeat": "Pulse only", "Blood vessel": "Systolic only",
            "Dissolved gasses": "None", "Toxicology": "None",
            "Hormone": "None", "Metabolic": "None",
            "price": 1200,
        },
    }
    preset_keys = list(PRESETS.keys())

    st.markdown("---")
    pd_cols = st.columns(int(pd_n))
    pd_results = []
    for i, col in enumerate(pd_cols):
        with col:
            # Preset picker
            preset_default_idx = min(i, len(preset_keys) - 1)
            preset_sel = st.selectbox(f"Preset for Product {i+1}", preset_keys,
                                       index=preset_default_idx, key=f"pd_preset_{i}")
            preset = PRESETS[preset_sel]

            # ── Base Features ─────────────────────────────────────────────────
            st.markdown(f"**Product {i+1} — Base Features**")
            selected_base = {}
            for attr, options in BASE_FEATURES.items():
                default_feat = preset.get(attr, list(options.keys())[0])
                default_idx = list(options.keys()).index(default_feat) if default_feat in options else 0
                selected_base[attr] = st.selectbox(attr, list(options.keys()),
                                                     index=default_idx,
                                                     key=f"pd_base_{attr}_{i}",
                                                     help=options[default_feat][3] if len(options[default_feat]) > 3 else "")

            # ── Detection Agenda ──────────────────────────────────────────────
            st.markdown(f"**Product {i+1} — Detection Agenda**")
            selected_det = {}
            for attr, options in DETECTION_FEATURES.items():
                default_feat = preset[attr]
                default_idx = list(options.keys()).index(default_feat) if default_feat in options else 0
                selected_det[attr] = st.selectbox(attr, list(options.keys()),
                                                    index=default_idx,
                                                    key=f"pd_det_{attr}_{i}")

            # ── Calculate totals from both base + detection ───────────────────
            base_days = max((BASE_FEATURES[a][selected_base[a]][0] for a in BASE_FEATURES), default=0)
            base_cost = sum(BASE_FEATURES[a][selected_base[a]][1] for a in BASE_FEATURES)
            base_mat = sum(BASE_FEATURES[a][selected_base[a]][2] for a in BASE_FEATURES)

            det_days = max(DETECTION_FEATURES[a][selected_det[a]][0] for a in DETECTION_FEATURES)
            det_cost = sum(DETECTION_FEATURES[a][selected_det[a]][1] for a in DETECTION_FEATURES)
            det_mat = sum(DETECTION_FEATURES[a][selected_det[a]][2] for a in DETECTION_FEATURES)

            total_days = max(base_days, det_days)  # parallel development
            total_cost = base_cost + det_cost
            total_mat = base_mat + det_mat

            pd_target_price = st.number_input("Target Retail Price ($)",
                                                value=preset["price"], step=25,
                                                key=f"pd_price_{i}")
            pd_daily_sales = st.number_input("Est. Daily Sales",
                                               value=5, step=1,
                                               key=f"pd_sales_{i}")

            pd_margin = pd_target_price * (1 - comm_frac) - PG_HANDLING_COST - total_mat
            pd_break_days = (total_cost / (pd_margin * pd_daily_sales)
                              if pd_margin > 0 and pd_daily_sales > 0 else float("inf"))

            st.metric("Design Time", f"{total_days} days")
            st.metric("Design Cost", f"${total_cost:,}")
            st.metric("Materials / Unit", f"${total_mat}")
            st.metric("Unit Margin (before tax)", f"${pd_margin:.0f}",
                       help=f"Price × 80% − ${PG_HANDLING_COST} handling − ${total_mat} materials")
            if pd_break_days < float("inf"):
                st.metric("Break-even (design cost)", f"{pd_break_days:.0f} days",
                           delta=f"{pd_break_days/30:.1f} months",
                           delta_color="off")
            else:
                st.error("Negative margin")

            # Feature summary in expander
            with st.expander("Feature breakdown", expanded=False):
                feat_df = []
                for attr, feat in selected_base.items():
                    d, c, m = BASE_FEATURES[attr][feat][:3]
                    feat_df.append({"Type": "Base", "Attribute": attr, "Feature": feat,
                                     "Days": d, "Cost": f"${c:,}", "Mat $/u": f"${m}"})
                for attr, feat in selected_det.items():
                    d, c, m = DETECTION_FEATURES[attr][feat]
                    feat_df.append({"Type": "Detection", "Attribute": attr, "Feature": feat,
                                     "Days": d, "Cost": f"${c:,}", "Mat $/u": f"${m}"})
                st.dataframe(pd.DataFrame(feat_df), use_container_width=True, hide_index=True)

            # Deal-breaker flag check
            db_warnings = []
            if selected_det["Toxicology"] != "None" and selected_base["GPS"] == "No GPS":
                db_warnings.append("Law Narcotic: needs GPS")
            if selected_det["Toxicology"] != "None" and selected_base["Network"] == "Bluetooth":
                db_warnings.append("Law Narcotic: needs cellular")
            if selected_det["Hormone"] != "None" and "Two-battery pack" in selected_base["Power"]:
                db_warnings.append("Fertility: avoid bulky battery packs")
            if db_warnings:
                st.warning("⚠ Potential deal breakers:\n" + "\n".join(f"- {w}" for w in db_warnings))

            pd_results.append({
                "Product": f"P{i+1}: {preset_sel}",
                "Platform": selected_base["Platform"].split(" ")[0],
                "Detection": ", ".join(f"{a[:4]}:{f}" for a, f in selected_det.items() if f != "None") or "None",
                "Days": total_days,
                "Design $": f"${total_cost:,}",
                "Mat/u": f"${total_mat}",
                "Price": f"${pd_target_price}",
                "Margin/u": f"${pd_margin:.0f}",
                "Break-even": f"{pd_break_days:.0f}d" if pd_break_days < float("inf") else "N/A",
            })

    # Side-by-side summary comparison
    st.markdown("---")
    st.markdown("**Product Comparison Summary**")
    st.dataframe(pd.DataFrame(pd_results), use_container_width=True, hide_index=True)

    # Visualize: Design cost vs break-even days
    if len(pd_results) > 1:
        viz_data = []
        for i, r in enumerate(pd_results):
            try:
                design_val = int(r["Design $"].replace("$", "").replace(",", ""))
                be_str = r["Break-even"]
                if be_str != "N/A":
                    be_val = int(be_str.replace("d", ""))
                    viz_data.append({"product": f"P{i+1}",
                                      "design_cost": design_val,
                                      "break_even_days": be_val,
                                      "name": r["Product"]})
            except Exception:
                pass
        if viz_data:
            fig_pd = go.Figure()
            fig_pd.add_trace(go.Scatter(
                x=[d["design_cost"] for d in viz_data],
                y=[d["break_even_days"] for d in viz_data],
                mode="markers+text",
                marker=dict(size=16, color=["#800000", "#1a3c5e", "#b8860b", "#2d6a2e"][:len(viz_data)]),
                text=[d["product"] for d in viz_data],
                textposition="top center",
                hovertext=[d["name"] for d in viz_data],
            ))
            fig_pd.update_layout(height=300, xaxis_title="Design Cost ($)",
                                  yaxis_title="Break-even (days)",
                                  title="Design Cost vs Break-even Speed (lower-left = best)",
                                  margin=dict(l=0, r=0, t=40, b=0),
                                  xaxis_tickformat="$,.0f")
            st.plotly_chart(fig_pd, use_container_width=True)

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 5: BOND ISSUANCE CALCULATOR
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("5. Bond Issuance & Credit Rating Calculator")
    st.caption("Determine debt capacity based on interest coverage ratio")

    BOND_RATINGS = {
        "Excellent": {"threshold": 20, "rate": 10.0, "color": "#2d6a2e"},
        "Good": {"threshold": 7, "rate": 15.0, "color": "#b8860b"},
        "Poor": {"threshold": 2, "rate": 25.0, "color": "#b22222"},
    }

    bd_col1, bd_col2 = st.columns([1, 2])
    with bd_col1:
        bd_ebit = st.number_input("Yearly EBIT ($)", value=500000, step=50000, key="bd_ebit",
                                    help="Last full quarter EBIT × 4")
        bd_existing_interest = st.number_input("Existing Interest ($/yr)", value=0, step=1000, key="bd_ei")
        bd_new_bonds = st.number_input("New Bonds to Issue (face value)", value=100000, step=10000, key="bd_bonds")

    with bd_col2:
        st.markdown("**Rating vs Capacity**")
        bond_data = []
        for rating, params in BOND_RATINGS.items():
            # Max total interest at this threshold = EBIT / threshold
            max_total_interest = bd_ebit / params["threshold"] if params["threshold"] > 0 else 0
            max_new_interest = max_total_interest - bd_existing_interest
            # How much bonds at this rate gives this interest?
            max_bonds = max_new_interest / (params["rate"] / 100) if params["rate"] > 0 else 0

            # Check if our new_bonds fits
            new_interest = bd_new_bonds * params["rate"] / 100
            total_interest = bd_existing_interest + new_interest
            coverage = bd_ebit / total_interest if total_interest > 0 else float("inf")

            bond_data.append({
                "Rating": rating,
                "Threshold (coverage)": f"{params['threshold']}×",
                "Interest Rate": f"{params['rate']:.1f}% APR",
                "Max Bonds @ Rating": f"${max(0, max_bonds):,.0f}",
                "Actual Coverage": f"{coverage:.1f}×" if coverage < float("inf") else "∞",
                "Qualifies": "Yes" if coverage >= params["threshold"] else "No",
            })

        st.dataframe(pd.DataFrame(bond_data), use_container_width=True, hide_index=True)

        # Recommendation
        for rating, params in BOND_RATINGS.items():
            new_interest = bd_new_bonds * params["rate"] / 100
            total_interest = bd_existing_interest + new_interest
            coverage = bd_ebit / total_interest if total_interest > 0 else float("inf")
            if coverage >= params["threshold"]:
                st.success(f"Your ${bd_new_bonds:,} issuance qualifies for **{rating}** rating at **{params['rate']:.1f}% APR**")
                st.info(f"Annual interest: ${new_interest:,.0f} | 5-year total cost (interest only): ${new_interest*5:,.0f}")
                break
        else:
            st.error(f"Your ${bd_new_bonds:,} issuance does not qualify at any rating. Reduce bond amount or increase EBIT.")

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 6: NPV CALCULATOR FOR EXPANSION
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("6. Factory / DC Expansion NPV Calculator")
    st.caption("Evaluate whether new capacity pays back within the 4-year game horizon")

    npv_col1, npv_col2 = st.columns(2)
    with npv_col1:
        npv_type = st.radio("Asset Type", ["New Factory", "New DC", "Capital Addition"], key="npv_type")
        if npv_type == "New Factory":
            default_capex = 500000  # min for production line
            default_build = PG_FACTORY_BUILD
            default_daily = 2500
            default_rev = 5000
        elif npv_type == "New DC":
            default_capex = PG_DC_CAPITAL + PG_DC_LAND
            default_build = PG_DC_BUILD
            default_daily = PG_DC_DAILY
            default_rev = 8000
        else:
            default_capex = 100000
            default_build = 30
            default_daily = 0
            default_rev = 500

        npv_capex = st.number_input("Total Capital Investment ($)", value=default_capex, step=50000, key="npv_capex")
        npv_land = st.number_input("Of which: Land (non-depreciating)", value=100000, step=10000, key="npv_land",
                                    help="Land doesn't depreciate — only plant/equipment gets the tax shield")
        npv_build = st.number_input("Build Time (days)", value=default_build, step=5, key="npv_build")
        npv_daily_cost = st.number_input("Incremental Daily Cost ($)", value=default_daily, step=100, key="npv_daily")
        npv_daily_rev = st.number_input("Incremental Daily Revenue ($)", value=default_rev, step=500, key="npv_rev")
        npv_remaining_days = st.number_input("Game Days Remaining", value=1095, step=30, key="npv_days",
                                               help="From day 365 to day 1460 = 1095 days. Game continues autonomously day 910 → 1092.")
        npv_discount = st.number_input("Discount Rate (% APR)", value=15.0, step=0.5, key="npv_dr") / 100
        npv_tax_rate = 0.35
        npv_dep_years = PG_DEPRECIATION_YRS

    with npv_col2:
        # Depreciable capital (excludes land)
        npv_depreciable = max(0, npv_capex - npv_land)
        daily_depreciation = npv_depreciable / npv_dep_years / 364
        daily_dep_tax_shield = daily_depreciation * npv_tax_rate

        # Build period: capex + daily costs, no revenue. Depreciation starts upon operational.
        operating_days = max(0, npv_remaining_days - npv_build)

        # Daily discount
        daily_discount = (1 + npv_discount) ** (1/365) - 1
        npv = -npv_capex  # Initial outflow at day 0

        total_dep_taken = 0
        for day in range(int(npv_remaining_days)):
            if day < npv_build:
                # Build period: incremental daily cost, no revenue, no depreciation yet
                cash_flow = -npv_daily_cost
            else:
                # Operating period: revenue - cost, plus depreciation tax shield
                # Depreciation only for the days it applies within the game window
                operating_profit = npv_daily_rev - npv_daily_cost
                # Tax is on operating profit minus depreciation (reduces taxable income)
                taxable = operating_profit - daily_depreciation
                tax = max(0, taxable) * npv_tax_rate
                cash_flow = operating_profit - tax  # = op_profit × (1-t) + dep × t
                total_dep_taken += daily_depreciation
            npv += cash_flow / ((1 + daily_discount) ** day)

        # Compute payback (undiscounted) with tax shield
        cumulative_cash = -npv_capex
        payback_day = None
        for day in range(int(npv_remaining_days)):
            if day < npv_build:
                cumulative_cash -= npv_daily_cost
            else:
                operating_profit = npv_daily_rev - npv_daily_cost
                taxable = operating_profit - daily_depreciation
                tax = max(0, taxable) * npv_tax_rate
                cumulative_cash += operating_profit - tax
            if cumulative_cash > 0 and payback_day is None:
                payback_day = day
                break

        # PV of depreciation tax shield (for display)
        pv_tax_shield = 0
        for day in range(int(npv_build), int(npv_remaining_days)):
            pv_tax_shield += daily_dep_tax_shield / ((1 + daily_discount) ** day)

        # Undiscounted annual operating profit for display
        annual_op_profit_after_tax = (npv_daily_rev - npv_daily_cost - daily_depreciation) * 364 * (1 - npv_tax_rate) + daily_depreciation * 364

        npv_color = "#2d6a2e" if npv > 0 else "#b22222"
        st.markdown(f"""
<div style="border:2px solid {npv_color};border-radius:10px;padding:1rem;">
<h4 style="margin:0;color:{npv_color};">NPV Analysis (with Depreciation Tax Shield)</h4>
<table style="width:100%;margin-top:0.5rem;">
<tr><td>Total Capital Investment</td><td style="text-align:right;">-${npv_capex:,}</td></tr>
<tr><td>&nbsp;&nbsp;&nbsp;of which Land (no depreciation)</td><td style="text-align:right;">${npv_land:,}</td></tr>
<tr><td>&nbsp;&nbsp;&nbsp;Depreciable Capital</td><td style="text-align:right;">${npv_depreciable:,}</td></tr>
<tr><td>Daily Depreciation ({npv_dep_years}-yr SL)</td><td style="text-align:right;">${daily_depreciation:.2f}</td></tr>
<tr><td>Daily Tax Shield (35% × dep.)</td><td style="text-align:right;">${daily_dep_tax_shield:.2f}</td></tr>
<tr><td>Build Period</td><td style="text-align:right;">{npv_build} days</td></tr>
<tr><td>Operating Period</td><td style="text-align:right;">{operating_days} days</td></tr>
<tr><td>PV of Depreciation Tax Shield</td><td style="text-align:right;color:#2d6a2e;">+${pv_tax_shield:,.0f}</td></tr>
<tr><td>Annual Operating Profit (after tax + shield)</td><td style="text-align:right;">${annual_op_profit_after_tax:,.0f}</td></tr>
<tr><td><b>NPV (at {npv_discount*100:.1f}% APR)</b></td><td style="text-align:right;font-size:1.3rem;font-weight:700;color:{npv_color};">${npv:,.0f}</td></tr>
<tr><td>Payback Period</td><td style="text-align:right;">{payback_day if payback_day else "Never"} days</td></tr>
</table>
</div>
""", unsafe_allow_html=True)

        if npv > 0:
            st.success(f"✓ Positive NPV — investment creates value of ${npv:,.0f} (includes ${pv_tax_shield:,.0f} from depreciation tax shield)")
        else:
            st.error(f"✗ Negative NPV — avoid this investment")

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 7: CHEAT SHEET
    # ══════════════════════════════════════════════════════════════════════════
    with st.expander("**Production Game Cheat Sheet — Key Differences from Monopoly Game**", expanded=False):
        st.markdown("""
| Aspect | Monopoly Game (Apr 12) | Production Game (Apr 13+) |
|---|---|---|
| **Demand Model** | Uniform WTP + constant arrival | **Bass diffusion** (p + q × A/M) × (M - A) |
| **Horizon** | 91 days | **4 years (1,460 days)** |
| **Starting Cash** | $1.5M | **$4M** (you own 50% after paying $2M) |
| **Products** | 1 specialty + 1 hormone | **Design your own** via features |
| **Production** | Fixed batch/time | **Cobb-Douglas**: Y = A × K^α × L^β |
| **Technologies** | 1 | **3** (Benches, Production Line, Automated Cell) |
| **Handling Cost** | None | **$10/unit** |
| **Sales Commission** | None | **20% of price** |
| **Markets** | Simple | **5+ segments** (Clinical, Medical, Law, Military) |
| **Financing** | None | **Bonds** (Excellent/Good/Poor ratings) |
| **Cost of Capital** | 10% | **15% APR** |
| **Raw Materials Payable** | 15 days | **30 days** |
| **Build Times** | DC: 60d, Factory: 90d | Same |
| **Depreciation** | 15 years | 15 years |

**Critical strategic differences:**
1. **Bass model means early investments pay off much later** — demand ramps slowly
2. **20% commission is a massive cost** — always factor into pricing
3. **4-year horizon justifies bigger investments** — factory/DC NPV works
4. **Product design is a strategic lever** — pick markets with best WTP/cost ratio
5. **Advertising shifts the Bass curve** — not just short-term demand
6. **Debt is available** at 10-25% APR depending on credit rating
        """)

    with st.expander("**Strategic Playbook for Tonight**", expanded=False):
        st.markdown("""
### Early Game (Days 1-90): Foundation
1. **Run a focus group** ($20K, 7 days) for market research before designing products
2. **Don't rush product design** — pick your target market carefully
3. **Start small**: Stick with Benches technology until you have sales validation
4. **Low retail prices initially** to seed the Bass curve (innovators → imitators)

### Mid Game (Days 90-900): Growth
1. **Add advertising** once you have a product selling — amplifies p coefficient
2. **Consider Production Line** if daily demand > 10 units/day consistently
3. **Build DC in second region** if market research shows higher WTP there
4. **Issue bonds** at Excellent/Good rating to fund expansion (10-15% vs 15% cost of capital)

### Late Game (Days 900+): Scale
1. **Automated Cell** only if you're producing >50 units/day
2. **Multi-product portfolio** — cover different market segments
3. **Defend market share** against imitator teams
4. **Wind down inventory** in final 30 days to convert to cash

### Watch-Outs
- **Don't over-invest in capex early** — payback period matters with 4-year horizon
- **Commission (20%) + handling ($10) + materials** are the real cost base
- **DSO varies wildly** (10-90 days) — cash flow matters
- **Emergency loans at 40%** will destroy you — always keep cash buffer
        """)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: LEARNING DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📊 Learning Dashboard":
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
    today_str = datetime.now().strftime("%Y-%m-%d")
    fig_gantt.add_shape(
        type="line", x0=today_str, x1=today_str, y0=0, y1=1,
        xref="x", yref="paper",
        line=dict(color="red", width=2, dash="dash"),
    )
    fig_gantt.add_annotation(
        x=today_str, y=1.05, xref="x", yref="paper",
        text="TODAY", showarrow=False, font=dict(color="red", size=12),
    )
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

    # ── Build the full graph (needed for popups and main viz) ─────────────────
    categories = ["All"] + list(CATEGORY_COLORS.keys())
    filter_col, explore_col = st.columns([1, 2])
    with filter_col:
        sel_cat = st.selectbox("Filter by discipline", categories)

    # Build NetworkX graph
    G = nx.Graph()
    # Track which course each framework belongs to
    fw_to_courses = {}

    for c in COURSES:
        if sel_cat != "All" and c["category"] != sel_cat:
            continue
        G.add_node(c["name"], node_type="course", category=c["category"],
                    size=max(c["units"], 30), color=CATEGORY_COLORS.get(c["category"], "#999"))
        for f in c["frameworks"]:
            G.add_node(f, node_type="framework", category=c["category"],
                        size=15, color=CATEGORY_COLORS.get(c["category"], "#999"))
            G.add_edge(c["name"], f, weight=2)
            fw_to_courses.setdefault(f, []).append(c["name"])

    for src, tgt, label in CONCEPT_CONNECTIONS:
        if G.has_node(src) and G.has_node(tgt):
            G.add_edge(src, tgt, weight=1, label=label)

    # Framework explorer selector
    all_frameworks = sorted([n for n in G.nodes()
                             if G.nodes[n].get("node_type") == "framework"])
    with explore_col:
        selected_fw = st.selectbox(
            "Explore a framework (click to open details)",
            ["— Select a framework —"] + all_frameworks,
            key="kg_explore_fw",
        )

    # ── Framework popup overlay ───────────────────────────────────────────────
    if selected_fw and selected_fw != "— Select a framework —":
        fw_data = G.nodes[selected_fw]
        neighbors = list(G.neighbors(selected_fw))
        neighbor_courses = [n for n in neighbors if G.nodes[n].get("node_type") == "course"]
        neighbor_fws = [n for n in neighbors if G.nodes[n].get("node_type") == "framework"]

        # Find cross-discipline bridges involving this framework
        fw_bridges = []
        for src, tgt, label in CONCEPT_CONNECTIONS:
            if (src == selected_fw or tgt == selected_fw) and G.has_node(src) and G.has_node(tgt):
                other = tgt if src == selected_fw else src
                fw_bridges.append({"Connected To": other, "Bridge Path": label})

        # Popup overlay card
        cat_color = fw_data.get("color", "#666")
        st.markdown(f"""
<div style="border:2px solid {cat_color}; border-radius:12px; padding:20px 24px;
            margin:8px 0 16px 0; background:rgba(255,255,255,0.03);
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);">
  <div style="display:flex; align-items:center; gap:12px; margin-bottom:12px;">
    <div style="width:14px; height:14px; border-radius:50%; background:{cat_color};"></div>
    <span style="font-size:1.4em; font-weight:700; color:{cat_color};">{selected_fw}</span>
    <span style="font-size:0.85em; color:rgba(180,180,180,0.8); margin-left:auto;">
      {fw_data.get("category", "")} &middot; {G.degree(selected_fw)} connections
    </span>
  </div>
</div>
""", unsafe_allow_html=True)

        pop_col1, pop_col2 = st.columns([1, 1])

        with pop_col1:
            # Parent courses
            st.markdown("**Taught In**")
            for course in neighbor_courses:
                c_color = G.nodes[course].get("color", "#666")
                st.markdown(f'<span style="color:{c_color}; font-weight:600;">&#9679;</span> {course}',
                            unsafe_allow_html=True)

            # Connected frameworks
            st.markdown("")
            st.markdown("**Connected Frameworks**")
            for fw in sorted(neighbor_fws):
                fw_cat = G.nodes[fw].get("category", "")
                fw_color = G.nodes[fw].get("color", "#888")
                # Find edge label if it exists
                edge_data = G.edges.get((selected_fw, fw), {})
                edge_label = edge_data.get("label", "")
                label_str = f' <span style="color:rgba(160,160,160,0.7); font-size:0.85em;">via {edge_label}</span>' if edge_label else ""
                parent_courses = fw_to_courses.get(fw, [])
                course_str = f' <span style="color:rgba(140,140,140,0.6); font-size:0.8em;">({", ".join(parent_courses[:2])})</span>' if parent_courses else ""
                st.markdown(
                    f'<span style="color:{fw_color};">&#9656;</span> {fw}{label_str}{course_str}',
                    unsafe_allow_html=True,
                )

            # Cross-discipline bridges
            if fw_bridges:
                st.markdown("")
                st.markdown("**Cross-Discipline Bridges**")
                for b in fw_bridges:
                    st.markdown(f"&#8644; **{b['Connected To']}** — _{b['Bridge Path']}_")

        with pop_col2:
            # Mini neighborhood subgraph
            st.markdown("**Neighborhood Graph**")
            subgraph_nodes = [selected_fw] + neighbors
            H = G.subgraph(subgraph_nodes).copy()
            sub_pos = nx.spring_layout(H, k=1.8, iterations=40, seed=42)

            # Build mini plotly figure
            sub_edge_x, sub_edge_y = [], []
            for e in H.edges():
                x0, y0 = sub_pos[e[0]]
                x1, y1 = sub_pos[e[1]]
                sub_edge_x.extend([x0, x1, None])
                sub_edge_y.extend([y0, y1, None])

            sub_edge_trace = go.Scatter(
                x=sub_edge_x, y=sub_edge_y, mode="lines",
                line=dict(width=1.5, color="rgba(200,200,200,0.6)"),
                hoverinfo="none",
            )

            # Nodes for mini graph
            sub_nx, sub_ny, sub_text, sub_colors, sub_sizes, sub_symbols = (
                [], [], [], [], [], [])
            for node in H.nodes():
                x, y = sub_pos[node]
                ndata = H.nodes[node]
                sub_nx.append(x); sub_ny.append(y)
                sub_text.append(node)
                if node == selected_fw:
                    sub_colors.append(cat_color)
                    sub_sizes.append(22)
                    sub_symbols.append("star")
                elif ndata.get("node_type") == "course":
                    sub_colors.append(ndata.get("color", "#666"))
                    sub_sizes.append(16)
                    sub_symbols.append("square")
                else:
                    sub_colors.append(ndata.get("color", "#888"))
                    sub_sizes.append(12)
                    sub_symbols.append("circle")

            sub_node_trace = go.Scatter(
                x=sub_nx, y=sub_ny, mode="markers+text",
                marker=dict(size=sub_sizes, color=sub_colors, symbol=sub_symbols,
                            line=dict(width=1.5, color="white")),
                text=sub_text, textposition="top center",
                textfont=dict(size=9, color="#ccc"),
                hoverinfo="text",
            )

            sub_fig = go.Figure(data=[sub_edge_trace, sub_node_trace])
            sub_fig.update_layout(
                showlegend=False, height=350,
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                margin=dict(l=5, r=5, t=5, b=5),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(sub_fig, use_container_width=True)

        st.markdown("---")

    # ── Main graph with highlighting ──────────────────────────────────────────
    pos = nx.spring_layout(G, k=2.5, iterations=60, seed=42)

    # Determine highlighted nodes
    highlight_nodes = set()
    if selected_fw and selected_fw != "— Select a framework —":
        highlight_nodes = {selected_fw} | set(G.neighbors(selected_fw))

    # Build edge traces — split into highlighted and normal
    edge_x, edge_y = [], []
    hl_edge_x, hl_edge_y = [], []
    hl_edge_labels = []  # (midx, midy, label) for edge annotations
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        if highlight_nodes and (u in highlight_nodes and v in highlight_nodes):
            hl_edge_x.extend([x0, x1, None])
            hl_edge_y.extend([y0, y1, None])
            # Collect edge label at midpoint
            edge_data = G.edges.get((u, v), {})
            label = edge_data.get("label", "")
            if label:
                hl_edge_labels.append(((x0 + x1) / 2, (y0 + y1) / 2, label))
        else:
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

    traces = []
    traces.append(go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=0.5, color="rgba(200,200,200,0.3)" if highlight_nodes else "#ccc"),
        hoverinfo="none", showlegend=False,
    ))
    if hl_edge_x:
        traces.append(go.Scatter(
            x=hl_edge_x, y=hl_edge_y, mode="lines",
            line=dict(width=2.5, color="rgba(255,200,50,0.7)"),
            hoverinfo="none", showlegend=False,
        ))

    # Separate course nodes
    course_x, course_y, course_text, course_colors, course_sizes = [], [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        data = G.nodes[node]
        if data.get("node_type") == "course":
            course_x.append(x); course_y.append(y)
            course_text.append(node)
            course_colors.append(data["color"])
            course_sizes.append(data["size"] / 3)

    # Split highlighted framework nodes: show text labels on them
    hl_fw_x, hl_fw_y, hl_fw_text, hl_fw_colors, hl_fw_sizes = [], [], [], [], []
    reg_fw_x, reg_fw_y, reg_fw_text, reg_fw_colors, reg_fw_sizes, reg_fw_opacities = (
        [], [], [], [], [], [])

    for i, node in enumerate([n for n in G.nodes() if G.nodes[n].get("node_type") != "course"]):
        x, y = pos[node]
        data = G.nodes[node]
        degree = G.degree(node)
        if highlight_nodes and node in highlight_nodes:
            hl_fw_x.append(x); hl_fw_y.append(y)
            hl_fw_text.append(node)
            hl_fw_colors.append(data["color"])
            hl_fw_sizes.append(12 + degree * 4)
        elif highlight_nodes:
            reg_fw_x.append(x); reg_fw_y.append(y)
            reg_fw_text.append(f"{node} ({degree} connections)")
            reg_fw_colors.append("#555")
            reg_fw_sizes.append(6 + degree * 2)
            reg_fw_opacities.append(0.2)
        else:
            reg_fw_x.append(x); reg_fw_y.append(y)
            reg_fw_text.append(f"{node} ({degree} connections)")
            reg_fw_colors.append(data["color"])
            reg_fw_sizes.append(8 + degree * 3)
            reg_fw_opacities.append(0.7)

    # Regular (non-highlighted) framework nodes
    traces.append(go.Scatter(
        x=reg_fw_x, y=reg_fw_y, mode="markers",
        marker=dict(size=reg_fw_sizes, color=reg_fw_colors,
                    opacity=reg_fw_opacities if reg_fw_opacities else [0.7],
                    line=dict(width=1, color="white")),
        text=reg_fw_text, hoverinfo="text", name="Frameworks",
    ))

    # Highlighted framework nodes — with text labels overlaid
    if hl_fw_x:
        traces.append(go.Scatter(
            x=hl_fw_x, y=hl_fw_y, mode="markers+text",
            marker=dict(size=hl_fw_sizes, color=hl_fw_colors, opacity=1.0,
                        line=dict(width=2, color="white")),
            text=hl_fw_text, textposition="top center",
            textfont=dict(size=9, color="#333"),
            hoverinfo="text", name="Connected",
            showlegend=False,
        ))

    traces.append(go.Scatter(
        x=course_x, y=course_y, mode="markers+text",
        marker=dict(size=course_sizes, color=course_colors, line=dict(width=2, color="white")),
        text=course_text, textposition="top center",
        textfont=dict(size=10, color="#333"),
        hoverinfo="text", name="Courses",
    ))

    fig_graph = go.Figure(data=traces)

    # Add edge label annotations on highlighted connections
    annotations = []
    for mx, my, label in hl_edge_labels:
        annotations.append(dict(
            x=mx, y=my, text=label, showarrow=False,
            font=dict(size=8, color="rgba(255,200,50,0.9)"),
            bgcolor="rgba(0,0,0,0.6)", borderpad=2,
        ))

    fig_graph.update_layout(
        showlegend=True, height=700,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=0, r=0, t=30, b=0),
        plot_bgcolor="white",
        annotations=annotations,
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
# PAGE: WAR ROOM PREP (ISM Day 0 Readings - 14 docs summarized)
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📖 War Room Prep":
    st.markdown('<p class="big-header">War Room Prep</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Summarized learnings from 14 ISM Day 0 readings — apply directly to the Gleacher Game</p>',
                unsafe_allow_html=True)
    st.markdown("")

    # ── Category Overview ────────────────────────────────────────────────────
    st.markdown("""
<div style="background:linear-gradient(135deg,#1a3c5e,#2d5a8e);color:white;
    border-radius:12px;padding:1.2rem 1.5rem;margin-bottom:1rem;">
<h4 style="color:#ffd700;margin:0 0 0.5rem 0;">14 readings organized into 4 disciplines</h4>
<div style="display:flex;gap:2rem;flex-wrap:wrap;">
<div><b>Strategy</b> (5 docs)<br><span style="opacity:0.8;font-size:0.85rem;">What is Strategy, Strategy Statement, Positioning, 5 Forces, Competitive Advantage</span></div>
<div><b>Market Dynamics</b> (3 docs)<br><span style="opacity:0.8;font-size:0.85rem;">Corporate Scope, Entry Decisions, Commitment, Price Competition</span></div>
<div><b>Finance</b> (3 docs)<br><span style="opacity:0.8;font-size:0.85rem;">Capital Allocation, Financial Statements, APV Valuation</span></div>
<div><b>Operations</b> (2 docs)<br><span style="opacity:0.8;font-size:0.85rem;">Newsvendor Model, Production Inventories</span></div>
</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")

    # ═══ STRATEGY (5 documents) ═══
    st.markdown("## 📘 Strategy")

    with st.expander("**1. What Is Strategy? — Michael Porter (HBR 1996)**", expanded=False):
        st.markdown("""
### Core Thesis
**Operational effectiveness ≠ strategy.** Companies compete on both productivity frontier (OE) and strategic positioning.
OE is necessary but not sufficient — everyone can imitate best practices.

### Key Concepts
- **Strategic positioning** = performing different activities than rivals OR performing similar activities in different ways
- **Three bases of positioning:**
  1. **Variety-based** — subset of industry's products/services (e.g., Jiffy Lube = only oil changes)
  2. **Needs-based** — serve most needs of a particular customer group (e.g., IKEA for young urban shoppers)
  3. **Access-based** — reach customers in unusual ways (e.g., rural vs urban)
- **Trade-offs** are essential — you can't be all things to all people. Activities incompatible with one strategy must be rejected.
- **Fit** across activities creates sustainable advantage (three types: simple consistency, reinforcing, optimization of effort)

### Gleacher Game Application
- **Don't chase every market segment** — pick a strategic position (e.g., law enforcement only, or cancer premium only)
- **Make trade-offs explicit**: If you go premium (Automated Cell, high-WTP markets), you can't also serve low-WTP markets
- **Activity fit matters**: Your factory technology, product design, pricing, and advertising must all reinforce one strategy
- **Avoid competitive convergence**: If all 8 teams copy each other's pricing, margins collapse (Bertrand)
        """)

    with st.expander("**2. Can You Say What Your Strategy Is? — Collis & Rukstad (HBR 2008)**", expanded=False):
        st.markdown("""
### Core Thesis
Most executives cannot articulate their company's strategy in 35 words or fewer. A good strategy statement has three components:
**Objective + Scope + Advantage**.

### The Framework
**1. Objective** — Ends (measurable, time-bound goal)
- Example: "Grow to $1B revenue by 2030" (not "maximize shareholder value")

**2. Scope** — Where you compete
- Customer segment
- Geography
- Vertical integration (how much of value chain)

**3. Competitive Advantage** — Why customers choose you
- Customer value proposition
- Unique activities that enable it

### Hierarchy of Company Statements
- **Mission** (why we exist) → **Values** (what we believe) → **Vision** (what we want to be) → **Strategy** (how we win) → **Balanced scorecard** (implementation)

### Gleacher Game Application
- **Write your team's strategy statement in 35 words before the game starts**
- Example: "We will achieve $10M cash by Year 4 by dominating the high-margin law enforcement market in 3 regions through premium-designed narcotic monitors manufactured via Production Line technology."
- Every decision should reinforce the strategy — if debating a trade deal, ask: "does this fit our 35-word strategy?"
        """)

    with st.expander("**3. Note on Competitive Positioning**", expanded=False):
        st.markdown("""
### Porter's Three Generic Strategies
Firms must choose one — trying to straddle leads to being "stuck in the middle."

**1. Cost Leadership** — lowest cost producer
- Tight cost controls, efficient scale facilities, minimization of overhead
- Works when: price-sensitive customers, commoditized product, large market share possible

**2. Differentiation** — unique product dimensions valued by customers
- Brand, design, technology, customer service, dealer network
- Works when: customers value uniqueness, willing to pay premium, imitation is difficult

**3. Focus** — narrow market segment with either cost or differentiation advantage
- Geographic, demographic, product-use focus
- Works when: segment has distinct preferences, broad players underserving, size of segment justifies

### Value Chain Analysis
Every firm performs a collection of activities. Competitive advantage comes from how these activities are linked.
- **Primary activities:** Inbound logistics → Operations → Outbound logistics → Marketing → Service
- **Support activities:** Firm infrastructure, HR, technology development, procurement

### Gleacher Game Application
- **Cost leadership** = Automated Cell, minimal product features, low prices, single market
- **Differentiation** = premium features, multiple hormones/toxicology, higher WTP markets
- **Focus** = pick 1-2 market segments (e.g., Law + Military only) and dominate
- Map your **value chain** — which activities give you advantage? Production tech? Shipping network? Product design?
        """)

    with st.expander("**4. Note on the Structural Analysis of Industries — Porter's 5 Forces**", expanded=False):
        st.markdown("""
### The Five Forces That Shape Industry Competition

**1. Threat of New Entrants**
- Entry barriers: economies of scale, capital requirements, brand identity, switching costs, distribution access, govt policy
- **Game relevance:** DC build cost ($2.6M), factory build cost, learning curve on product design

**2. Bargaining Power of Suppliers**
- High when: few suppliers, no substitutes, buyer is unimportant customer, differentiated inputs
- **Game relevance:** Materials cost is fixed $100/unit — no supplier power variation

**3. Bargaining Power of Buyers**
- High when: concentrated buyers, standardized products, low switching costs, price-sensitive
- **Game relevance:** In the Gleacher Game, retail customers have low power (atomistic), but other teams buying from you in Trading Game have moderate power

**4. Threat of Substitutes**
- Substitutes limit industry pricing power
- **Game relevance:** Your product competes against "not buying at all" (customer reservation price)

**5. Rivalry Among Existing Competitors**
- High when: many/similar competitors, slow growth, high fixed costs, exit barriers, low differentiation
- **Game relevance:** 8 similar teams, slow-growing market (Bass model caps at M), high fixed costs (factory depreciation) → intense rivalry

### Strategic Implication
Industry profitability depends on these 5 forces. Defend against forces or shift them in your favor.

### Gleacher Game Application
- You're in a HIGH-rivalry industry (8 identical teams)
- **Defensive moves:** product differentiation, geographic focus, long-term trade agreements, brand/advertising
- **Shift the industry:** first-mover in automated cell, lock in exclusive markets via product features
        """)

    with st.expander("**5. Creating Competitive Advantage**", expanded=False):
        st.markdown("""
### What Creates Competitive Advantage?
A firm has competitive advantage when it creates **more economic value** than rivals.
**Economic Value = Willingness to Pay (WTP) − Supplier Opportunity Cost**

### The Value-Price-Cost Framework
```
WTP (customer willingness to pay)
 ↓ captured by customer as consumer surplus
Price (what customer pays)
 ↓ captured by firm as producer surplus
Cost (firm's costs)
 ↓ captured by suppliers
Supplier Opportunity Cost
```
Firms can gain advantage by:
1. **Lowering cost** (below rivals)
2. **Raising WTP** (above rivals through differentiation)
3. **Both simultaneously** (dual advantage — rare)

### VRIN Resources
For a resource to create sustainable advantage, it must be:
- **V**aluable (customer values it)
- **R**are (rivals don't have it)
- **I**nimitable (hard to copy)
- **N**on-substitutable (no alternatives)

### Gleacher Game Application
- **Your specialty product** (in the Trading Game) has rare/inimitable quality — your region is protected
- **Raising WTP:** Add more features to products (at cost of design $/day and materials $/unit)
- **Lowering cost:** Scale up production (Cobb-Douglas shows increasing returns with labor)
- **Dual advantage:** Only achievable via superior product design AND efficient production technology
        """)

    st.markdown("---")

    # ═══ MARKET DYNAMICS (3 documents) ═══
    st.markdown("## 🌐 Market Dynamics")

    with st.expander("**6. Choosing Corporate and Global Scope**", expanded=False):
        st.markdown("""
### Three Scope Decisions

**1. Horizontal Scope** — How broad a product/service range?
- Related diversification → share resources, leverage capabilities
- Unrelated diversification → risk mitigation, financial synergies only

**2. Vertical Scope** — How much of the value chain to own?
- Transaction cost economics: make vs. buy depends on asset specificity, uncertainty, frequency
- Vertical integration reduces transaction costs but adds complexity

**3. Geographic Scope** — How many countries/regions?
- Entry modes: export, license, JV, wholly-owned subsidiary
- Trade-offs: local responsiveness vs. global integration

### Corporate Advantage
Parent must add value beyond what business units could achieve standalone.
- **Portfolio management:** Cash allocation across BUs
- **Restructuring:** Turn around underperformers
- **Transferring skills:** Share capabilities across units
- **Sharing activities:** Joint operations, distribution

### Gleacher Game Application
- **Horizontal**: Design multiple products (cancer + heart + fertility) → diversification via features
- **Vertical**: You own factory + DC — consider building more DCs vs. selling wholesale to other teams (make vs. buy)
- **Geographic**: Expand DCs to 2-3 regions? NPV depends on 4-year horizon and demand Bass curve timing
- **Synergy check**: Do new markets share factory capacity? (Yes — factory produces any developed design)
        """)

    with st.expander("**7. The Pros and Cons of Entering a Market — Chevalier (FT 1999)**", expanded=False):
        st.markdown("""
### When to Enter a Market
Entry creates value if:
1. Market is growing and profitable
2. Entry barriers are surmountable
3. Incumbents will not retaliate aggressively
4. Your cost or differentiation advantage is sustainable

### Assessing Incumbent Retaliation
Incumbents retaliate aggressively when:
- They have **high sunk costs** (can't exit) → fight to defend share
- Entrant is **small** (cheap to drive out)
- **Excess capacity** exists (marginal cost pricing possible)
- **Reputation at stake** (other markets watching)

### Types of Entry
- **De novo** (build from scratch)
- **Acquisition** (buy existing player)
- **Joint venture / alliance** (shared risk)

### Timing Considerations
- **First-mover advantages:** learning curve, network effects, customer switching costs
- **First-mover disadvantages:** R&D costs, market education, technology uncertainty
- **Fast-follower:** learn from pioneer's mistakes, wait for technology to stabilize

### Gleacher Game Application
- **Entering a new region (DC build) = market entry**: Does the region already have competitors? What's their capacity?
- **Product design = product line entry**: Adding a new feature creates a "new market"
- **First-mover in automated cell**: Huge capex commitment ($3M+), but signals permanence and scales with volume
- **Retaliation**: Be cautious about entering regions where competitors have excess capacity — they'll drop prices
        """)

    with st.expander("**8. When It Can Be Good to Burn Your Boats — Chevalier (FT 1999)**", expanded=False):
        st.markdown("""
### Core Thesis
**Commitment creates competitive advantage through credibility.** By making irreversible investments, you signal to rivals that you won't back down — which can deter their investment.

### The Logic
- **Reversible investments** → rivals may doubt your resolve
- **Sunk costs + capacity** → signal "we're here to stay, compete with us at your peril"
- In game theory: **credible commitment** changes the equilibrium

### Historic Examples
- Cortés burning his ships at Vera Cruz (1519) to prevent his men from retreating
- Airbus's huge capital commitments to A380 to deter Boeing from building competitor
- Retail chains overbuilding capacity to deter entry

### When to Commit
- When industry has **winner-take-most** dynamics
- When you have **superior execution capability**
- When rival's **best response** to your commitment is to back off

### When Not to Commit
- Uncertain demand
- Volatile technology
- Weak balance sheet (can't weather war of attrition)

### Gleacher Game Application
- **Building an Automated Cell factory ($3M+ capex)** = burning your boats
  - Signals you're serious about scale
  - Irrevocable — cannot recover capex if demand disappoints
  - Makes competitors think twice before entering your region
- **Multi-region DC expansion** = commitment to national presence
- **Product design investment** ($135K+ for Temporal heartbeat) = commitment to that feature set
- **Warning**: Don't burn boats without demand certainty — with 4-year horizon and Bass uncertainty, wait for signals before huge capex
        """)

    with st.expander("**9. The Dynamics of Price Competition — Garicano & Gertner (FT 1999)**", expanded=False):
        st.markdown("""
### How Price Wars Start
Price wars emerge from:
1. **Oversupply** (too much capacity chasing too little demand)
2. **Perishable inventory** (use-it-or-lose-it)
3. **Customer heterogeneity** (some willing to shop, others loyal)
4. **Signaling errors** (misreading competitor's pricing as aggression)

### Price War Dynamics
- **Bertrand competition**: When products are identical, price → marginal cost (zero economic profit)
- **Tacit coordination** is fragile — small shocks can trigger defection cascade
- **Asymmetric firms** (different costs) lead to more stable pricing: low-cost firm sets price

### How to Avoid Price Wars
1. **Differentiate** — make direct price comparison impossible
2. **Multi-market contact** — punish defectors across markets
3. **Meet-the-competition clauses** — signal matching willingness
4. **Volume discounts** — make pricing less transparent
5. **Communicate non-aggressively** via industry channels

### How to End a Price War
- **Recognize** you're in one (many don't)
- **Signal willingness to cooperate** without explicit collusion
- **Differentiate quickly** to exit commoditization
- **Exit the market** if rivals have sustainable cost advantage

### Gleacher Game Application
- **With 8 similar teams, you're AT RISK of price war every day**
- **Defensive moves:**
  - Product differentiation via unique feature combinations
  - Multi-region presence (deter price cuts in your home)
  - Wholesale/trade relationships (give competitors skin in the game)
- **If a price war starts:**
  - Don't retaliate immediately — may be noise
  - Consider **inventory/demand signals** before reacting
  - If prolonged — exit the commoditizing market, focus on high-WTP segments
        """)

    st.markdown("---")

    # ═══ FINANCE (3 documents) ═══
    st.markdown("## 💰 Finance")

    with st.expander("**10. Capital Allocation — Morgan Stanley (Dec 2022)**", expanded=False):
        st.markdown("""
### The Five Capital Allocation Choices
Every dollar of free cash flow must be allocated to one of:

**1. Mergers & Acquisitions (M&A)**
- Should be evaluated on NPV basis — beware overpayment
- Synergy estimates often overstated
- Success rate: 50-70% of deals destroy value

**2. Capital Expenditures (CAPEX)**
- Maintenance capex (replace depreciation) vs. growth capex
- Hurdle rate: must exceed WACC for value creation
- Watch for capital cycle — overinvest during booms, underinvest in downturns

**3. Research & Development (R&D)**
- Long-duration investments with uncertain payoff
- Firms with high R&D intensity should have longer-term orientation

**4. Share Buybacks**
- Good when: stock undervalued, excess cash, limited growth options
- Bad when: using debt to fund buybacks for short-term EPS boost
- Signals management's view of intrinsic value

**5. Dividends**
- Signals stability and cash generation
- Sticky — cutting dividends is painful, seen as distress signal

### The Ranking Matters
**Best to worst (generally):**
1. High-NPV organic growth (CAPEX, R&D)
2. Accretive M&A with clear synergies
3. Share buybacks below intrinsic value
4. Dividends
5. Dilutive M&A, buybacks above value

### Gleacher Game Application
**Your capital allocation choices:**
- **Organic CAPEX**: Build new factory / DC / upgrade technology
- **R&D**: Product design (use focus groups, develop features)
- **Debt vs Equity**: Issue bonds (10-25% APR) vs. retain earnings
- **Dividends**: 6.5% after-tax return on dividends paid — small positive adjustment to return to investors

**Rule of thumb:** Only invest if NPV > 0 at 15% cost of capital AND payback within remaining game horizon.
        """)

    with st.expander("**11. Financial Statement Analysis**", expanded=False):
        st.markdown("""
### The Three Statements
**Income Statement** — Performance over a period
- Revenue → Gross Profit → Operating Income (EBIT) → Net Income

**Balance Sheet** — Position at a moment
- Assets = Liabilities + Equity
- Working capital = Current assets - Current liabilities

**Cash Flow Statement** — Reconciles net income to cash
- Operating + Investing + Financing = Change in cash

### Key Ratios to Monitor

**Profitability:**
- Gross Margin = (Revenue - COGS) / Revenue
- Operating Margin = EBIT / Revenue
- Net Margin = Net Income / Revenue
- ROA = Net Income / Total Assets
- ROE = Net Income / Shareholders' Equity

**Efficiency:**
- Inventory Turnover = COGS / Inventory
- Days Inventory Outstanding = 365 / Inventory Turnover
- Days Sales Outstanding (DSO) = AR / (Revenue / 365)
- Asset Turnover = Revenue / Total Assets

**Liquidity:**
- Current Ratio = Current Assets / Current Liabilities (> 1.0 is healthy)
- Quick Ratio = (Cash + AR) / Current Liabilities
- Cash Ratio = Cash / Current Liabilities

**Leverage:**
- Debt/Equity = Total Debt / Equity
- Interest Coverage = EBIT / Interest Expense (>= 7× for Good rating in Gleacher)

### DuPont Decomposition
**ROE = Net Margin × Asset Turnover × Financial Leverage**
= (Net Income/Revenue) × (Revenue/Assets) × (Assets/Equity)

### Gleacher Game Application
- **Monitor your quarterly income statement** — see which markets drive profit
- **Watch interest coverage** — Gleacher uses this for bond rating (20× = Excellent, 7× = Good, 2× = Poor)
- **DSO management** — Law market has 90 days, Medical has 30 days — affects working capital
- **Current ratio > 1.0** always — avoid emergency loans at 40% APR
        """)

    with st.expander("**12. Note on Adjusted Present Value (APV)**", expanded=False):
        st.markdown("""
### APV vs. WACC Valuation
Two methods for valuing a leveraged firm:

**WACC Method:**
- Discount unlevered FCF at WACC
- WACC reflects benefit of debt (via lower tax-adjusted rate)
- **Assumes constant capital structure**

**APV Method:**
- Step 1: Value as if all-equity (discount at unlevered cost of capital)
- Step 2: Add PV of interest tax shields separately
- **Formula: APV = NPV(all-equity) + PV(tax shield)**

### When APV is Better
- When capital structure changes over time (LBOs, project finance)
- When the firm has significant net operating losses
- When debt is tied to specific projects

### The Tax Shield Value
- Interest expense is tax-deductible
- PV of tax shield = Debt × Tax rate (Modigliani-Miller)
- In reality: tax shield depends on ability to use deductions, risk of financial distress

### Practical Steps
1. Calculate unlevered FCF
2. Find unlevered cost of equity (remove debt effect from beta)
3. Discount FCF at unlevered cost → get unlevered value
4. Calculate PV of tax shields (often at debt's cost, not unlevered)
5. Sum: APV = Unlevered Value + PV(Tax Shield)

### Gleacher Game Application
- **Valuing your firm for the final project report** — use APV
- **Unlevered value** = PV of operating cash flows at 15% (asset cost of capital)
- **Tax shield** = PV of (interest × 35%) — straightforward with bonds
- **Capital structure changes** as you issue/retire bonds — APV handles this cleanly
- **Distress risk** is real with emergency loans at 40% — apply a distress discount in final year projection
        """)

    st.markdown("---")

    # ═══ OPERATIONS (2 documents) ═══
    st.markdown("## ⚙️ Operations")

    with st.expander("**13. Managing Inventories: The Newsvendor Model**", expanded=False):
        st.markdown("""
### The Core Problem
How much inventory to order when:
- Demand is uncertain
- Overages (too much inventory) cost money
- Underages (stockouts) cost money
- Only one order is possible before demand realizes

### The Critical Fractile Formula
Optimal order quantity Q* satisfies:

> **P(demand ≤ Q*) = Cu / (Cu + Co)**

Where:
- **Cu** = underage cost (lost margin per unsold unit) = Price − Cost
- **Co** = overage cost (cost per unsold unit) = Cost − Salvage Value
- **P(demand ≤ Q*)** = service level

### Interpretation
- If Cu >> Co → order a lot (high service level, say 95%)
- If Co >> Cu → order little (low service level)

### Extension: Normal Demand
If demand ~ N(μ, σ), then Q* = μ + z × σ
- z is the z-score for the critical fractile
- z = 1.65 for 95% service level
- z = 1.96 for 97.5%
- z = 2.33 for 99%

### Service Level Decision
- **Higher SL = more safety stock = more holding cost but fewer stockouts**
- **Optimal SL = Cu / (Cu + Co)**
- Example: margin=$200, cost=$100, salvage=$0 → SL = 200/(200+100) = 67%

### Gleacher Game Application
- **Reorder point** is the newsvendor extension to continuous review:
  - Reorder Point = μ_demand_during_lead_time + z × σ_demand_during_lead_time
  - This is exactly what the Trial War Room calculates!
- **End-of-game wind-down**: Newsvendor problem with salvage = $0
  - At game end, all inventory obsolete → Co = full cost
  - Cu = foregone margin
  - Optimal order: minimal buffer in last lead-time window
- **Fire-sale pricing**: If you have excess inventory late-game, drop price to move units before Co realizes
        """)

    with st.expander("**14. Note on Production Inventories**", expanded=False):
        st.markdown("""
### Types of Inventory
1. **Raw materials** — inputs waiting to be processed
2. **Work-in-process (WIP)** — partially completed goods
3. **Finished goods** — ready for sale

### Why Hold Inventory? (Four Reasons)
1. **Pipeline (transit) inventory** — needed due to lead time
2. **Cycle stock** — economies of scale in ordering/producing
3. **Safety stock** — buffer against demand/supply uncertainty
4. **Anticipation inventory** — planned for future demand peaks

### The EOQ (Economic Order Quantity) Model
When demand is constant and known, optimal order size balances:
- Ordering cost (fixed cost per order)
- Holding cost (per unit per period)

**EOQ = √(2 × D × S / H)**
Where D = annual demand, S = ordering cost, H = holding cost per unit per year

### Little's Law
**Inventory = Throughput × Cycle Time**

Corollaries:
- Want lower inventory? Reduce cycle time
- Want higher throughput? Relax inventory constraint

### Key Cost Trade-offs
- **Ordering frequency vs. batch size** (EOQ)
- **Safety stock vs. stockout cost** (newsvendor)
- **Cycle time vs. inventory levels** (Little's Law)
- **Inventory vs. capacity** (add capacity to reduce cycle time)

### Inventory Performance Metrics
- **Inventory Turnover** = COGS / Avg Inventory (higher = more efficient)
- **Days of Supply** = Inventory / Daily Demand
- **Fill Rate** = Orders fulfilled from stock / Total orders

### Gleacher Game Application
- **Pipeline inventory**: WIP + in-transit is real inventory — your system tracks this
- **Cycle stock**: Batch size of 100 units (changeable) — balances setup cost vs holding
- **Safety stock**: Use newsvendor formula — already built into the Trial War Room
- **Little's Law**: Factory throughput × cycle time = WIP inventory
  - Cobb-Douglas gives you throughput
  - Cycle time = batch time + setup
  - WIP = throughput × cycle time
- **Inventory costs in Gleacher**: No explicit holding cost in game, BUT obsolescence at game end IS the ultimate holding cost
        """)

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # INTEGRATED GAME PLAYBOOK
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("Integrated Game Playbook — Apply All 14 Readings")

    st.markdown("""
<div style="background:linear-gradient(135deg,#800000,#b22222);color:white;
    border-radius:12px;padding:1.5rem;margin:1rem 0;">
<h4 style="color:#ffd700;margin-top:0;">Your 35-Word Strategy Template (Collis & Rukstad)</h4>
<p style="font-style:italic;opacity:0.9;">"We will [OBJECTIVE: measurable goal by Year X]
by [SCOPE: which markets, which products, which regions]
through [ADVANTAGE: how you win — cost? differentiation? focus?]."</p>
<p style="margin-bottom:0;"><b>Example for Team Panem:</b><br>
"We will reach $15M cash by end of Year 4 by dominating the high-margin Law Enforcement
market in 3 regions through premium narcotic monitors produced via Automated Cell technology."</p>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
### Decision Checklist — Run Every Major Move Through This

| Decision | Key Reading | Test |
|---|---|---|
| Market selection | Competitive Positioning, 5 Forces | Is this a focused position with low rivalry? |
| Entering new region | Entry Decisions | Will incumbents retaliate? Can we win? |
| Product design | Creating Competitive Advantage | Does it raise WTP vs rivals? VRIN? |
| Factory tech choice | Production Inventories, Burn Boats | Does scale justify commitment? |
| Pricing | Price Competition | Avoid Bertrand — differentiate |
| Reorder point | Newsvendor | z × √(Np(1-p)L) at chosen service level |
| CAPEX decision | Capital Allocation | NPV > 0 at 15%? Payback in horizon? |
| Debt issuance | Financial Statements | Coverage ratio for Excellent rating? |
| End-game wind-down | Newsvendor (Co = full cost) | Fire-sale before obsolescence |
| Final valuation | APV | Unlevered FCF + PV(tax shield) |
    """)

    st.info("""
**The capstone insight:** These 14 readings are not separate topics — they're lenses on the same problem.
Every game decision touches strategy + finance + operations + market dynamics simultaneously.
That's why ISM exists — to force you to integrate.
    """)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5: CAPSTONE PREP HUB (now in Misc)
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
