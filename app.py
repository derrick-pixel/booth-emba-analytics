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
            "⚔️ Trial War Room",
            "🎮 ISM War Room",
            "📊 Learning Dashboard",
            "🕸️ Knowledge Graph",
            "📈 Content Analytics",
            "🎯 Capstone Prep Hub",
        ],
        index=0,
    )
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
        {"Day": "Friday Apr 17", "Class": "Wrap-Up",
         "Game (7-9pm)": "—", "Assignment": "A6 + Final Project due May 3"},
    ]
    st.dataframe(pd.DataFrame(schedule_data), use_container_width=True, hide_index=True)

    # ── Grading Weights ───────────────────────────────────────────────────────
    grade_col1, grade_col2 = st.columns([1, 2])
    with grade_col1:
        st.subheader("Grading Breakdown")
        grade_data = {
            "Component": ["Assignments 1-3 (Group)", "Assignments 4-5 (Group)", "Assignment 6 (Individual)",
                          "Final Project (Group)", "Participation", "Game Performance"],
            "Points": [30, 10, 10, 30, 10, 10],
        }
        grade_df = pd.DataFrame(grade_data)
        fig_grade = px.pie(grade_df, values="Points", names="Component",
                           color_discrete_sequence=["#800000", "#b22222", "#cd5c5c",
                                                     "#1a3c5e", "#2d6a2e", "#b8860b"],
                           hole=0.4)
        fig_grade.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_grade, use_container_width=True)

    with grade_col2:
        st.subheader("Game Performance = 10% of grade")
        st.markdown("""
        The **final project** (30%) is the biggest single component — it requires:
        - Summary & analysis of your decisions and outcomes during the simulation
        - A projection for the future
        - A **valuation of the business**

        **Key insight:** Game Performance is only 10%, but the Final Project (30%) depends on
        how well you played. Playing well = better story to tell = higher combined score (40%).
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

elif page == "⚔️ Trial War Room":
    st.markdown('<p class="big-header">Trial War Room</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Live calculators for the Monopoly & Trading Game trial (Apr 12-13)</p>',
                unsafe_allow_html=True)
    st.markdown("")

    # ── Game Parameters (all adjustable) ─────────────────────────────────────
    st.subheader("Game Parameters")
    st.caption("Adjust these to match the current game scenario — all sections below recalculate automatically")

    gp_col1, gp_col2, gp_col3, gp_col4, gp_col5 = st.columns(5)
    with gp_col1:
        MAX_WTP = st.number_input("Max WTP ($)", value=500, step=50, key="gp_maxwtp")
    with gp_col2:
        MC_PRODUCTION = st.number_input("Materials Cost ($/unit)", value=100, step=10, key="gp_mc")
    with gp_col3:
        BATCH_SIZE = st.number_input("Batch Size (units)", value=100, step=10, key="gp_batch")
    with gp_col4:
        PRODUCTION_DAYS = st.number_input("Production Time (days)", value=2.5, step=0.5, key="gp_proddays")
    with gp_col5:
        ARRIVAL_RATE = st.number_input("Arrival Rate", value=0.0001, step=0.00001, format="%.5f", key="gp_arrival")

    gp2_col1, gp2_col2, gp2_col3, gp2_col4 = st.columns(4)
    with gp2_col1:
        HORMONE_MARKET = st.number_input("Hormone Market Size", value=300000, step=10000, key="gp_hmkt")
    with gp2_col2:
        SPECIALTY_MARKET = st.number_input("Specialty Market Size", value=140000, step=10000, key="gp_smkt")
    with gp2_col3:
        BASE_LEAD_TIME = st.number_input("Lead Time (days, single product)", value=3.5, step=0.5, key="gp_lt")
    with gp2_col4:
        COST_OF_CAPITAL = st.number_input("Cost of Capital (%)", value=10.0, step=1.0, key="gp_coc")

    CAPACITY_PER_DAY = BATCH_SIZE / PRODUCTION_DAYS if PRODUCTION_DAYS > 0 else 40

    SHIPPING = {
        "Same region": {"cost_per_unit": 0, "days": 1, "label": "Free, 1 day"},
        "Mail (between regions)": {"cost_per_unit": 40, "days": 3, "label": "$40/unit, 3 days"},
        "Container (between regions)": {"cost_per_unit": 10, "days": 21, "label": "$10/unit, 21 days"},
    }

    # ── Dynamic banner (recalculates from inputs above) ──────────────────────
    _opt_home = (MAX_WTP + MC_PRODUCTION) / 2
    _opt_ship = (MAX_WTP + MC_PRODUCTION + 40) / 2
    _h_demand = ARRIVAL_RATE * HORMONE_MARKET * (MAX_WTP - _opt_home) / MAX_WTP if MAX_WTP > _opt_home else 0
    _s_demand = ARRIVAL_RATE * SPECIALTY_MARKET * (MAX_WTP - _opt_home) / MAX_WTP if MAX_WTP > _opt_home else 0
    _excess = CAPACITY_PER_DAY - _h_demand - _s_demand

    st.markdown(f"""
<div style="background:linear-gradient(135deg,#800000,#b22222);color:white;
    border-radius:12px;padding:1.2rem 1.5rem;margin-bottom:1rem;">
<h4 style="color:#ffd700;margin:0 0 0.5rem 0;">Derived Metrics (auto-calculated)</h4>
<div style="display:flex;gap:2rem;flex-wrap:wrap;">
<div><span style="opacity:0.7;">Optimal Home Retail</span><br><b style="font-size:1.3rem;">${_opt_home:.0f}</b><br><span style="font-size:0.75rem;opacity:0.6;">({MAX_WTP}+{MC_PRODUCTION})/2</span></div>
<div><span style="opacity:0.7;">Optimal w/ Mail Ship</span><br><b style="font-size:1.3rem;">${_opt_ship:.0f}</b><br><span style="font-size:0.75rem;opacity:0.6;">({MAX_WTP}+{MC_PRODUCTION}+40)/2</span></div>
<div><span style="opacity:0.7;">Hormone Demand</span><br><b style="font-size:1.3rem;">{_h_demand:.1f} /day</b><br><span style="font-size:0.75rem;opacity:0.6;">{ARRIVAL_RATE*HORMONE_MARKET:.0f} × ({MAX_WTP}-{_opt_home:.0f})/{MAX_WTP}</span></div>
<div><span style="opacity:0.7;">Specialty Demand</span><br><b style="font-size:1.3rem;">{_s_demand:.1f} /day</b><br><span style="font-size:0.75rem;opacity:0.6;">{ARRIVAL_RATE*SPECIALTY_MARKET:.0f} × ({MAX_WTP}-{_opt_home:.0f})/{MAX_WTP}</span></div>
<div><span style="opacity:0.7;">Factory Capacity</span><br><b style="font-size:1.3rem;">{CAPACITY_PER_DAY:.0f} /day</b><br><span style="font-size:0.75rem;opacity:0.6;">{BATCH_SIZE} / {PRODUCTION_DAYS} days</span></div>
<div><span style="opacity:0.7;">Excess Capacity</span><br><b style="font-size:1.3rem;">{_excess:.1f} /day</b><br><span style="font-size:0.75rem;opacity:0.6;">{CAPACITY_PER_DAY:.0f} - {_h_demand:.1f} - {_s_demand:.1f}</span></div>
</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 1: PRICING OPTIMIZER
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("1. Retail Pricing Optimizer")
    st.caption("Monopoly pricing with uniform WTP distribution [0, MaxWTP] — uses game parameters above")

    pr_col1, pr_col2 = st.columns([1, 2])
    with pr_col1:
        pr_product = st.radio("Product", ["Hormone", "Specialty"], horizontal=True, key="pr_prod")
        pr_market = HORMONE_MARKET if pr_product == "Hormone" else SPECIALTY_MARKET
        pr_arrival = ARRIVAL_RATE * pr_market

        # Optimal price
        pr_optimal = (MAX_WTP + MC_PRODUCTION) / 2
        pr_demand_opt = pr_arrival * (MAX_WTP - pr_optimal) / MAX_WTP if MAX_WTP > pr_optimal else 0
        pr_profit_opt = pr_demand_opt * (pr_optimal - MC_PRODUCTION)

        st.markdown("---")
        st.metric("Optimal Retail Price", f"${pr_optimal:,.0f}")
        st.metric("Daily Demand", f"{pr_demand_opt:,.1f} units")
        st.metric("Daily Profit", f"${pr_profit_opt:,.0f}")
        st.metric("Daily Revenue", f"${pr_demand_opt * pr_optimal:,.0f}")

    with pr_col2:
        price_range = np.arange(MC_PRODUCTION, MAX_WTP, max(1, (MAX_WTP - MC_PRODUCTION) // 80))
        demand_arr = pr_arrival * (MAX_WTP - price_range) / MAX_WTP
        revenue_arr = demand_arr * price_range
        profit_arr = demand_arr * (price_range - MC_PRODUCTION)

        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(x=price_range, y=profit_arr,
                                     name="Daily Profit", line=dict(color="#2d6a2e", width=3)))
        fig_pr.add_trace(go.Scatter(x=price_range, y=revenue_arr,
                                     name="Daily Revenue", line=dict(color="#800000", width=2, dash="dash")))
        fig_pr.add_trace(go.Scatter(x=price_range, y=demand_arr * MC_PRODUCTION,
                                     name="Daily COGS", line=dict(color="#999", width=1, dash="dot")))
        fig_pr.add_vline(x=pr_optimal, line_dash="dash", line_color="#2d6a2e",
                          annotation_text=f"Optimal: ${pr_optimal:,.0f}",
                          annotation_position="top left")
        fig_pr.update_layout(height=400, xaxis_title="Retail Price ($)", yaxis_title="$ per day",
                              margin=dict(l=0, r=0, t=30, b=0), yaxis_tickformat="$,.0f")
        st.plotly_chart(fig_pr, use_container_width=True)

        # Dynamic price sensitivity table
        st.markdown("**Price Sensitivity Table**")
        step = max(25, int((MAX_WTP - MC_PRODUCTION) / 10 / 25) * 25)
        price_points = list(range(int(MC_PRODUCTION) + step, int(MAX_WTP), step))
        sens_data = []
        for p in price_points:
            d = pr_arrival * (MAX_WTP - p) / MAX_WTP
            r = d * p
            prof = d * (p - MC_PRODUCTION)
            sens_data.append({
                "Price": f"${p}",
                "P(buy)": f"{(MAX_WTP - p) / MAX_WTP:.1%}",
                "Demand/day": f"{d:.1f}",
                "Revenue/day": f"${r:,.0f}",
                "Profit/day": f"${prof:,.0f}",
                "vs Optimal": f"{(prof / pr_profit_opt - 1) * 100:+.1f}%" if pr_profit_opt > 0 else "—",
            })
        st.dataframe(pd.DataFrame(sens_data), use_container_width=True, hide_index=True)

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2: TRADE DEAL EVALUATOR (Double Marginalization)
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("2. Trade Deal Evaluator")
    st.caption("Evaluate wholesale deals — watch for double marginalization!")

    td_col1, td_col2, td_col3 = st.columns(3)
    with td_col1:
        st.markdown("**Deal Parameters**")
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
    buyer_optimal_retail = (MAX_WTP + buyer_mc) / 2
    buyer_demand = td_buyer_arrival * (MAX_WTP - buyer_optimal_retail) / MAX_WTP
    buyer_daily_profit = buyer_demand * (buyer_optimal_retail - buyer_mc)

    # Supply chain optimal (if vertically integrated)
    sc_optimal_retail = (MAX_WTP + seller_total_cost) / 2
    sc_demand = td_buyer_arrival * (MAX_WTP - sc_optimal_retail) / MAX_WTP
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
        ws_range = np.arange(seller_total_cost + 10, MAX_WTP * 0.6, 10)
        seller_profits = []
        buyer_profits = []
        sc_profits = []
        for w in ws_range:
            b_retail = (MAX_WTP + w) / 2
            b_demand = td_buyer_arrival * (MAX_WTP - b_retail) / MAX_WTP
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
    _opt_default = int((MAX_WTP + MC_PRODUCTION) / 2)
    gs_col1, gs_col2, gs_col3, gs_col4 = st.columns(4)
    with gs_col1:
        gs_hormone_price = st.number_input("Hormone Retail Price ($)", value=_opt_default, step=25, key="gs_p1")
    with gs_col2:
        gs_specialty_price = st.number_input("Specialty Retail Price ($)", value=_opt_default, step=25, key="gs_p2")
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
    p1_demand = ARRIVAL_RATE * p1_market * (MAX_WTP - gs_hormone_price) / MAX_WTP if gs_hormone_price < MAX_WTP else 0
    p2_demand = ARRIVAL_RATE * p2_market * (MAX_WTP - gs_specialty_price) / MAX_WTP if gs_specialty_price < MAX_WTP else 0
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

    inv_lead = BASE_LEAD_TIME * 2 - 1 if both_running else BASE_LEAD_TIME  # 6.0 if both, 3.5 if single
    inv_N = ARRIVAL_RATE * inv_market  # arrivals per day
    inv_p = (MAX_WTP - inv_price) / MAX_WTP if inv_price < MAX_WTP else 0

    inv_col1, inv_col2, inv_col3 = st.columns(3)
    with inv_col1:
        st.markdown(f"**{inv_product}** at **${inv_price}**")
        inv_service = st.selectbox("Service Level", [90, 95, 97.5, 99], index=1,
                                    format_func=lambda x: f"{x}%", key="inv_svc")
        inv_z = {90: 1.28, 95: 1.65, 97.5: 1.96, 99: 2.33}[inv_service]

        # Proper safety stock: z × √(Np(1-p)L)
        inv_demand_lt = inv_daily_demand * inv_lead
        inv_std_lt = (inv_N * inv_p * (1 - inv_p) * inv_lead) ** 0.5
        inv_safety = inv_z * inv_std_lt
        inv_reorder = inv_demand_lt + inv_safety

        st.metric("Daily Demand (Np)", f"{inv_daily_demand:.1f} units")
        st.metric("Lead Time", f"{inv_lead} days" + (" (both running)" if both_running else ""))
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
    eg_demand = ARRIVAL_RATE * eg_mkt * (MAX_WTP - eg_price) / MAX_WTP if eg_price < MAX_WTP else 0
    eg_units_sellable = eg_demand * gs_days_left
    eg_surplus = eg_total_inv - eg_units_sellable

    with eg_col2:
        st.metric("Daily Demand at Current Price", f"{eg_demand:.1f} units ({eg_product} @ ${eg_price})")
        st.metric("Units Sellable in {0} Days".format(gs_days_left), f"{eg_units_sellable:.0f}")
        if eg_surplus > 0:
            st.metric("Surplus (will be wasted)", f"{eg_surplus:.0f} units",
                       delta=f"-${eg_surplus * MC_PRODUCTION:,.0f} wasted", delta_color="inverse")
            needed_prob = eg_total_inv / (ARRIVAL_RATE * eg_mkt * gs_days_left) if gs_days_left > 0 else 1
            fire_sale = max(MC_PRODUCTION, MAX_WTP * (1 - needed_prob))
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
