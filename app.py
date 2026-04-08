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
            "🎮 ISM War Room",
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

    # ── KPI Cheat Sheet ───────────────────────────────────────────────────────
    st.subheader("Critical Game Parameters")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Batch Size", "100 units")
    p2.metric("Material Cost", "$100/unit")
    p3.metric("Production Cycle", "2.5 days")
    p4.metric("Factory→DC Transit", "1 day")

    p5, p6, p7, p8 = st.columns(4)
    p5.metric("Customer Arrival", "0.01% x Mkt/day")
    p6.metric("Emergency Loan APR", "40%", delta="-Avoid at all costs", delta_color="inverse")
    p7.metric("Cash Interest", "3% APR")
    p8.metric("Tax Rate", "35% quarterly")

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
                               annotation_text=f"Profit-max: ${profit_max_price:,.0f}")
        fig_pricing.add_vline(x=optimal_price, line_dash="dot", line_color="#800000",
                               annotation_text=f"Revenue-max: ${optimal_price:,.0f}")
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
        st.markdown("""
        **From Operations Management (Little's Law, Newsvendor):**

        | Parameter | Hormone | Specialty |
        |---|---|---|
        | Market size | 300,000 | 140,000 |
        | Arrival rate | 30 customers/day | 14 customers/day |
        | Production cycle | 2.5 days | 2.5 days |
        | Transit to DC | 1 day | 1 day |
        | **Lead time** | **3.5 days** | **3.5 days** |

        **Reorder Point = Daily Demand x Lead Time + Safety Stock**

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
        lead_time = 6.0 if calc_both else 3.5
        daily_demand_calc = 0.0001 * calc_market * (calc_max_wtp - calc_price) / calc_max_wtp if calc_max_wtp > calc_price else 0
        safety_stock = 50
        reorder = daily_demand_calc * lead_time + safety_stock
        with rc3:
            st.metric("Daily Demand", f"{daily_demand_calc:,.1f} units")
            st.metric("Lead Time", f"{lead_time} days")
            st.metric("Recommended Reorder Point", f"{reorder:,.0f} units")

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

    TEAMS = {
        "B612": {"id": "1-25", "color": "#6b3fa0",
                  "members": ["Masa Ishigaki", "Takeshi Tanaka", "Hyeyoung Lee",
                              "Valerii Egorov", "Carlos Naibryf", "Fengshu Jin"],
                  "theme": "The Little Prince"},
        "Dune": {"id": "1-26", "color": "#b8860b",
                  "members": ["Jo Hayes", "Jenny Yang", "Morry Mori",
                              "Suliya Suliya", "Betty Wang", "Kosuke Okura"],
                  "theme": "Dune"},
        "Globex": {"id": "1-27", "color": "#c44e00",
                    "members": ["Prashanth Palepu", "Lisa Lau", "George Chia",
                                "Jeffrey Chen", "Kacey Du", "Lambert Xu"],
                    "theme": "The Simpsons"},
        "Gotham": {"id": "1-28", "color": "#333333",
                    "members": ["Inge Supatra", "Bryan Wong", "Delphine Terrien",
                                "Benjamin Jiang", "Dai Kato", "Ngiap Seng Khoo"],
                    "theme": "Batman / Gotham"},
        "Panem": {"id": "1-29", "color": "#800000",
                   "members": ["Chris Ma", "Shiyuan Tian", "Yohei Nakadate",
                               "Derrick Teo", "Jack Meng", "Jason Weng"],
                   "theme": "The Hunger Games",
                   "is_us": True},
        "Vulcan": {"id": "1-31", "color": "#1a3c5e",
                    "members": ["Eric Zhang", "Jony Hu", "Laurence Zhu",
                                "Yehwan Kim", "Tom Hsieh", "Sudeep Rathee"],
                    "theme": "Star Trek"},
        "Westeros": {"id": "1-32", "color": "#2d6a2e",
                      "members": ["Kosuke Shinohara", "Ken Ng", "Taku Yasuda",
                                  "Ryan Kim", "Andy Yoo", "Jumpei Maruyama"],
                      "theme": "Game of Thrones"},
        "Zion": {"id": "1-33", "color": "#0e7c7b",
                  "members": ["Ken Chew", "Ryo Aikawa", "Louis Woenardi",
                              "Dimas Purnama", "Chris Kwan", "Alex Wang"],
                  "theme": "The Matrix"},
    }

    # Team overview cards
    team_cols = st.columns(4)
    for i, (team_name, team_data) in enumerate(TEAMS.items()):
        col = team_cols[i % 4]
        is_us = team_data.get("is_us", False)
        border = "3px solid #ffd700" if is_us else "1px solid rgba(255,255,255,0.15)"
        badge = " (US)" if is_us else ""
        with col:
            members_html = "<br>".join(team_data["members"])
            st.markdown(f"""
            <div style="background: {team_data['color']}; color: white; border-radius: 10px;
                padding: 1rem; margin-bottom: 0.8rem; border: {border}; min-height: 220px;">
                <h4 style="color: white; margin: 0 0 0.2rem 0;">{team_name}{badge}</h4>
                <p style="margin: 0 0 0.5rem 0; font-size: 0.75rem; opacity: 0.7;">
                    Seat {team_data['id']} | {team_data['theme']}</p>
                <p style="margin: 0; font-size: 0.82rem; line-height: 1.5;">{members_html}</p>
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
