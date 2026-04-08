"""
Chicago Booth EMBA AXP-25 — Complete Course Data Model
Derrick Teo's program: Autumn 2024 – Spring 2026
"""

import pandas as pd
from datetime import date

# ── Program Timeline ──────────────────────────────────────────────────────────

PROGRAM_START = date(2024, 9, 15)
PROGRAM_END = date(2026, 6, 6)

QUARTERS = [
    {
        "quarter": "Autumn 2024",
        "label": "Kick Off",
        "week1_start": "2024-09-15", "week1_end": "2024-09-21",
        "week2_start": None, "week2_end": None,
        "location": "Chicago",
        "color": "#800000",
    },
    {
        "quarter": "Autumn 2024 (Intl)",
        "label": "International Sessions",
        "week1_start": "2024-10-13", "week1_end": "2024-11-02",
        "week2_start": "2024-11-17", "week2_end": "2024-12-14",
        "location": "London & Hong Kong",
        "color": "#800000",
    },
    {
        "quarter": "Winter 2025",
        "label": "Winter Q1",
        "week1_start": "2025-01-20", "week1_end": "2025-01-25",
        "week2_start": "2025-02-24", "week2_end": "2025-03-01",
        "location": "Chicago",
        "color": "#1a3c5e",
    },
    {
        "quarter": "Spring 2025",
        "label": "Spring Q1",
        "week1_start": "2025-04-07", "week1_end": "2025-04-12",
        "week2_start": "2025-05-26", "week2_end": "2025-05-31",
        "location": "Chicago",
        "color": "#2d6a2e",
    },
    {
        "quarter": "Summer 2025",
        "label": "Summer",
        "week1_start": "2025-07-07", "week1_end": "2025-07-12",
        "week2_start": None, "week2_end": None,
        "location": "Chicago",
        "color": "#b8860b",
    },
    {
        "quarter": "Electives 2025",
        "label": "Electives",
        "week1_start": "2025-08-03", "week1_end": "2025-08-16",
        "week2_start": None, "week2_end": None,
        "location": "Chicago (Harper Center)",
        "color": "#6b3fa0",
    },
    {
        "quarter": "Autumn 2025",
        "label": "Autumn Q2",
        "week1_start": "2025-09-21", "week1_end": "2025-09-27",
        "week2_start": "2025-11-03", "week2_end": "2025-11-08",
        "location": "Chicago",
        "color": "#800000",
    },
    {
        "quarter": "Winter 2026",
        "label": "Winter Q2",
        "week1_start": "2026-01-05", "week1_end": "2026-01-10",
        "week2_start": "2026-02-02", "week2_end": "2026-02-07",
        "location": "Chicago",
        "color": "#1a3c5e",
    },
    {
        "quarter": "Spring 2026",
        "label": "Spring Q2 (Final)",
        "week1_start": "2026-03-16", "week1_end": "2026-03-21",
        "week2_start": "2026-04-12", "week2_end": "2026-04-18",
        "location": "Chicago",
        "color": "#2d6a2e",
    },
]

MILESTONES = [
    {"date": "2024-09-15", "event": "Program Kick Off", "icon": "🎓"},
    {"date": "2024-10-13", "event": "International Sessions Begin", "icon": "🌏"},
    {"date": "2025-08-03", "event": "Electives at Harper Center", "icon": "📚"},
    {"date": "2026-04-18", "event": "Hong Kong Closing Celebration", "icon": "🎉"},
    {"date": "2026-06-04", "event": "Chicago Graduation Events", "icon": "🎊"},
    {"date": "2026-06-06", "event": "Graduation Day", "icon": "🎓"},
]

# ── Courses ───────────────────────────────────────────────────────────────────

COURSES = [
    {
        "code": "38815", "name": "Managerial Psychology",
        "units": 50, "quarter": "Autumn 2024",
        "category": "Leadership & Behavior",
        "professor": "TBD",
        "weeks": "Kick Off Week",
        "sessions": 5,
        "key_topics": [
            "Self-awareness", "Motivation", "Team dynamics",
            "Influence", "Leadership styles"
        ],
        "frameworks": ["Myers-Briggs", "Emotional Intelligence", "Situational Leadership"],
        "case_studies": [],
    },
    {
        "code": "31102", "name": "Analytical Methods for the MBA",
        "units": 0, "quarter": "Autumn 2024",
        "category": "Quantitative",
        "professor": "TBD",
        "weeks": "Kick Off Week",
        "sessions": 3,
        "key_topics": ["Quantitative foundations", "Mathematical tools", "Analytical reasoning"],
        "frameworks": ["Optimization", "Linear algebra basics"],
        "case_studies": [],
    },
    {
        "code": "33801", "name": "Microeconomics",
        "units": 100, "quarter": "Autumn 2024 (Intl)",
        "category": "Economics",
        "professor": "Various",
        "weeks": "Week 1 & 2 (International)",
        "sessions": 9,
        "key_topics": [
            "Supply & demand", "Consumer theory", "Producer theory",
            "Market structures", "Game theory", "Oligopoly",
            "Externalities", "Public goods", "Information asymmetry"
        ],
        "frameworks": [
            "Nash Equilibrium", "Price elasticity", "Marginal analysis",
            "Pareto efficiency", "Moral hazard", "Adverse selection",
            "Prisoner's dilemma", "Bertrand competition", "Cournot competition"
        ],
        "case_studies": ["Group assignments 1-6"],
    },
    {
        "code": "41800", "name": "Statistics",
        "units": 100, "quarter": "Autumn 2024 (Intl)",
        "category": "Quantitative",
        "professor": "TBD",
        "weeks": "Week 1 & 2 (International)",
        "sessions": 9,
        "key_topics": [
            "Probability", "Distributions", "Hypothesis testing",
            "Regression analysis", "Confidence intervals",
            "Statistical inference", "Correlation"
        ],
        "frameworks": [
            "Central Limit Theorem", "OLS Regression", "p-values",
            "Bayes' theorem", "t-tests", "Chi-square tests"
        ],
        "case_studies": [],
    },
    {
        "code": "30800", "name": "Financial Accounting",
        "units": 100, "quarter": "Winter 2025",
        "category": "Finance & Accounting",
        "professor": "TBD",
        "weeks": "Week 1 & 2",
        "sessions": 9,
        "key_topics": [
            "Balance sheet", "Income statement", "Cash flow statement",
            "Revenue recognition", "Depreciation & amortization",
            "Leases", "Inventory accounting", "Financial ratios",
            "Earnings quality"
        ],
        "frameworks": [
            "Double-entry bookkeeping", "GAAP", "IFRS",
            "DuPont analysis", "Accrual vs cash accounting",
            "Operating vs finance leases"
        ],
        "case_studies": ["TA Review Sessions 1-4"],
    },
    {
        "code": "42800", "name": "Competitive Strategy",
        "units": 100, "quarter": "Winter 2025",
        "category": "Strategy",
        "professor": "Luis Garicano",
        "weeks": "Week 1 & 2",
        "sessions": 9,
        "key_topics": [
            "Industry analysis", "Value creation & capture",
            "Competitive advantage", "Entry deterrence",
            "Vertical chain & bargaining", "Network effects",
            "Standards wars", "Innovation & disruption",
            "Scope & diversification", "Vertical integration & outsourcing"
        ],
        "frameworks": [
            "Porter's Five Forces", "Value-based strategy",
            "Entry barriers", "Bertrand price competition",
            "Technology S-curve", "Disruptive innovation",
            "Core competence", "Make vs buy",
            "Economies of scope", "Winner-take-all markets"
        ],
        "case_studies": ["Tesla", "Netflix", "Google Android", "Newell", "AI industry analysis"],
    },
    {
        "code": "37800", "name": "Marketing Management",
        "units": 100, "quarter": "Spring 2025",
        "category": "Marketing",
        "professor": "TBD",
        "weeks": "Week 1 & 2",
        "sessions": 9,
        "key_topics": [
            "Segmentation", "Targeting", "Positioning",
            "Brand strategy", "Viral engineering", "Pricing in marketing",
            "Price discrimination", "Advertising effectiveness",
            "Product line decisions", "Competitor analysis",
            "Company strategy"
        ],
        "frameworks": [
            "STP framework", "4Ps of marketing", "Customer lifetime value",
            "Conjoint analysis", "Positioning maps",
            "Price segmentation", "Ad ROI measurement"
        ],
        "case_studies": ["Unilever", "Trump Targeting", "Coca-Cola vending machine", "Tock/restaurants"],
    },
    {
        "code": "35801", "name": "Corporate Finance",
        "units": 100, "quarter": "Spring 2025",
        "category": "Finance & Accounting",
        "professor": "Pietro Veronesi",
        "weeks": "Week 1 & 2",
        "sessions": 9,
        "key_topics": [
            "Present value", "Stock valuation", "Portfolio diversification",
            "CAPM", "Bond valuation", "Discount rates",
            "APV & WACC", "Multiples valuation", "Modigliani-Miller",
            "Capital structure", "Bankruptcy costs", "Dividends & buybacks",
            "ESG & finance"
        ],
        "frameworks": [
            "DCF valuation", "CAPM", "APV", "WACC",
            "Modigliani-Miller theorem", "Trade-off theory",
            "Pecking order theory", "Efficient frontier",
            "Beta & systematic risk", "Cost of capital estimation"
        ],
        "case_studies": [
            "Beta Management Co.", "Ameritrade cost of capital",
            "Snap IPO valuation", "Marriott Corp", "Coronavirus crisis pricing"
        ],
    },
    {
        "code": "37802", "name": "Pricing Strategies",
        "units": 50, "quarter": "Summer 2025",
        "category": "Marketing",
        "professor": "Jean-Pierre Dube",
        "weeks": "Week 1",
        "sessions": 5,
        "key_topics": [
            "Economic value to customer (EVC)", "Price elasticity",
            "Cost-based pricing", "Competitive pricing",
            "Segmented pricing", "Nonlinear pricing",
            "Bundling & metering", "Customer lifetime value pricing",
            "Inflation pricing"
        ],
        "frameworks": [
            "EVC analysis", "Inverse elasticity rule",
            "Competitor reaction simulation", "Price discrimination degrees",
            "Two-part tariffs", "Bundling strategy",
            "Razor-and-blades model", "Geo-conquesting"
        ],
        "case_studies": [
            "KONE Monospace", "ZipRecruiter", "Curled Metal Inc.",
            "Beauregard Textile", "Keurig", "Nashua Photo"
        ],
    },
    {
        "code": "38803", "name": "Negotiations",
        "units": 50, "quarter": "Summer 2025",
        "category": "Leadership & Behavior",
        "professor": "TBD",
        "weeks": "Week 1",
        "sessions": 5,
        "key_topics": [
            "BATNA", "Reservation prices", "Zone of possible agreement",
            "Credible commitments", "Interest-based negotiation",
            "Joint gains & value creation", "Coalition building",
            "Multi-party negotiations"
        ],
        "frameworks": [
            "BATNA analysis", "ZOPA", "Integrative bargaining",
            "Distributive bargaining", "Credible commitment mechanisms",
            "Coalition formation", "Logrolling",
            "Anchoring in negotiations"
        ],
        "case_studies": ["3-way Pluto", "Strategic Advice exercises 1-5"],
    },
    {
        "code": "33860", "name": "Macroeconomics and the Global Environment",
        "units": 100, "quarter": "Autumn 2025",
        "category": "Economics",
        "professor": "TBD",
        "weeks": "Week 1 & 2",
        "sessions": 9,
        "key_topics": [
            "GDP & national accounts", "Monetary policy",
            "Fiscal policy", "Inflation", "Exchange rates",
            "International trade", "Business cycles",
            "Central banking", "Global financial crises"
        ],
        "frameworks": [
            "IS-LM model", "Phillips curve", "Taylor rule",
            "Mundell-Fleming", "Purchasing power parity",
            "Aggregate supply-demand", "Quantity theory of money"
        ],
        "case_studies": [],
    },
    {
        "code": "30801", "name": "Managerial Accounting and Analysis",
        "units": 50, "quarter": "Autumn 2025",
        "category": "Finance & Accounting",
        "professor": "TBD",
        "weeks": "Week 1",
        "sessions": 5,
        "key_topics": [
            "Cost behavior & classification", "Activity-based costing",
            "Transfer pricing", "Variance analysis",
            "Performance measurement", "Balanced scorecard",
            "Divisional performance", "ESG metrics"
        ],
        "frameworks": [
            "ABC costing", "Time-driven ABC",
            "Flexible budgeting", "Variance decomposition",
            "Balanced scorecard", "EVA",
            "Cost-volume-profit analysis"
        ],
        "case_studies": [
            "RegionFly", "FinePrint Company", "ZPhone",
            "Bullwinkle", "J&M Divisional", "VMD"
        ],
    },
    {
        "code": "38802", "name": "Managerial Decision Making",
        "units": 50, "quarter": "Autumn 2025",
        "category": "Leadership & Behavior",
        "professor": "George Wu",
        "weeks": "Week 2",
        "sessions": 5,
        "key_topics": [
            "Decision making under uncertainty", "Heuristics & biases",
            "Judgment errors", "Risk perception",
            "Choice architecture", "Nudging",
            "Normative vs descriptive vs prescriptive"
        ],
        "frameworks": [
            "Expected utility theory", "Prospect theory",
            "Anchoring bias", "Availability heuristic",
            "Representativeness heuristic", "Confirmation bias",
            "Nudge theory", "Choice architecture",
            "System 1 vs System 2 thinking"
        ],
        "case_studies": ["John Brown case"],
    },
    {
        "code": "35802", "name": "Financial Strategy",
        "units": 100, "quarter": "Winter 2026",
        "category": "Finance & Accounting",
        "professor": "TBD",
        "weeks": "Week 1 & 2",
        "sessions": 9,
        "key_topics": [
            "Capital structure analysis", "Leverage ratios",
            "Debt ratings", "Debt capacity",
            "Mergers & acquisitions financing",
            "Corporate restructuring", "LBOs",
            "International capital structure"
        ],
        "frameworks": [
            "Capital structure optimization", "Debt rating models",
            "Steps in capital structure analysis",
            "LBO valuation", "Accretion/dilution analysis",
            "Leverage capacity assessment"
        ],
        "case_studies": [
            "MCI WorldCom", "FANUC", "Molycorp",
            "Roche", "Thule", "Lyondell Chemicals"
        ],
    },
    {
        "code": "40801", "name": "Operations Management",
        "units": 100, "quarter": "Winter 2026",
        "category": "Operations",
        "professor": "Rene Caldentey",
        "weeks": "Week 1 & 2",
        "sessions": 8,
        "key_topics": [
            "Process analysis", "Bottleneck identification",
            "Linear programming", "Service operations",
            "Cycle time management", "TQM & Lean",
            "Inventory management (EOQ, newsvendor)",
            "Bullwhip effect", "Supply chain postponement"
        ],
        "frameworks": [
            "Little's Law", "Bottleneck analysis",
            "Critical fractile / newsvendor model",
            "EOQ model", "Kanban / pull systems",
            "Lean / Toyota Production System",
            "Beer game / bullwhip", "Postponement strategy",
            "Cash-to-cash cycle"
        ],
        "case_studies": ["Beer game simulation", "Various industry examples"],
    },
    {
        "code": "32800", "name": "Business Analytics",
        "units": 50, "quarter": "Spring 2026",
        "category": "Quantitative",
        "professor": "TBD",
        "weeks": "Week 1",
        "sessions": 5,
        "key_topics": [
            "Data-driven decision making", "Predictive analytics",
            "Machine learning basics", "A/B testing",
            "Causal inference"
        ],
        "frameworks": [
            "Regression for prediction", "Decision trees",
            "Cross-validation", "Experimental design",
            "Causal vs correlational analysis"
        ],
        "case_studies": [],
    },
    {
        "code": "38830", "name": "Ethics in the Workplace",
        "units": 50, "quarter": "Spring 2026",
        "category": "Leadership & Behavior",
        "professor": "TBD",
        "weeks": "Week 1",
        "sessions": 5,
        "key_topics": [
            "Ethical frameworks", "Stakeholder theory",
            "Corporate responsibility", "Ethical dilemmas",
            "Whistleblowing", "Culture & ethics"
        ],
        "frameworks": [
            "Utilitarianism", "Deontological ethics",
            "Virtue ethics", "Stakeholder analysis",
            "Ethical decision framework"
        ],
        "case_studies": [],
    },
    {
        "code": "42805", "name": "Integrated Strategic Management (Capstone)",
        "units": 50, "quarter": "Spring 2026",
        "category": "Strategy",
        "professor": "TBD",
        "weeks": "Week 2",
        "sessions": 5,
        "key_topics": [
            "Cross-functional integration", "Simulation-based learning",
            "Strategic decision making under uncertainty",
            "Market dynamics", "Financial planning",
            "Operations optimization", "Competitive response"
        ],
        "frameworks": [
            "Balanced scorecard (integrated)", "Scenario planning",
            "War gaming", "Cross-functional trade-offs",
            "Dynamic competitive strategy"
        ],
        "case_studies": ["Business simulation game"],
    },
    {
        "code": "33832", "name": "Organizations and Incentives",
        "units": 50, "quarter": "Spring 2026",
        "category": "Economics",
        "professor": "TBD",
        "weeks": "Week 2",
        "sessions": 5,
        "key_topics": [
            "Incentive design", "Principal-agent problems",
            "Organizational design", "Compensation structures",
            "Team incentives", "Performance measurement"
        ],
        "frameworks": [
            "Principal-agent model", "Moral hazard in organizations",
            "Tournament theory", "Multi-tasking problem",
            "Relational contracts"
        ],
        "case_studies": [],
    },
    {
        "code": "31800", "name": "LEAD (Leadership Exploration & Development)",
        "units": 0, "quarter": "All Quarters",
        "category": "Leadership & Behavior",
        "professor": "Various",
        "weeks": "Integrated throughout",
        "sessions": 20,
        "key_topics": [
            "Self-assessment", "360 feedback",
            "Executive coaching", "Personal leadership brand",
            "Global executive connect"
        ],
        "frameworks": [
            "Leadership competency model", "360-degree feedback",
            "Personal development planning"
        ],
        "case_studies": [],
    },
    {
        "code": "42813", "name": "Storytelling (Elective)",
        "units": 50, "quarter": "Electives 2025",
        "category": "Leadership & Behavior",
        "professor": "TBD",
        "weeks": "Electives Week",
        "sessions": 5,
        "key_topics": [
            "Narrative structure", "Public speaking",
            "Persuasion through story", "Corporate narratives",
            "Reputation management", "Amazon 6-page memo"
        ],
        "frameworks": [
            "Story arc (setup/conflict/resolution)", "Narrative economics",
            "Reputation management framework",
            "6-page narrative memo (Amazon)"
        ],
        "case_studies": ["Warren Buffett / Berkshire Hathaway", "Elon Musk"],
    },
]

# ── Categories & Colors ───────────────────────────────────────────────────────

CATEGORY_COLORS = {
    "Finance & Accounting": "#1a3c5e",
    "Strategy": "#800000",
    "Economics": "#2d6a2e",
    "Marketing": "#b8860b",
    "Leadership & Behavior": "#6b3fa0",
    "Quantitative": "#c44e00",
    "Operations": "#0e7c7b",
}

# ── Cross-Course Concept Connections (for Knowledge Graph) ────────────────────

CONCEPT_CONNECTIONS = [
    # Finance connections
    ("DCF valuation", "Present value", "Corporate Finance"),
    ("DCF valuation", "WACC", "Corporate Finance"),
    ("WACC", "CAPM", "Corporate Finance"),
    ("CAPM", "Beta & systematic risk", "Corporate Finance"),
    ("Capital structure optimization", "Modigliani-Miller theorem", "Financial Strategy → Corporate Finance"),
    ("Capital structure optimization", "Leverage ratios", "Financial Strategy"),
    ("Debt rating models", "Leverage ratios", "Financial Strategy"),
    ("LBO valuation", "DCF valuation", "Financial Strategy → Corporate Finance"),

    # Accounting connections
    ("DuPont analysis", "Financial ratios", "Financial Accounting"),
    ("Financial ratios", "Leverage ratios", "Financial Accounting → Financial Strategy"),
    ("ABC costing", "Cost-volume-profit analysis", "Managerial Accounting"),
    ("Variance decomposition", "Flexible budgeting", "Managerial Accounting"),
    ("Balanced scorecard", "Performance measurement", "Managerial Accounting → Capstone"),

    # Strategy connections
    ("Porter's Five Forces", "Entry barriers", "Competitive Strategy"),
    ("Entry barriers", "Bertrand price competition", "Competitive Strategy → Microeconomics"),
    ("Disruptive innovation", "Technology S-curve", "Competitive Strategy"),
    ("Core competence", "Make vs buy", "Competitive Strategy"),
    ("Economies of scope", "Make vs buy", "Competitive Strategy"),
    ("Value-based strategy", "EVC analysis", "Competitive Strategy → Pricing"),
    ("Dynamic competitive strategy", "Porter's Five Forces", "Capstone → Competitive Strategy"),

    # Marketing-Pricing connections
    ("STP framework", "Price segmentation", "Marketing → Pricing"),
    ("Customer lifetime value", "Razor-and-blades model", "Marketing → Pricing"),
    ("Price discrimination degrees", "Two-part tariffs", "Pricing"),
    ("EVC analysis", "Inverse elasticity rule", "Pricing"),
    ("Bundling strategy", "Two-part tariffs", "Pricing"),
    ("Competitor reaction simulation", "Nash Equilibrium", "Pricing → Microeconomics"),
    ("Ad ROI measurement", "Regression for prediction", "Marketing → Business Analytics"),

    # Economics connections
    ("Nash Equilibrium", "Prisoner's dilemma", "Microeconomics"),
    ("Nash Equilibrium", "Bertrand competition", "Microeconomics"),
    ("Moral hazard", "Principal-agent model", "Microeconomics → Orgs & Incentives"),
    ("Adverse selection", "Moral hazard", "Microeconomics"),
    ("Price elasticity", "Inverse elasticity rule", "Microeconomics → Pricing"),
    ("IS-LM model", "Monetary policy", "Macroeconomics"),
    ("Phillips curve", "Taylor rule", "Macroeconomics"),
    ("Principal-agent model", "Tournament theory", "Organizations & Incentives"),

    # Operations connections
    ("Little's Law", "Bottleneck analysis", "Operations Management"),
    ("EOQ model", "Critical fractile / newsvendor model", "Operations Management"),
    ("Lean / Toyota Production System", "Kanban / pull systems", "Operations Management"),
    ("Beer game / bullwhip", "Postponement strategy", "Operations Management"),
    ("Bottleneck analysis", "Cost-volume-profit analysis", "Operations → Managerial Accounting"),
    ("Cash-to-cash cycle", "Financial ratios", "Operations → Financial Accounting"),

    # Behavioral / Decision connections
    ("Prospect theory", "Expected utility theory", "Managerial Decision Making"),
    ("Anchoring bias", "Anchoring in negotiations", "Decision Making → Negotiations"),
    ("System 1 vs System 2 thinking", "Heuristics & biases", "Managerial Decision Making"),
    ("Nudge theory", "Choice architecture", "Managerial Decision Making"),
    ("BATNA analysis", "ZOPA", "Negotiations"),
    ("Integrative bargaining", "Logrolling", "Negotiations"),
    ("Coalition formation", "Nash Equilibrium", "Negotiations → Microeconomics"),

    # Cross-domain strategic connections
    ("Scenario planning", "Dynamic competitive strategy", "Capstone"),
    ("War gaming", "Competitor reaction simulation", "Capstone → Pricing"),
    ("Cross-functional trade-offs", "Balanced scorecard", "Capstone"),
    ("Narrative economics", "Prospect theory", "Storytelling → Decision Making"),
    ("Reputation management framework", "Stakeholder analysis", "Storytelling → Ethics"),
    ("Ethical decision framework", "Stakeholder analysis", "Ethics"),
    ("Stakeholder analysis", "Principal-agent model", "Ethics → Orgs & Incentives"),

    # Quantitative foundations
    ("OLS Regression", "Regression for prediction", "Statistics → Business Analytics"),
    ("Bayes' theorem", "Prospect theory", "Statistics → Decision Making"),
    ("Hypothesis testing", "A/B testing", "Statistics → Business Analytics"),
    ("Causal vs correlational analysis", "Experimental design", "Business Analytics"),
]

# ── Capstone Integration Map ──────────────────────────────────────────────────
# Maps simulation game decision areas to relevant course frameworks

CAPSTONE_MAP = {
    "Market Entry & Positioning": {
        "courses": ["Competitive Strategy", "Marketing Management", "Microeconomics"],
        "frameworks": [
            "Porter's Five Forces", "STP framework", "Entry barriers",
            "Value-based strategy", "Nash Equilibrium", "Positioning maps"
        ],
        "key_questions": [
            "Which segments to target?",
            "How to differentiate from competitors?",
            "What are the barriers to entry in this market?",
        ],
    },
    "Pricing Decisions": {
        "courses": ["Pricing Strategies", "Marketing Management", "Microeconomics"],
        "frameworks": [
            "EVC analysis", "Price elasticity", "Competitor reaction simulation",
            "Price discrimination degrees", "Bundling strategy", "Inverse elasticity rule"
        ],
        "key_questions": [
            "What is the customer's willingness to pay?",
            "How will competitors react to our pricing?",
            "Should we use segmented or uniform pricing?",
        ],
    },
    "Financial Planning & Valuation": {
        "courses": ["Corporate Finance", "Financial Accounting", "Financial Strategy"],
        "frameworks": [
            "DCF valuation", "WACC", "Capital structure optimization",
            "DuPont analysis", "Financial ratios", "Leverage ratios"
        ],
        "key_questions": [
            "What is our cost of capital?",
            "What is the optimal debt/equity mix?",
            "How do we evaluate investment opportunities (NPV)?",
        ],
    },
    "Operations & Supply Chain": {
        "courses": ["Operations Management", "Managerial Accounting"],
        "frameworks": [
            "Bottleneck analysis", "Little's Law", "EOQ model",
            "Lean / Toyota Production System", "ABC costing",
            "Cash-to-cash cycle"
        ],
        "key_questions": [
            "Where is the bottleneck in our process?",
            "How much inventory should we hold?",
            "How do we reduce cycle time?",
        ],
    },
    "Strategic Decision Making": {
        "courses": ["Managerial Decision Making", "Negotiations", "Competitive Strategy"],
        "frameworks": [
            "Prospect theory", "Anchoring bias", "BATNA analysis",
            "Scenario planning", "War gaming", "System 1 vs System 2 thinking"
        ],
        "key_questions": [
            "Are we falling prey to cognitive biases?",
            "What is our BATNA in this negotiation?",
            "What scenarios should we plan for?",
        ],
    },
    "Organizational Design & Incentives": {
        "courses": ["Organizations and Incentives", "Managerial Accounting", "Ethics in the Workplace"],
        "frameworks": [
            "Principal-agent model", "Balanced scorecard",
            "Variance decomposition", "Ethical decision framework",
            "Tournament theory"
        ],
        "key_questions": [
            "How should we structure incentives for our team?",
            "What KPIs should we track?",
            "Are there ethical considerations in our strategy?",
        ],
    },
    "Macro Environment & Risk": {
        "courses": ["Macroeconomics", "Corporate Finance", "Managerial Decision Making"],
        "frameworks": [
            "IS-LM model", "Exchange rate models",
            "Beta & systematic risk", "Scenario planning",
            "Expected utility theory"
        ],
        "key_questions": [
            "How do interest rates affect our strategy?",
            "What macro risks should we hedge against?",
            "How do exchange rates impact our international operations?",
        ],
    },
    "Communication & Stakeholder Management": {
        "courses": ["Storytelling", "Negotiations", "Ethics in the Workplace"],
        "frameworks": [
            "Story arc (setup/conflict/resolution)", "Reputation management framework",
            "BATNA analysis", "Stakeholder analysis",
            "6-page narrative memo (Amazon)"
        ],
        "key_questions": [
            "How do we communicate our strategy to stakeholders?",
            "What narrative frames our competitive position?",
            "How do we manage reputation risk?",
        ],
    },
}

# ── Helper functions ──────────────────────────────────────────────────────────

def get_courses_df():
    rows = []
    for c in COURSES:
        rows.append({
            "Code": c["code"],
            "Course": c["name"],
            "Units": c["units"],
            "Quarter": c["quarter"],
            "Category": c["category"],
            "Professor": c["professor"],
            "Sessions": c["sessions"],
            "Topics": len(c["key_topics"]),
            "Frameworks": len(c["frameworks"]),
            "Case Studies": len(c["case_studies"]),
        })
    return pd.DataFrame(rows)


def get_units_by_category():
    data = {}
    for c in COURSES:
        cat = c["category"]
        data[cat] = data.get(cat, 0) + c["units"]
    return data


def get_units_by_quarter():
    data = {}
    for c in COURSES:
        q = c["quarter"]
        data[q] = data.get(q, 0) + c["units"]
    return data


def get_all_frameworks():
    """Return list of (framework, course_name, category)."""
    result = []
    for c in COURSES:
        for f in c["frameworks"]:
            result.append((f, c["name"], c["category"]))
    return result


def get_all_topics():
    """Return list of (topic, course_name, category)."""
    result = []
    for c in COURSES:
        for t in c["key_topics"]:
            result.append((t, c["name"], c["category"]))
    return result


def search_frameworks(query: str):
    """Search frameworks and topics across all courses."""
    query_lower = query.lower()
    results = []
    for c in COURSES:
        for f in c["frameworks"]:
            if query_lower in f.lower():
                results.append({"match": f, "type": "Framework", "course": c["name"], "category": c["category"]})
        for t in c["key_topics"]:
            if query_lower in t.lower():
                results.append({"match": t, "type": "Topic", "course": c["name"], "category": c["category"]})
    return results
