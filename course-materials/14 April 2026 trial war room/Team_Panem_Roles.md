# Team Panem (1-29) — Gleacher Game Role Playbook

**Real game starts: Wednesday, April 15 2026**
Expands the D2 "Suggested Team Roles" handout with Class 3 Practice Game learnings.
Companion app: https://github.com/derrick-pixel/booth-emba-analytics → "14 New War Room" page.

---

## Week-1 strategic priorities (everyone reads this)

1. **$549K → $600K puzzle.** We start with only $549K cash. Line factory = $100K land + $500K capex = $600K min. Until we close that gap, **no focus groups, no new products**. Just run the pilot Bench hard. This is Assignment 4. (Class 3 slide 85.)
2. **Don't advertise until Line is under construction.** Capacity = 9 u/day on the pilot. Advertising creates demand we can't fulfill; stockouts permanently break the imitator flywheel. (Class 3 slide 50.)
3. **First-mover compounds through imitators.** `q = 0.0035` is ~17× the innovator rate. Imitators only come to OUR store and scale with OUR share. Every day we wait, a competitor's flywheel is spinning instead of ours.
4. **All finance math is 364-day compounding.** Not 365. EAR = (1 + APR/364)^364 − 1.
5. **Left hand must know right hand.** Finance locks the ad budget; Production tells Pricing the cost floor; Inventory tells Production when to shut down Bench. Miscommunication is how teams lose.

---

## 1. Pricing, Product Development & Advertising *(CMO-style)*

**Lead:** Derrick Teo
**Partner:** Jack Meng
**Backup:** Jason Weng

### Core mission
Set prices that maximize contribution margin given WTP, competitor prices, and demand mechanics. Decide what products to launch and when. Own the advertising dial.

### Daily responsibilities
- **Price setting, reviewed per quarter.** Default to **value-based pricing** (anchored on mean/max WTP from focus groups), not cost-plus. Check price against the competitive cliff in §14 of the app — the point where innovators flip to a competitor.
- **Pricing modes to rotate through:**
  - **Skimming** for new premium products (start high, move down as WTP distribution is explored). Works when our product has features competitors don't — e.g., +GPS in Clinical Cardiovascular.
  - **Penetration** when we need imitator flywheel acceleration or to block a competitor's entry. Accept thin margin early for compounding share.
  - **Competitive/parity pricing** for commodity segments where innovators shop on price. Aim for just-under the lowest competitor.
  - **NEVER cost-based pricing.** The game rewards WTP extraction. If WTP is $700, we charge $700-ish — the cost side only sets our walk-away floor.
- **Focus groups continually, but not on day 1.** Each focus group is ~$10K and a few days. Run them when we need WTP data for a new market, not for sport. Assignment 4 note: don't run any until Line is under construction.
- **Advertising budget**, co-owned with Finance:
  - Rule of thumb: $500/day = +1p of innovator arrivals.
  - **Never flip ads on before capacity exists to fulfill.** Use app §15 (Ad ROI + Capacity Gate) to find the ad ceiling at current throughput.
  - Stockouts don't just waste ad dollars — they **kill the imitator flywheel for the rest of the game**. Irreversible.
- **Product launch sequencing.** Confirm new product can be profitably built on our Line factory BEFORE scheduling development. Use app §8 (Product Design ROI) to screen.

### Key numbers to memorize
- Commission: **20% flat** (retailer pays, comes off our revenue)
- Handling: $10/unit (retailer pays)
- Mail shipping in-region: $50/unit (we pay)
- WTP distribution: Normal, with mean from focus group, σ ≈ mean/10 or $30 (Heart baseline)

### Hand-offs
- **From Finance:** quarterly ad budget envelope.
- **From Production:** unit cost floor and max daily throughput.
- **To Shipping Agreements:** price list per product per region.
- **To Competitor Analysis:** what competitor prices we need intel on.

---

## 2. Finance *(CFO)*

**Lead:** Chris Ma
**Backup:** Yohei Nakadate

### Core mission
Keep the company solvent, fund growth at the lowest cost of capital, never trigger the 40% emergency loan. Own the cap table and the tax clock.

### Daily responsibilities
- **Cash runway tracking.** Daily check: how many days of opex does current cash cover? Flag to team when runway drops below 60 days.
- **Debt issuance.** Once we have a full quarter of positive EBIT, issue bonds — 5-year zero-coupon, daily compounding over 364 days. Tiers (app §12):
  - **Excellent (10% APR, 20× coverage)** — always issue first, cheapest capital.
  - **Good (15% APR, 7× coverage)** — second tranche if we need more.
  - **Poor (25% APR, 2× coverage)** — last resort, still cheaper than 40% emergency.
  - Total capacity = f(quarterly EBIT). At $25K QEBIT, total ≈ $230K PV / 639 bonds.
- **Capital allocation.** Reserve funds for:
  1. **Line factory $600K** (week 1-2 priority).
  2. **DC expansion**: $100K land + $2.5M capex + $2K/day opex + fulfillment ($10 + 20% of retail). Only after Line is producing.
  3. **Tax reserve** — quarterly tax = 35% × quarterly operating income. Never let cash drop below (tax due + 1 month opex).
- **DSO management.** Trade credit terms: materials net/30, others net/15 is the default. Every day of extra DSO is a day of free working capital; every day shorter accelerates cash in but may hurt partnership terms. Review DSO quarterly with Shipping Agreements lead.
- **Working capital ratio.** Keep ≥ 1.0 at all times. Below 1.0 = technically insolvent.
- **Dividends.** Not relevant until year 2+. Park this.

### Hand-offs
- **To Pricing/Ads:** quarterly ad budget envelope + capital constraint for product launches.
- **To Production:** approved capex and timing for new factories/DCs.
- **From Shipping Agreements:** DSO impact per agreement before signing.

---

## 3. Production *(COO — Production)*

**Lead:** Shiyuan Tian
**Backup:** Chris Ma + Yohei Nakadate

### Core mission
Produce units at the lowest unit cost given the Cobb-Douglas technology constraints. Decide when to upgrade factory tech. Run the labor/capital dials without ruining anything.

### Daily responsibilities
- **Factory technology decisions** (app §11):
  - **Bench** (A=0.009, α=0.10, β=0.85): pilot factory. Keep running until Line is online.
  - **Line** (A=0.010, α=0.30, β=0.75, K_min=$500K): workhorse. Buy as soon as cash ≥ $600K. Schedule old Bench to close ~10 days after Line opens. Clone shipping agreements to the new Line factory.
  - **Cell** (A=0.020, α=0.80, β=0.30, K_min=$3M): only for year 2+ when we have steady demand >50 u/day and $3M to burn.
- **K and L tuning:**
  - Only add INCREMENTAL capital to a factory (capex is sticky).
  - Labor daily expenditure — never set to $1 or near $1. That ruins the factory. Set to $0 only to shut down.
  - Use app §11 to find the cheapest tech at current K/L point; re-check after any capex change.
- **Batch size decisions.** Setup penalty per batch: Bench 0.05d · Line 0.50d · Cell 1.0d. Bigger batches = less setup drag but more WIP. Default batch 100u; raise to 200u on Line when ad spend ramps.
- **Production priorities.** Coordinate with Inventory to set reorder points >0 for active agreements; -1 to suspend without cancelling.

### Hand-offs
- **To Pricing:** cost-per-unit at current K/L (this sets the walk-away floor).
- **From Finance:** capex approval for new Line / DC.
- **To Inventory:** daily throughput estimate; scheduled shutdowns.

---

## 4. Inventory *(COO — Logistics)*

**Lead:** Yohei Nakadate
**Backup:** Shiyuan Tian

### Core mission
Right units, right place, right time. Zero stockouts on priority products. Zero inventory rot.

### Daily responsibilities
- **Reorder points (ROP) per DC per product.** Set > 0 for active agreements. -1 to suspend. Review whenever demand pattern changes (new ad campaign, competitor entry, new product launch).
- **Production batches** — coordinate with Production on batch sizing. Smaller batches = more setup drag but leaner inventory; bigger batches = more WIP cash tied up.
- **Shipping mode: mail vs container** (app cost parameter section):
  - Mail: $50/unit in-region (flat per unit).
  - Container: fixed cost per container with capacity.
  - **Breakeven rule:** switch from mail to container when volume/container ≥ container_cost / $50. Below that, mail wins.
- **Stockout monitoring.** Watch "Inventory Status" report in HQ daily. A stockout doesn't just lose today's sales — it breaks the imitator flywheel, which compounds for years.
- **Competitor inventory tracking.** Infer competitor inventory from their daily sales velocity. If their DC retail sales drop but their ads are still on, they're stocked out — opportunity for us. Coordinate with Competitor Analysis.

### Hand-offs
- **To Production:** "produce more of X, less of Y" signals based on ROP hits.
- **To Shipping Agreements:** shipping mode choice affects per-unit cost in agreements.
- **From Competitor Analysis:** competitor stockout intel = pricing opportunity.

---

## 5. Shipping Agreements *(BD / Partnerships)*

**Lead:** Derrick Teo
**Partner:** Jason Weng
**Backup:** Jack Meng

### Core mission
Every shipping agreement is a contract with price, terms, batch, and DSO. Make sure each one is CM-positive and fits our cash cycle.

### Daily responsibilities
- **Pre-negotiation:** agree terms in chat or in-person BEFORE sending formal agreements. The game gives a chat envelope per team — use it.
- **Include WTP info in chat comments** when sending agreements to cut negotiation time. If they see our WTP data and agree on price, they're unlikely to walk.
- **DSO awareness.** Every agreement has implicit/explicit payment terms. Shorter DSO = faster cash, but may lose partners. Longer DSO = slower cash but stickier partnership. Default: materials net/30, others net/15. **Never agree to terms that break working capital ratio.**
- **CM waterfall check** before signing any agreement (app §7 Market Segment Analyzer shows this):
  ```
  Price − Commission(20%) − Handling($10) − Shipping($10-50) − Materials − Mfg Overhead = CM/unit
  ```
  If CM/unit < 0, decline. No exceptions.
- **Priority management.** Set priority on each agreement so our DCs get served first when factory is capacity-constrained.
- **Reorder points on agreements.** >0 to keep active, -1 to suspend cleanly, 0 effectively kills the agreement.
- **SA cloning.** When we build a new factory, clone existing shipping agreements to the new factory so we don't recreate them manually.

### Hand-offs
- **From Pricing:** price list and CM floor per product.
- **To Finance:** DSO impact of each new agreement (for cash flow projection).
- **From Inventory:** shipping mode (mail vs container) affects per-unit cost.

---

## 6. Competitor Analysis *(CSO)*

**Lead:** Jason Weng

### Core mission
Know every other team's product portfolio, price list, and capacity trajectory. Feed intel to Pricing and Inventory within hours, not days.

### Daily responsibilities
- **Price surveillance.** Every morning, record each competitor's retail price per product per DC. Flag any change ≥$25 to Pricing immediately.
- **Product portfolio tracking.** What markets is each team in? Are they launching new products? Use the Product Line Architecture report.
- **Capacity/facilities tracking.** From the map: who's building a new factory? Which tech? When will it open? (Build Duration: 90 days for factory, 60 days for DC.) Flag when competitor Line capacity is ~30 days from opening — that's when we need to have our pricing war plan ready.
- **Advertising detection.** Sudden revenue jumps with stable inventory = ad campaign live. Match spend patterns to their cash constraints.
- **Inventory inference.** Combine with Inventory team: falling on-hand + stable sales velocity + no new production = incoming stockout. Window to raise our price or seize their market share.
- **Home-region map.** 8 teams, 10 regions. Serenity (military) and Metropolis (large) have NO home team — they're up for grabs.

### Hand-offs
- **To Pricing:** daily competitor price changes, ad campaigns, stockouts.
- **To Inventory:** competitor stockout windows (we should push volume).
- **To Shipping Agreements:** which teams have excess capacity (potential wholesale partners).

---

## Daily operating cadence

| Time | Who | What |
|------|-----|------|
| Start of session | All | 10-min standup: cash, EBIT, stockouts, competitor moves |
| During play | Pricing | Reprice daily based on competitor data from Jason |
| During play | Production + Inventory | Run factory dials, adjust ROPs per agreement |
| During play | Finance | Monitor cash runway, flag emergency-loan risk |
| End of quarter | Finance | Tax reserve, debt issuance if Q EBIT>0, report to team |
| End of quarter | All | 10-min retro: what worked, what to change |

---

## Escalation / tie-breaking

- **Pricing vs Production conflict** (e.g., Pricing wants to lower price but CM would go negative): Pricing owns it but must run new numbers with Production's cost floor. If still stuck → CEO (tbd, or consensus vote).
- **Finance vs Pricing on ad budget:** Finance has veto (we can't spend what we don't have).
- **Inventory vs Production on batch size:** Inventory owns it (they see stockouts first), Production delivers.
- **Anyone can pull the emergency brake** on a risky decision. Losing the game is worse than blocking a tactical move.

---

## Quick-reference: key numbers (memorize)

| Item | Value | Source |
|------|-------|--------|
| Starting cash (real game) | **$549K** | Class 3 slide 85 |
| Line factory total cost | **$600K** ($100K land + $500K capex) | Class 3 slide 35 |
| DC total cost | ~$2.6M + $2K/day + fulfillment | Class 3 slide 43 |
| Sales commission | **20%** (paid by retailer) | Class 3 slide 53 |
| Handling | $10/unit (paid by retailer) | Class 3 slide 53 |
| Shipping mail in-region | $50/unit (we pay) | Class 3 slide 53 |
| Innovator rate p | 0.0002 | Class 3 slide 51 |
| Imitator rate q | 0.0035 | Class 3 slide 51 |
| Ad: +1p cost | $500/day | Class 3 slide 47 |
| Emergency loan | 40% APR (AVOID) | Gleacher tips |
| Excellent bond | 10% APR / 20× coverage | Class 3 slide 59 |
| Good bond | 15% APR / 7× coverage | Class 3 slide 59 |
| Poor bond | 25% APR / 2× coverage | Class 3 slide 59 |
| Tax rate | 35% × quarterly op income | Game standard |
| Compounding | **364 days/year** (not 365) | Class 3 slide 65 |
| Quarter-end days | Q1=91, Q4=364, Q8=728, Q12=1092 | Class 3 slide 53 |
| Practice horizon | Day 1092 (Q12) | Class 3 slide 72 |

---

## Where in the app to find each tool

(App URL: our Streamlit dashboard → "14 New War Room" page)

- **Pricing / Product:** §7 Market Segment Analyzer · §8 Product Design ROI · §14 Competitive Innovator Split
- **Ads:** §15 Ad ROI + Capacity Gate
- **Flywheel argument (for debates):** §16 Market Share Flywheel
- **Production tech:** §11 Production Technology Picker
- **Finance:** §10 Cash & Tax Planner · §12 Debt Issuance Calculator · §13 Get-to-$600K Planner
- **Shipping agreements CM:** §9 Supply Chain Trade-Off Calculator

---

*Owner: Derrick Teo · Team Panem (1-29) · Drafted from D2 handout + Class 3 slides, April 14 2026*
