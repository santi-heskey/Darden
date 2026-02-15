import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# ==================== SIMULATION PARAMETERS ====================
N_ITERATIONS = 10000
N_YEARS = 25
DISCOUNT_RATE = 0.08

# ==================== STATIC PARAMETERS ====================
# Costs and Expenses
INTANGIBLE_COST_PCT = 0.725  # 72.5% of total well cost
MONTHLY_OPERATING_COST = 1000  # $1k per month
ANNUAL_LEASE_EXPENSE = 10000  # $10k per year
INFLATION_FACTOR_WELL = 0.5  # Half of inflation rate for well expense inflation

# Tax Rates
FEDERAL_TAX_RATE = 0.24
STATE_TAX_RATE = 0.045
SEVERANCE_TAX_RATE = 0.034
COUNTY_TAX_RATE = 0.045
ROYALTY_RATE = 0.152344

# Price and Escalation
CURRENT_PRICE_MMBTU = 4.0  # $/MMBTU
PRICE_INCREASE_START_YEAR = 5  # Year 5 (index 4)

# Decline Rates by Year (static, before multiplier)
DECLINE_RATES = {
    1: 0.225,   # 22.5%
    2: 0.175,   # 17.5%
    3: 0.125,   # 12.5%
    4: 0.125,
    5: 0.125,
    6: 0.10,    # 10%
    7: 0.10,
    8: 0.10,
    9: 0.10,
    10: 0.10,
    11: 0.10,
    12: 0.10,
    13: 0.10,
    14: 0.10,
    15: 0.05,   # 5%
    16: 0.05,
    17: 0.05,
    18: 0.05,
    19: 0.05,
    20: 0.05,
    21: 0.05,
    22: 0.05,
    23: 0.05,
    24: 0.05,
}

# ==================== RANDOM VARIABLE DISTRIBUTIONS ====================
def generate_random_variables(n):
    """Generate all random variables for n iterations"""
    
    # 1. GDP Deflator: Normal(3.5%, 95% CI: 3-4%)
    # 95% CI means ±1.96σ = 0.5%, so σ = 0.5/1.96 ≈ 0.255%
    gdp_deflator = np.random.normal(0.035, 0.00255, n)
    
    # 2. Total Well Cost: Normal($750k, 95% CI: ±$30k)
    # σ = 30/1.96 ≈ 15.3k
    total_well_cost = np.random.normal(750000, 15306, n)
    
    # 3. Production Adequacy: Bernoulli(p=0.90)
    production_adequate = np.random.binomial(1, 0.90, n)
    
    # 4. 1st Year McF: Log Normal(85k, sd=15k)
    first_year_mcf = np.random.lognormal(mean=np.log(85000), sigma=15000/85000, size=n)
    
    # 5. Decline Multiplier: triangular(mean=1, 95% low-high: 0.5-1.75)
    decline_multiplier = np.random.triangular(0.5, 1.0, 1.75, n)
    
    # 6. BTU Content: Triangular(low=1055, mode=1260, high=1350)
    btu_content = np.random.triangular(1055, 1260, 1350, n)
    
    return {
        'gdp_deflator': gdp_deflator,
        'total_well_cost': total_well_cost,
        'production_adequate': production_adequate,
        'first_year_mcf': first_year_mcf,
        'decline_multiplier': decline_multiplier,
        'btu_content': btu_content
    }

# ==================== PRODUCTION CALCULATION ====================
def calculate_production(first_year_mcf, decline_multiplier, production_adequate):
    """Calculate annual production for 25 years"""
    production = np.zeros(N_YEARS)
    
    if production_adequate == 0:
        return production  # All zeros if production not adequate
    
    production[0] = first_year_mcf
    
    for year in range(1, N_YEARS):
        year_num = year + 1  # Year 1 = index 0, Year 2 = index 1, etc.
        static_decline = DECLINE_RATES.get(year_num, 0.05)
        effective_decline = static_decline * decline_multiplier
        production[year] = production[year - 1] * (1 - effective_decline)
    
    return production

# ==================== PRICE CALCULATION ====================
def calculate_prices(gdp_deflator):
    """Calculate price per MMBTU for each year with escalation starting year 5"""
    prices = np.zeros(N_YEARS)
    
    for year in range(N_YEARS):
        if year < PRICE_INCREASE_START_YEAR:
            prices[year] = CURRENT_PRICE_MMBTU
        else:
            years_since_increase = year - PRICE_INCREASE_START_YEAR + 1
            prices[year] = CURRENT_PRICE_MMBTU * (1 + gdp_deflator) ** years_since_increase
    
    return prices

# ==================== REVENUE CALCULATION ====================
def calculate_revenue(production_mcf, btu_content, prices):
    """Calculate gross and net revenue"""
    # Convert MCF to MMBTU
    mmbtu = production_mcf * btu_content / 1000
    
    # Gross Revenue
    gross_revenue = mmbtu * prices
    
    # Net Revenue (after royalties)
    net_revenue = gross_revenue * (1 - ROYALTY_RATE)
    
    return gross_revenue, net_revenue

# ==================== COST CALCULATION ====================
def calculate_costs(total_well_cost, gdp_deflator):
    """Calculate initial investment and annual operating costs"""
    # Year 0 Investment
    intangible_cost = total_well_cost * INTANGIBLE_COST_PCT
    tangible_cost = total_well_cost * (1 - INTANGIBLE_COST_PCT)
    
    # Operating Costs (inflated at half the GDP deflator rate)
    annual_operating = np.zeros(N_YEARS)
    inflation_rate_opex = gdp_deflator * INFLATION_FACTOR_WELL
    
    for year in range(N_YEARS):
        annual_operating[year] = (MONTHLY_OPERATING_COST * 12) * (1 + inflation_rate_opex) ** year
    
    return intangible_cost, tangible_cost, annual_operating

# ==================== DEPRECIATION ====================
def calculate_depreciation(intangible_cost, tangible_cost):
    """Calculate annual depreciation"""
    depreciation = np.zeros(N_YEARS)
    
    # Year 0 (index 0): Intangible cost fully expensed
    depreciation[0] = intangible_cost
    
    # Years 1-7 (indices 0-6): Tangible cost straight-line over 7 years
    annual_tangible_depreciation = tangible_cost / 7
    for year in range(7):
        depreciation[year] += annual_tangible_depreciation
    
    return depreciation

# ==================== TAX CALCULATION ====================
def calculate_taxes(gross_revenue, net_revenue, operating_costs, lease_expense, 
                   depreciation, production_adequate):
    """Calculate taxes per the provided formula"""
    
    if production_adequate == 0:
        return np.zeros(N_YEARS), np.zeros(N_YEARS), np.zeros(N_YEARS), np.zeros(N_YEARS)
    
    # Severance Tax (on gross revenue)
    severance_tax = gross_revenue * SEVERANCE_TAX_RATE
    
    # County Tax (on gross revenue)
    county_tax = gross_revenue * COUNTY_TAX_RATE
    
    # Profit Before Tax
    profit_before_tax = net_revenue - operating_costs - lease_expense - county_tax
    
    # Depletion = min(0.5 * profit_before_tax, 0.15 * net_revenue)
    depletion = np.minimum(0.5 * profit_before_tax, 0.15 * net_revenue)
    depletion = np.maximum(depletion, 0)  # Cannot be negative
    
    # State Income Tax = state_rate × (profit_before_tax - depletion) - 0.5 × severance_tax
    state_income_tax = STATE_TAX_RATE * (profit_before_tax - depletion) - 0.5 * severance_tax
    state_income_tax = np.maximum(state_income_tax, 0)  # Cannot be negative
    
    # Federal Income Tax = federal_rate × (profit_before_tax - depletion - state_income_tax)
    federal_income_tax = FEDERAL_TAX_RATE * (profit_before_tax - depletion - state_income_tax)
    federal_income_tax = np.maximum(federal_income_tax, 0)  # Cannot be negative
    
    return severance_tax, county_tax, state_income_tax, federal_income_tax

# ==================== CASH FLOW CALCULATION ====================
def calculate_cash_flows(gross_revenue, net_revenue, operating_costs, lease_expense,
                        severance_tax, county_tax, state_income_tax, federal_income_tax,
                        total_well_cost, production_adequate):
    """Calculate annual cash flows"""
    
    cash_flows = np.zeros(N_YEARS + 1)  # Include Year 0
    
    # Year 0: Initial Investment (negative)
    cash_flows[0] = -total_well_cost
    
    if production_adequate == 0:
        return cash_flows  # Only initial investment, no other cash flows
    
    # Years 1-25
    for year in range(N_YEARS):
        cf = (net_revenue[year] 
              - operating_costs[year] 
              - lease_expense 
              - severance_tax[year]
              - county_tax[year]
              - state_income_tax[year] 
              - federal_income_tax[year])
        cash_flows[year + 1] = cf
    
    return cash_flows

# ==================== NPV CALCULATION ====================
def calculate_npv(cash_flows, discount_rate):
    """Calculate Net Present Value"""
    npv = 0
    for year, cf in enumerate(cash_flows):
        npv += cf / (1 + discount_rate) ** year
    return npv

# ==================== IRR CALCULATION ====================
def calculate_irr(cash_flows, max_iterations=1000):
    """Calculate Internal Rate of Return using Newton-Raphson method"""
    # If all cash flows are zero or negative, IRR is undefined
    if np.sum(cash_flows[1:]) <= 0:
        return np.nan
    
    # Initial guess
    irr = 0.1
    
    for _ in range(max_iterations):
        # Calculate NPV at current IRR
        npv = sum(cf / (1 + irr) ** i for i, cf in enumerate(cash_flows))
        
        # Calculate derivative of NPV
        dnpv = sum(-i * cf / (1 + irr) ** (i + 1) for i, cf in enumerate(cash_flows))
        
        # Avoid division by zero
        if abs(dnpv) < 1e-10:
            break
        
        # Newton-Raphson update
        irr_new = irr - npv / dnpv
        
        # Check for convergence
        if abs(irr_new - irr) < 1e-6:
            return irr_new
        
        irr = irr_new
        
        # Keep IRR in reasonable bounds
        if irr < -0.99:
            irr = -0.99
        elif irr > 10:
            return np.nan
    
    return irr

# ==================== MAIN SIMULATION ====================
def run_simulation():
    """Run Monte Carlo simulation"""
    print("Generating random variables...")
    random_vars = generate_random_variables(N_ITERATIONS)
    
    # Storage for results
    npv_results = np.zeros(N_ITERATIONS)
    irr_results = np.zeros(N_ITERATIONS)
    
    # Storage for sensitivity analysis
    sensitivity_data = []
    
    print("Running simulation...")
    for i in range(N_ITERATIONS):
        if (i + 1) % 1000 == 0:
            print(f"  Iteration {i + 1}/{N_ITERATIONS}")
        
        # Extract random variables for this iteration
        gdp_def = random_vars['gdp_deflator'][i]
        well_cost = random_vars['total_well_cost'][i]
        prod_adequate = random_vars['production_adequate'][i]
        first_mcf = random_vars['first_year_mcf'][i]
        decline_mult = random_vars['decline_multiplier'][i]
        btu = random_vars['btu_content'][i]
        
        # Calculate production
        production = calculate_production(first_mcf, decline_mult, prod_adequate)
        
        # Calculate prices
        prices = calculate_prices(gdp_def)
        
        # Calculate revenue
        gross_rev, net_rev = calculate_revenue(production, btu, prices)
        
        # Calculate costs
        intangible, tangible, operating = calculate_costs(well_cost, gdp_def)
        
        # Calculate depreciation
        depreciation = calculate_depreciation(intangible, tangible)
        
        # Calculate taxes
        sev_tax, county_tax, state_tax, fed_tax = calculate_taxes(
            gross_rev, net_rev, operating, ANNUAL_LEASE_EXPENSE, depreciation, prod_adequate
        )
        
        # Calculate cash flows
        cash_flows = calculate_cash_flows(
            gross_rev, net_rev, operating, ANNUAL_LEASE_EXPENSE,
            sev_tax, county_tax, state_tax, fed_tax, well_cost, prod_adequate
        )
        
        # Calculate NPV and IRR
        npv = calculate_npv(cash_flows, DISCOUNT_RATE)
        irr = calculate_irr(cash_flows)
        
        npv_results[i] = npv
        irr_results[i] = irr
        
        # Store for sensitivity analysis
        sensitivity_data.append({
            'npv': npv,
            'irr': irr,
            'gdp_deflator': gdp_def,
            'total_well_cost': well_cost,
            'production_adequate': prod_adequate,
            'first_year_mcf': first_mcf,
            'decline_multiplier': decline_mult,
            'btu_content': btu
        })
    
    print("Simulation complete!")
    return npv_results, irr_results, pd.DataFrame(sensitivity_data)

# ==================== RESULTS ANALYSIS ====================
def analyze_results(npv_results, irr_results, sensitivity_df):
    """Analyze and display simulation results"""
    
    print("\n" + "="*70)
    print("MONTE CARLO SIMULATION RESULTS")
    print("="*70)
    
    # NPV Statistics
    print("\nNET PRESENT VALUE (NPV) STATISTICS:")
    print(f"  Mean NPV:           ${npv_results.mean():,.2f}")
    print(f"  Median NPV:         ${np.median(npv_results):,.2f}")
    print(f"  Std Dev:            ${npv_results.std():,.2f}")
    print(f"  Min NPV:            ${npv_results.min():,.2f}")
    print(f"  Max NPV:            ${npv_results.max():,.2f}")
    print(f"  P10 (90th %ile):    ${np.percentile(npv_results, 90):,.2f}")
    print(f"  P50 (50th %ile):    ${np.percentile(npv_results, 50):,.2f}")
    print(f"  P90 (10th %ile):    ${np.percentile(npv_results, 10):,.2f}")
    print(f"  Probability NPV>0:  {(npv_results > 0).sum() / N_ITERATIONS * 100:.1f}%")
    
    # IRR Statistics (excluding NaN values)
    valid_irr = irr_results[~np.isnan(irr_results)]
    print("\nINTERNAL RATE OF RETURN (IRR) STATISTICS:")
    print(f"  Valid IRR Count:    {len(valid_irr)} ({len(valid_irr)/N_ITERATIONS*100:.1f}%)")
    if len(valid_irr) > 0:
        print(f"  Mean IRR:           {valid_irr.mean()*100:.2f}%")
        print(f"  Median IRR:         {np.median(valid_irr)*100:.2f}%")
        print(f"  Std Dev:            {valid_irr.std()*100:.2f}%")
        print(f"  Min IRR:            {valid_irr.min()*100:.2f}%")
        print(f"  Max IRR:            {valid_irr.max()*100:.2f}%")
        print(f"  P10 (90th %ile):    {np.percentile(valid_irr, 90)*100:.2f}%")
        print(f"  P50 (50th %ile):    {np.percentile(valid_irr, 50)*100:.2f}%")
        print(f"  P90 (10th %ile):    {np.percentile(valid_irr, 10)*100:.2f}%")
        print(f"  Probability IRR>8%: {(valid_irr > 0.08).sum() / len(valid_irr) * 100:.1f}%")
    
    print("\n" + "="*70)
    
    return valid_irr

# ==================== SENSITIVITY ANALYSIS ====================
def sensitivity_analysis(sensitivity_df):
    """Perform correlation-based sensitivity analysis"""
    
    # Filter to only successful projects for meaningful correlation
    success_df = sensitivity_df[sensitivity_df['production_adequate'] == 1].copy()
    
    print("\nSENSITIVITY ANALYSIS (Correlation with NPV):")
    print("-" * 70)
    
    variables = ['gdp_deflator', 'total_well_cost', 'first_year_mcf', 
                 'decline_multiplier', 'btu_content']
    var_labels = ['GDP Deflator', 'Total Well Cost', '1st Year McF', 
                  'Decline Multiplier', 'BTU Content']
    
    correlations = []
    for var, label in zip(variables, var_labels):
        corr = success_df['npv'].corr(success_df[var])
        correlations.append((label, corr))
        print(f"  {label:20s}: {corr:+.3f}")
    
    # Also show production adequacy impact
    overall_mean_npv = sensitivity_df['npv'].mean()
    success_mean_npv = success_df['npv'].mean()
    failure_mean_npv = sensitivity_df[sensitivity_df['production_adequate'] == 0]['npv'].mean()
    
    print(f"\n  Production Adequacy Impact:")
    print(f"    Mean NPV (Success):  ${success_mean_npv:,.2f}")
    print(f"    Mean NPV (Failure):  ${failure_mean_npv:,.2f}")
    print(f"    Difference:          ${success_mean_npv - failure_mean_npv:,.2f}")
    
    return correlations, success_df

# ==================== VISUALIZATION ====================
def create_visualizations(npv_results, irr_results, correlations, sensitivity_df):
    """Create visualization plots"""
    
    fig = plt.figure(figsize=(16, 12))
    
    # 1. NPV Distribution
    ax1 = plt.subplot(3, 3, 1)
    ax1.hist(npv_results, bins=100, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='Break-even')
    ax1.axvline(np.mean(npv_results), color='green', linestyle='--', linewidth=2, label='Mean')
    ax1.set_xlabel('NPV ($)', fontsize=10)
    ax1.set_ylabel('Frequency', fontsize=10)
    ax1.set_title('NPV Distribution', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. NPV Cumulative Distribution
    ax2 = plt.subplot(3, 3, 2)
    sorted_npv = np.sort(npv_results)
    cumulative = np.arange(1, len(sorted_npv) + 1) / len(sorted_npv) * 100
    ax2.plot(sorted_npv, cumulative, linewidth=2, color='steelblue')
    ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='Break-even')
    ax2.axhline(50, color='gray', linestyle=':', alpha=0.5)
    ax2.set_xlabel('NPV ($)', fontsize=10)
    ax2.set_ylabel('Cumulative Probability (%)', fontsize=10)
    ax2.set_title('NPV Cumulative Distribution', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. IRR Distribution
    ax3 = plt.subplot(3, 3, 3)
    valid_irr = irr_results[~np.isnan(irr_results)]
    if len(valid_irr) > 0:
        ax3.hist(valid_irr * 100, bins=100, edgecolor='black', alpha=0.7, color='coral')
        ax3.axvline(8, color='red', linestyle='--', linewidth=2, label='Discount Rate (8%)')
        ax3.axvline(np.mean(valid_irr) * 100, color='green', linestyle='--', linewidth=2, label='Mean')
        ax3.set_xlabel('IRR (%)', fontsize=10)
        ax3.set_ylabel('Frequency', fontsize=10)
        ax3.set_title('IRR Distribution', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. Sensitivity Analysis - Tornado Chart
    ax4 = plt.subplot(3, 3, 4)
    labels, corrs = zip(*sorted(correlations, key=lambda x: abs(x[1]), reverse=True))
    colors = ['green' if c > 0 else 'red' for c in corrs]
    y_pos = np.arange(len(labels))
    ax4.barh(y_pos, corrs, color=colors, alpha=0.7, edgecolor='black')
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(labels)
    ax4.set_xlabel('Correlation with NPV', fontsize=10)
    ax4.set_title('Sensitivity Analysis (Tornado Chart)', fontsize=12, fontweight='bold')
    ax4.axvline(0, color='black', linewidth=0.8)
    ax4.grid(True, alpha=0.3, axis='x')
    
    # 5. NPV vs First Year McF
    ax5 = plt.subplot(3, 3, 5)
    success_df = sensitivity_df[sensitivity_df['production_adequate'] == 1]
    scatter = ax5.scatter(success_df['first_year_mcf'], success_df['npv'], 
                         alpha=0.3, s=10, c=success_df['btu_content'], cmap='viridis')
    ax5.set_xlabel('1st Year McF', fontsize=10)
    ax5.set_ylabel('NPV ($)', fontsize=10)
    ax5.set_title('NPV vs 1st Year McF (colored by BTU)', fontsize=12, fontweight='bold')
    plt.colorbar(scatter, ax=ax5, label='BTU Content')
    ax5.grid(True, alpha=0.3)
    ax5.axhline(0, color='red', linestyle='--', alpha=0.5)
    
    # 6. NPV vs Well Cost
    ax6 = plt.subplot(3, 3, 6)
    ax6.scatter(success_df['total_well_cost'], success_df['npv'], 
               alpha=0.3, s=10, color='steelblue')
    ax6.set_xlabel('Total Well Cost ($)', fontsize=10)
    ax6.set_ylabel('NPV ($)', fontsize=10)
    ax6.set_title('NPV vs Total Well Cost', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    ax6.axhline(0, color='red', linestyle='--', alpha=0.5)
    
    # 7. NPV vs Decline Multiplier
    ax7 = plt.subplot(3, 3, 7)
    ax7.scatter(success_df['decline_multiplier'], success_df['npv'], 
               alpha=0.3, s=10, color='coral')
    ax7.set_xlabel('Decline Multiplier', fontsize=10)
    ax7.set_ylabel('NPV ($)', fontsize=10)
    ax7.set_title('NPV vs Decline Multiplier', fontsize=12, fontweight='bold')
    ax7.grid(True, alpha=0.3)
    ax7.axhline(0, color='red', linestyle='--', alpha=0.5)
    
    # 8. NPV vs BTU Content
    ax8 = plt.subplot(3, 3, 8)
    ax8.scatter(success_df['btu_content'], success_df['npv'], 
               alpha=0.3, s=10, color='green')
    ax8.set_xlabel('BTU Content', fontsize=10)
    ax8.set_ylabel('NPV ($)', fontsize=10)
    ax8.set_title('NPV vs BTU Content', fontsize=12, fontweight='bold')
    ax8.grid(True, alpha=0.3)
    ax8.axhline(0, color='red', linestyle='--', alpha=0.5)
    
    # 9. Production Adequacy Impact
    ax9 = plt.subplot(3, 3, 9)
    prod_groups = sensitivity_df.groupby('production_adequate')['npv'].apply(list)
    box_data = [prod_groups[0], prod_groups[1]]
    bp = ax9.boxplot(box_data, labels=['Failure (0)', 'Success (1)'], 
                     patch_artist=True, widths=0.6)
    bp['boxes'][0].set_facecolor('red')
    bp['boxes'][1].set_facecolor('green')
    for box in bp['boxes']:
        box.set_alpha(0.6)
    ax9.set_ylabel('NPV ($)', fontsize=10)
    ax9.set_xlabel('Production Adequacy', fontsize=10)
    ax9.set_title('NPV by Production Adequacy', fontsize=12, fontweight='bold')
    ax9.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax9.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('/Users/santiagoriverabarbosa/desktop/darden/Q3/DA1/MonteCarloSimulator/monte_carlo_results.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved to: monte_carlo_results.png")
    
    return fig

# ==================== EXPORT RESULTS ====================
def export_results(npv_results, irr_results, sensitivity_df):
    """Export results to Excel"""

    with pd.ExcelWriter('/Users/santiagoriverabarbosa/desktop/darden/q3/da1/MonteCarloSimulator/monte_carlo_results.xlsx', engine='openpyxl') as writer:
        # Summary Statistics
        summary_data = {
            'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 
                      'P10 (90th %ile)', 'P50 (Median)', 'P90 (10th %ile)',
                      'Probability > 0'],
            'NPV ($)': [
                npv_results.mean(),
                np.median(npv_results),
                npv_results.std(),
                npv_results.min(),
                npv_results.max(),
                np.percentile(npv_results, 90),
                np.percentile(npv_results, 50),
                np.percentile(npv_results, 10),
                (npv_results > 0).sum() / N_ITERATIONS
            ]
        }
        
        valid_irr = irr_results[~np.isnan(irr_results)]
        summary_data['IRR (%)'] = [
            valid_irr.mean() * 100 if len(valid_irr) > 0 else np.nan,
            np.median(valid_irr) * 100 if len(valid_irr) > 0 else np.nan,
            valid_irr.std() * 100 if len(valid_irr) > 0 else np.nan,
            valid_irr.min() * 100 if len(valid_irr) > 0 else np.nan,
            valid_irr.max() * 100 if len(valid_irr) > 0 else np.nan,
            np.percentile(valid_irr, 90) * 100 if len(valid_irr) > 0 else np.nan,
            np.percentile(valid_irr, 50) * 100 if len(valid_irr) > 0 else np.nan,
            np.percentile(valid_irr, 10) * 100 if len(valid_irr) > 0 else np.nan,
            (valid_irr > 0.08).sum() / len(valid_irr) if len(valid_irr) > 0 else np.nan
        ]
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Full Results
        results_df = pd.DataFrame({
            'Iteration': range(1, N_ITERATIONS + 1),
            'NPV ($)': npv_results,
            'IRR (%)': irr_results * 100
        })
        results_df.to_excel(writer, sheet_name='Full Results', index=False)
        
        # Sensitivity Data
        sensitivity_export = sensitivity_df.copy()
        sensitivity_export['irr'] = sensitivity_export['irr'] * 100
        sensitivity_export.to_excel(writer, sheet_name='Sensitivity Data', index=False)
    
    print("Results exported to: monte_carlo_results.xlsx")

# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    print("="*70)
    print("OIL & GAS PROJECT - MONTE CARLO SIMULATION")
    print("="*70)
    print(f"Number of Iterations: {N_ITERATIONS:,}")
    print(f"Project Duration: {N_YEARS} years")
    print(f"Discount Rate: {DISCOUNT_RATE*100}%")
    print("="*70)
    
    # Run simulation
    npv_results, irr_results, sensitivity_df = run_simulation()
    
    # Analyze results
    valid_irr = analyze_results(npv_results, irr_results, sensitivity_df)
    
    # Sensitivity analysis
    correlations, success_df = sensitivity_analysis(sensitivity_df)
    
    # Create visualizations
    fig = create_visualizations(npv_results, irr_results, correlations, sensitivity_df)
    
    # Export results
    export_results(npv_results, irr_results, sensitivity_df)
    
    print("\n" + "="*70)
    print("SIMULATION COMPLETE!")
    print("="*70)
    print("\nFiles created:")
    print("  1. monte_carlo_results.png  - Visualization dashboard")
    print("  2. monte_carlo_results.xlsx - Detailed results spreadsheet")
    print("="*70)
