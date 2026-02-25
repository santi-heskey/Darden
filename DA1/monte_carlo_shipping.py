import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Constants and Initial Assumptions ---

# Costs
DAILY_OPERATING_COST = 10000  # All-inclusive daily cost

# Simulation settings
N_TRIALS = 10000              # Number of simulations to run
DISTRIBUTION_TYPE = 'normal'  # 'uniform' or 'normal'

# Parameters for Uniform Distribution
RATE_MULTIPLIER_MIN = 0.5
RATE_MULTIPLIER_MAX = 1.5

# Parameters for Normal Distribution (used if DISTRIBUTION_TYPE is 'normal')
# Mean of 1.0 means the average outcome is the spot rate.
# Std Dev of 0.25 means ~68% of outcomes are within 0.75x-1.25x of the spot rate.
NORMAL_DIST_MEAN = 1.0
NORMAL_DIST_STD_DEV = 0.25

# Voyage Details
# Leg 1: UK to Destination (Deadheading - no revenue)
UK_TO_BALTIC_DAYS = 5
UK_TO_MED_DAYS = 9

# Spot Rates (our baseline for the simulation)
TD17_BALTIC_SPOT_RATE = 40000
TD19_MED_SPOT_RATE = 45000

# Leg 2: Destination to Next Port (Laden - earning revenue)
MED_TO_UK_DAYS = 9
BALTIC_TO_ANOTHER_BALTIC_DAYS = 3
BALTIC_TO_UK_DAYS = 5
BALTIC_TO_MED_DAYS = 15

# --- Simulation Core ---

def get_rate_multipliers(n_trials):
    """Generates an array of random rate multipliers based on the chosen distribution."""
    if DISTRIBUTION_TYPE == 'uniform':
        return np.random.uniform(RATE_MULTIPLIER_MIN, RATE_MULTIPLIER_MAX, n_trials)
    elif DISTRIBUTION_TYPE == 'normal':
        multipliers = np.random.normal(NORMAL_DIST_MEAN, NORMAL_DIST_STD_DEV, n_trials)
        # Clip at 0 to prevent negative rates, which are not realistic.
        return np.clip(multipliers, 0, None)
    else:
        raise ValueError("Unsupported DISTRIBUTION_TYPE. Choose 'uniform' or 'normal'.")

# --- Voyage Scenarios ---

def simulate_mediterranean_voyage(n_trials):
    """Simulates the entire UK -> Med -> UK voyage."""
    print("Simulating Mediterranean voyage...")
    rate_multipliers = get_rate_multipliers(n_trials)
    
    simulated_rates = TD19_MED_SPOT_RATE * rate_multipliers
    total_revenue = simulated_rates * MED_TO_UK_DAYS
    
    total_days = UK_TO_MED_DAYS + MED_TO_UK_DAYS
    total_cost = DAILY_OPERATING_COST * total_days
    
    total_profit = total_revenue - total_cost
    return total_profit / total_days

def simulate_baltic_voyage(n_trials):
    """
    Simulates the UK -> Baltic voyage, analyzing all three potential onward legs.
    """
    print("Simulating Baltic voyage (analyzing 3 sub-routes)...")
    rate_multipliers = get_rate_multipliers(n_trials)
    simulated_rates = TD17_BALTIC_SPOT_RATE * rate_multipliers

    def calculate_avg_profit(laden_days):
        total_revenue = simulated_rates * laden_days
        total_days = UK_TO_BALTIC_DAYS + laden_days
        total_cost = DAILY_OPERATING_COST * total_days
        total_profit = total_revenue - total_cost
        return total_profit / total_days if total_days > 0 else np.zeros(n_trials)

    outcomes_to_baltic = calculate_avg_profit(BALTIC_TO_ANOTHER_BALTIC_DAYS)
    outcomes_to_uk = calculate_avg_profit(BALTIC_TO_UK_DAYS)
    outcomes_to_med = calculate_avg_profit(BALTIC_TO_MED_DAYS)
    
    # For each trial, find the maximum average daily profit among the three options
    return np.maximum.reduce([outcomes_to_baltic, outcomes_to_uk, outcomes_to_med])

# --- Reporting & Analysis ---

def generate_report(baltic_results, med_results):
    """
    Generates a detailed report comparing the simulation results of the two voyages.
    """
    # (The implementation of this function is unchanged)
    def get_stats(name, results):
        return {
            "name": name,
            "mean_profit": results.mean(),
            "std_dev": results.std(),
            "percentile_5": np.percentile(results, 5),
            "percentile_95": np.percentile(results, 95),
            "profit_prob": (results > 0).mean() * 100
        }

    baltic_stats = get_stats("Baltic", baltic_results)
    med_stats = get_stats("Mediterranean", med_results)

    report = f"""
--- Decision Analysis Report (Source: {DISTRIBUTION_TYPE.capitalize()} Distribution) ---

Based on {len(baltic_results):,} simulations:

--------------------------------------------------
Metric                 | {baltic_stats['name']:<20} | {med_stats['name']}
--------------------------------------------------
Avg. Daily Profit      | ${baltic_stats['mean_profit']:>19,.2f} | ${med_stats['mean_profit']:>19,.2f}
Std. Dev (Risk)        | ${baltic_stats['std_dev']:>19,.2f} | ${med_stats['std_dev']:>19,.2f}
5th Percentile Profit  | ${baltic_stats['percentile_5']:>19,.2f} | ${med_stats['percentile_5']:>19,.2f}
95th Percentile Profit | ${baltic_stats['percentile_95']:>19,.2f} | ${med_stats['percentile_95']:>19,.2f}
Probability of Profit  | {baltic_stats['profit_prob']:>18.1f}% | {med_stats['profit_prob']:>18.1f}%
--------------------------------------------------

--- Recommendation ---
"""
    if baltic_stats['mean_profit'] > med_stats['mean_profit']:
        recommendation = (
            "Choose the BALTIC route.\n"
            f"This option has a higher expected average daily profit (${baltic_stats['mean_profit']-med_stats['mean_profit']:,.2f} more per day)."
        )
    else:
        recommendation = (
            "Choose the MEDITERRANEAN route.\n"
            f"This option has a higher expected average daily profit (${med_stats['mean_profit']-baltic_stats['mean_profit']:,.2f} more per day)."
        )
    report += recommendation
    
    risk_diff_threshold = 500
    if abs(baltic_stats['std_dev'] - med_stats['std_dev']) > risk_diff_threshold:
        if baltic_stats['std_dev'] > med_stats['std_dev']:
            report += "\nNote: The Baltic route shows significantly higher volatility (risk)."
        else:
            report += "\nNote: The Mediterranean route shows significantly higher volatility (risk)."
    return report

def generate_visual(baltic_results, med_results, output_filename="shipping_analysis_plot.png"):
    """
    Generates and saves a visualization of the simulation results.
    """
    print(f"Generating visual report and saving to {output_filename}...")
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plotting the distributions
    sns.histplot(data={'Baltic': baltic_results, 'Mediterranean': med_results}, ax=ax, kde=True, bins=50)
    
    # Getting stats for annotations
    baltic_mean = baltic_results.mean()
    med_mean = med_results.mean()
    
    # Adding vertical lines for the means
    ax.axvline(baltic_mean, color=sns.color_palette()[0], linestyle='--', label=f'Baltic Mean: ${baltic_mean:,.2f}')
    ax.axvline(med_mean, color=sns.color_palette()[1], linestyle='--', label=f'Med Mean: ${med_mean:,.2f}')
    
    ax.set_title('Distribution of Average Daily Profit', fontsize=16)
    ax.set_xlabel('Average Daily Profit ($)', fontsize=12)
    ax.set_ylabel('Frequency (Number of Trials)', fontsize=12)
    ax.legend()
    
    # Improve formatting
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    plt.tight_layout()
    plt.savefig(output_filename)
    print("Visual report saved successfully.")

# --- Main Execution ---

def run_voyage_simulation(n_trials=N_TRIALS):
    """
    Main function to run the full analysis and generate a report.
    """
    print("--- Starting shipping voyage analysis ---")
    print(f"Running {n_trials:,} simulations with a '{DISTRIBUTION_TYPE}' rate distribution.")
    
    baltic_results = simulate_baltic_voyage(n_trials)
    med_results = simulate_mediterranean_voyage(n_trials)

    # Generate and print the text report
    report = generate_report(baltic_results, med_results)
    print(report)
    
    # Generate and save the visual report
    generate_visual(baltic_results, med_results)

if __name__ == "__main__":
    run_voyage_simulation()
