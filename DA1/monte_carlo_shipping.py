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
MED_TO_UK_DAYS = 10.0855615
MED_TO_MED_DAYS = 6.615789474
BALTIC_TO_ANOTHER_BALTIC_DAYS = 3.82799634
BALTIC_TO_UK_DAYS = 8.229845626
BALTIC_TO_MED_DAYS = 116.12011173

# --- Onward Leg Probability Distributions ---
# Rows = chosen voyage (Baltic or Med)
# Columns = next destination (Med, UK Continent, Baltic)
# Source: empirical distribution data

BALTIC_ONWARD_PROBS = {
    'med':    0.09,   # 9%  -> Mediterranean
    'uk':     0.62,   # 62% -> UK Continent
    'baltic': 0.29,   # 29% -> Another Baltic
}

MED_ONWARD_PROBS = {
    'med':    0.72,   # 72% -> Mediterranean
    'uk':     0.28,   # 28% -> UK Continent
    'baltic': 0.00,   # 0%  -> Baltic
}

# --- Simulation Core ---

def get_rate_multipliers(n_trials):
    """Generates an array of random rate multipliers based on the chosen distribution."""
    if DISTRIBUTION_TYPE == 'uniform':
        return np.random.uniform(RATE_MULTIPLIER_MIN, RATE_MULTIPLIER_MAX, n_trials)
    elif DISTRIBUTION_TYPE == 'normal':
        multipliers = np.random.normal(NORMAL_DIST_MEAN, NORMAL_DIST_STD_DEV, n_trials)
        return np.clip(multipliers, 0, None)
    else:
        raise ValueError("Unsupported DISTRIBUTION_TYPE. Choose 'uniform' or 'normal'.")

def sample_laden_days(probs, n_trials, laden_days_map):
    """
    For each trial, randomly selects an onward leg duration based on
    the provided probability distribution.

    Args:
        probs (dict): {'med': p1, 'uk': p2, 'baltic': p3}
        n_trials (int): Number of simulation trials.
        laden_days_map (dict): {'med': days, 'uk': days, 'baltic': days}

    Returns:
        np.ndarray: Array of laden voyage days, one per trial.
    """
    keys = list(probs.keys())
    weights = [probs[k] for k in keys]
    days_options = [laden_days_map[k] for k in keys]

    # Draw a destination index for each trial according to the probability weights
    chosen_indices = np.random.choice(len(keys), size=n_trials, p=weights)
    return np.array([days_options[i] for i in chosen_indices])

# --- Voyage Scenarios ---

def simulate_mediterranean_voyage(n_trials):
    """
    Simulates the UK -> Med voyage.
    Onward leg is sampled from the Med probability distribution:
      72% -> Med, 28% -> UK Continent, 0% -> Baltic
    """
    print("Simulating Mediterranean voyage...")
    rate_multipliers = get_rate_multipliers(n_trials)
    simulated_rates = TD19_MED_SPOT_RATE * rate_multipliers

    laden_days_map = {
        'med':    MED_TO_MED_DAYS,
        'uk':     MED_TO_UK_DAYS,
        'baltic': 0,  # 0% probability, value irrelevant
    }
    laden_days = sample_laden_days(MED_ONWARD_PROBS, n_trials, laden_days_map)

    total_revenue = simulated_rates * laden_days
    total_days = UK_TO_MED_DAYS + laden_days
    total_cost = DAILY_OPERATING_COST * total_days
    total_profit = total_revenue - total_cost

    return total_profit / total_days

def simulate_baltic_voyage(n_trials):
    """
    Simulates the UK -> Baltic voyage.
    Onward leg is sampled from the Baltic probability distribution:
      9% -> Med, 62% -> UK Continent, 29% -> Another Baltic
    """
    print("Simulating Baltic voyage...")
    rate_multipliers = get_rate_multipliers(n_trials)
    simulated_rates = TD17_BALTIC_SPOT_RATE * rate_multipliers

    laden_days_map = {
        'med':    BALTIC_TO_MED_DAYS,
        'uk':     BALTIC_TO_UK_DAYS,
        'baltic': BALTIC_TO_ANOTHER_BALTIC_DAYS,
    }
    laden_days = sample_laden_days(BALTIC_ONWARD_PROBS, n_trials, laden_days_map)

    total_revenue = simulated_rates * laden_days
    total_days = UK_TO_BALTIC_DAYS + laden_days
    total_cost = DAILY_OPERATING_COST * total_days
    total_profit = total_revenue - total_cost

    return total_profit / total_days

# --- Reporting & Analysis ---

def generate_report(baltic_results, med_results):
    """
    Generates a detailed report comparing the simulation results of the two voyages.
    """
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
Onward leg destinations sampled from empirical probability distributions.
  Baltic onward:      Med {BALTIC_ONWARD_PROBS['med']*100:.0f}% | UK {BALTIC_ONWARD_PROBS['uk']*100:.0f}% | Baltic {BALTIC_ONWARD_PROBS['baltic']*100:.0f}%
  Mediterranean onward: Med {MED_ONWARD_PROBS['med']*100:.0f}% | UK {MED_ONWARD_PROBS['uk']*100:.0f}% | Baltic {MED_ONWARD_PROBS['baltic']*100:.0f}%

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

    sns.histplot(data={'Baltic': baltic_results, 'Mediterranean': med_results}, ax=ax, kde=True, bins=50)

    baltic_mean = baltic_results.mean()
    med_mean = med_results.mean()

    ax.axvline(baltic_mean, color=sns.color_palette()[0], linestyle='--', label=f'Baltic Mean: ${baltic_mean:,.2f}')
    ax.axvline(med_mean, color=sns.color_palette()[1], linestyle='--', label=f'Med Mean: ${med_mean:,.2f}')

    ax.set_title('Distribution of Average Daily Profit\n(Onward legs sampled from empirical probabilities)', fontsize=16)
    ax.set_xlabel('Average Daily Profit ($)', fontsize=12)
    ax.set_ylabel('Frequency (Number of Trials)', fontsize=12)
    ax.legend()

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

    report = generate_report(baltic_results, med_results)
    print(report)

    generate_visual(baltic_results, med_results)

if __name__ == "__main__":
    run_voyage_simulation()