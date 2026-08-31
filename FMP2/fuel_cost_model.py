# This script will calculate fuel costs based on different hedging strategies and tranches.
# It will take user input for various parameters and visualize the results using Matplotlib.

import matplotlib.pyplot as plt
import numpy as np

def calculate_fuel_cost(market_price, fuel_tranches, hedge_strategy, hedge_params, fuel_volume, num_tranches):
    """
    Calculates the total fuel cost for a given market price and hedging strategy.

    Args:
        market_price (float): The current market price of fuel.
        fuel_tranches (list of dict): A list of dictionaries, where each dict
                                      contains 'spot_price', 'cap_price', and 'floor_price' for a tranche.
        hedge_strategy (str): The hedging strategy ('none', 'swap', 'cap', 'collar').
        hedge_params (dict): A dictionary containing parameters specific to the hedge strategy.
                             e.g., {'swap_price': float} for 'swap'.
        fuel_volume (float): The total fuel volume to be hedged.
        num_tranches (int): The number of fuel tranches.

    Returns:
        float: The total calculated fuel cost.
    """
    total_cost = 0.0
    tranche_quantity = fuel_volume / num_tranches
    
    for tranche in fuel_tranches:
        # The base price for the tranche is its specified price, but it can be affected by market price and hedging
        base_tranche_spot_price = tranche['spot_price'] 
        
        effective_price = market_price # Start with market price as the base for comparison

        if hedge_strategy == 'swap':
            swap_price = hedge_params.get('swap_price')
            if swap_price is not None:
                effective_price = swap_price
            else:
                raise ValueError("Swap price not provided for swap strategy.")
        elif hedge_strategy == 'cap':
            cap_price = tranche['cap_price']
            if cap_price is not None:
                effective_price = min(market_price, cap_price)
            else:
                raise ValueError("Cap price not provided for cap strategy in tranche.")
        elif hedge_strategy == 'collar':
            cap_price = tranche['cap_price']
            floor_price = tranche['floor_price']
            if cap_price is not None and floor_price is not None:
                effective_price = max(floor_price, min(market_price, cap_price))
            else:
                raise ValueError("Cap or floor price not provided for collar strategy in tranche.")
        elif hedge_strategy == 'none':
            effective_price = base_tranche_spot_price # For no hedge, use the tranche's spot price
        else:
            raise ValueError(f"Unknown hedge strategy: {hedge_strategy}")

        total_cost += tranche_quantity * effective_price
        
    return total_cost

def get_user_input():
    """
    Prompts the user for all necessary input parameters.

    Returns:
        tuple: A tuple containing (swap_price, fuel_volume, num_tranches, fuel_tranches)
    """
    print("--- Fuel Cost Model Configuration ---")
    fuel_volume = float(input("Enter the total fuel volume to be hedged (e.g., gallons): "))
    swap_price = float(input("Enter the Swap Price (spot price for swap strategy): "))

    num_tranches = int(input("Enter the number of fuel tranches: "))
    fuel_tranches = []
    for i in range(num_tranches):
        print(f"--- Tranche {i + 1} ---")
        spot_price = float(input(f"Enter spot price per unit for tranche {i + 1}: "))
        cap_price = float(input(f"Enter cap price per unit for tranche {i + 1}: "))
        floor_price = float(input(f"Enter floor price per unit for tranche {i + 1}: "))
        fuel_tranches.append({'spot_price': spot_price, 'cap_price': cap_price, 'floor_price': floor_price})
    
    return swap_price, fuel_volume, num_tranches, fuel_tranches

def main():
    swap_price, fuel_volume, num_tranches, fuel_tranches = get_user_input()

    # Define a range of market prices to simulate
    min_market_price = 0.5 * min(t['spot_price'] for t in fuel_tranches) # Example: 50% of lowest tranche spot price
    max_market_price = 1.5 * max(t['spot_price'] for t in fuel_tranches) # Example: 150% of highest tranche spot price
    market_prices = np.linspace(min_market_price, max_market_price, 100)

    # Calculate costs for each strategy
    costs_none = [calculate_fuel_cost(p, fuel_tranches, 'none', {}, fuel_volume, num_tranches) for p in market_prices]
    costs_swap = [calculate_fuel_cost(p, fuel_tranches, 'swap', {'swap_price': swap_price}, fuel_volume, num_tranches) for p in market_prices]
    costs_cap = [calculate_fuel_cost(p, fuel_tranches, 'cap', {}, fuel_volume, num_tranches) for p in market_prices]
    costs_collar = [calculate_fuel_cost(p, fuel_tranches, 'collar', {}, fuel_volume, num_tranches) for p in market_prices]

    # Plotting
    plt.figure(figsize=(12, 7))
    plt.plot(market_prices, costs_none, label='No Hedge', linestyle='--', color='grey')
    plt.plot(market_prices, costs_swap, label=f'Swap (at {swap_price:.2f})', color='red')
    plt.plot(market_prices, costs_cap, label=f'Cap', color='green')
    plt.plot(market_prices, costs_collar, label=f'Collar', color='blue')

    plt.xlabel('Market Fuel Price Per Unit')
    plt.ylabel('Total Fuel Cost')
    plt.title('Fuel Cost with Different Hedging Strategies')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
