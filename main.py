import numpy as np
from scipy.stats import norm
from scipy.optimize import bisect
import matplotlib.pyplot as plt
import math

class LivestockRiskProtection:

    def __init__(self, r, F):
        self.r = r # interest rate
        self.F = F # futures price (i.e. feeder cattle index)

    def black_scholes_put_price_with_futures(self, K, T, sigma):
        """Compute the Black-Scholes price for a European put option using the futures price."""
        d1 = (np.log(self.F / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        put_price = K * np.exp(-self.r * T) * norm.cdf(-d2) - self.F * np.exp(-self.r * T) * norm.cdf(-d1)
        return put_price

    def implied_volatility_with_futures(self, P, K, T):
        """Compute the implied volatility from the option price using Brent's method with futures price."""

        # Define the objective function for finding the root
        def objective_function(sigma):
            return self.black_scholes_put_price_with_futures(K, T, sigma) - P

        lowerbound = np.max([0, (K - self.F) * np.exp(-self.r * T)])
        if P < lowerbound:
            return np.nan
        if P == lowerbound:
            return 0
        if P >= self.F * np.exp(-self.r * T):
            return np.nan
        hi = 0.2

        while self.black_scholes_put_price_with_futures(K, T, hi) > P:
            hi = hi / 2
        while self.black_scholes_put_price_with_futures(K, T, hi) < P:
            hi = hi * 2
        lo = hi / 2

        # Use bisect to find the root, i.e., the implied volatility
        implied_vol = bisect(objective_function, lo, hi)
        return implied_vol

    def black_scholes_put_delta_with_futures(self, K, T, sigma):
        """Compute the Black-Scholes delta for a European put option using the futures price."""
        d1 = (np.log(self.F / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
        delta_put = -np.exp(-self.r * T) * norm.cdf(-d1)
        return delta_put

    # def compute_put_delta_from_premium_with_futures(self, P, K, T, sigma):
    #     """Compute the Black-Scholes put delta from the option premium using the futures price."""
    #     # sigma = self.implied_volatility_with_futures(P, K, T)
    #     delta_put = self.black_scholes_put_delta_with_futures(K, T, sigma)
    #     return delta_put

    def synthetic_put_delta(self, P0, K0, F0, T, sigma):
        # Strike prices
        K1 = K0 - 1.5 * abs(K0 - F0)  # K1 is 1.5 times further away from F than K0
        K2 = K0 - 5 * abs(K0 - F0)  # K2 is 5 times further away from F than K0

        # Compute individual deltas
        delta_K0 = self.black_scholes_put_delta_with_futures(K0, T, sigma)
        delta_K1 = self.black_scholes_put_delta_with_futures(K1, T, sigma) # assume 0 premium
        delta_K2 = self.black_scholes_put_delta_with_futures(K2, T, sigma) # assume 0 premium

        # Compute synthetic delta based on LRP substitution nature
        synthetic_delta = -1 * delta_K0 + 0.9 * delta_K1 + 0.1 * delta_K2

        return synthetic_delta

    def put_payoff(self, K):
        """Compute the payoff of a put option at maturity."""
        return np.maximum(K - self.F, 0)

    def synthetic_option_payoff(self, P, K0, F0, T, sigma):
        """Compute the synthetic option payoff with respect to K0."""
        if (K0 - 5 * abs(K0 - F0)) >= 0:

            K1 = K0 - 1.5 * abs(K0 - F0)  # K1 is 1.5 times further away from F than K0
            K2 = K0 - 5 * abs(K0 - F0)  # K2 is 5 times further away from F than K0

            # Calculate individual put option payoffs at each strike
            # sigma = self.implied_volatility_with_futures(P, self.F, T) # Assume IV is constant based on ATM vol
            payoff_K0 = self.black_scholes_put_price_with_futures(K0, T, sigma)
            payoff_K1 = self.black_scholes_put_price_with_futures(K1, T, sigma)
            payoff_K2 = self.black_scholes_put_price_with_futures(K2, T, sigma)

            # Calculate the synthetic payoff
            total_payoff = P * np.exp(-self.r*T) - payoff_K0 + 0.9 * payoff_K1 + 0.1 * payoff_K2

            return total_payoff

def plot_synthetic_option_payoff(r, P, K0, F0, T):
    # Plot the synthetic option payoff
    price_values = np.linspace(0, 2 * K0, 100)
    total_payoffs = []
    price_values_valid = []

    livestock_risk_protection = LivestockRiskProtection(r, K0)  # Calculate ATM vol
    if T == 0:
        sigma = 0
    else:
        sigma = livestock_risk_protection.implied_volatility_with_futures(P, K0, T)
    for price in price_values:
        livestock_risk_protection_new = LivestockRiskProtection(r, price)
        payoff = livestock_risk_protection_new.synthetic_option_payoff(P, K0, F0, T, sigma)
        try:
            if not math.isnan(float(payoff)):
                total_payoffs.append(float(payoff))
                price_values_valid.append(price)
        except:
            pass

    # normalize payoffs
    min_payoff = min(total_payoffs)
    max_payoff = max(total_payoffs)
    normalized_payoffs = [(i-min_payoff)/(max_payoff-min_payoff) for i in total_payoffs]
    plt.figure(figsize=(10, 6))
    plt.plot(price_values_valid, normalized_payoffs, label='Synthetic Option Payoff')
    plt.title("Normalized Synthetic Option Payoff/Premium vs. K0")
    plt.xlabel("K0 (Strike Price)")
    plt.ylabel("Synthetic Payoff/Premium")
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_synthetic_option_delta(r, P, K0, F0, T):
    # Plot the synthetic option payoff
    price_values = np.linspace(0, 2 * K0, 100)
    total_payoffs = []
    price_values_valid = []

    livestock_risk_protection = LivestockRiskProtection(r, K0)  # Calculate ATM vol
    if T == 0:
        sigma = 0
    else:
        sigma = livestock_risk_protection.implied_volatility_with_futures(P, K0, T)
    for price in price_values:
        livestock_risk_protection_new = LivestockRiskProtection(r, price)
        payoff = livestock_risk_protection_new.synthetic_put_delta(P, K0, F0, T, sigma)
        try:
            if not math.isnan(float(payoff)):
                total_payoffs.append(float(payoff))
                price_values_valid.append(price)
        except:
            pass

    plt.figure(figsize=(10, 6))
    plt.plot(price_values_valid, total_payoffs, label='Synthetic Option Payoff')
    plt.title("Synthetic Option Delta vs. K0")
    plt.xlabel("K0 (Strike Price)")
    plt.ylabel("Synthetic Delta")
    plt.grid(True)
    plt.legend()
    plt.show()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # T is faction_of_year
    plot_synthetic_option_payoff(0.02, 10, 100, 90, 1/2)  # payoff is normalized [0,1)
    plot_synthetic_option_delta(0.02, 10, 100, 90, 1/2)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
