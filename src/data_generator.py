import numpy as np
import pandas as pd

def generate_synthetic_data(n_samples=10000, random_state=42):
    """
    Generate synthetic data for MBDP prediction.
    Features: ga, bw, sga, sex, last_alp, mean_alp, slope_alp,
              last_phosphate, cum_pn_days, weight_gain_velocity
    Target: mbdp (binary)
    """
    np.random.seed(random_state)
    ga = np.random.normal(32, 3, n_samples)
    bw = np.random.normal(1800, 500, n_samples)
    sga = np.random.binomial(1, 0.2, n_samples)
    sex = np.random.binomial(1, 0.5, n_samples)
    last_alp = np.random.normal(300, 100, n_samples)
    mean_alp = np.random.normal(280, 80, n_samples)
    slope_alp = np.random.normal(5, 10, n_samples)
    last_phosphate = np.random.normal(1.5, 0.5, n_samples)
    cum_pn_days = np.random.poisson(10, n_samples)

    # Weight gain velocity: typical preterm growth ~15-30 g/day.
    weight_gain_velocity = np.random.normal(20, 5, n_samples)

    logit = (
        -3.0
        + 0.02 * (last_alp - 300) / 100
        - 0.5 * (last_phosphate - 1.5)
        + 0.1 * cum_pn_days
        - 0.05 * (ga - 32)
        - 0.0005 * (bw - 1800)
        + 0.5 * sga
        + 0.2 * sex
        + 0.01 * slope_alp
        - 0.1 * (weight_gain_velocity - 20)   # slower growth -> higher risk
    )
    prob = 1 / (1 + np.exp(-logit))
    mbdp = np.random.binomial(1, prob)

    df = pd.DataFrame({
        'ga': ga,
        'bw': bw,
        'sga': sga,
        'sex': sex,
        'last_alp': last_alp,
        'mean_alp': mean_alp,
        'slope_alp': slope_alp,
        'last_phosphate': last_phosphate,
        'cum_pn_days': cum_pn_days,
        'weight_gain_velocity': weight_gain_velocity,
        'mbdp': mbdp
    })
    return df

if __name__ == "__main__":
    df = generate_synthetic_data(10000)
    df.to_csv("data/train.csv", index=False)
    print("Synthetic data saved to data/train.csv")
    print(df['mbdp'].value_counts())