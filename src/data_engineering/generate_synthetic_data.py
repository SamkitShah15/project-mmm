import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def generate_beauty_mmm_data(
    start_date="2022-01-01",
    periods=365*3,
    seed=42
):
    """
    Generates a synthetic dataset for a Beauty Brand's Marketing Mix Modeling.
    Includes separate channels for 'Performance' (FB, Google) and 'Brand' (TV, TikTok).
    Simulates adstock (lag) and saturation (diminishing returns) for the target variable 'Sales'.
    """
    np.random.seed(seed)
    
    # 1. Date Range
    dates = pd.date_range(start=start_date, periods=periods, freq='D')
    df = pd.DataFrame({'date': dates})
    
    # 2. Seasonality (Beauty industry peaks: Valentine's, Black Friday, Christmas)
    # Simple sine wave for general seasonality + spikes for events
    t = np.arange(periods)
    seasonality = 10000 + 2000 * np.sin(2 * np.pi * t / 365.25) 
    
    # Add spikes for Q4 (Nov-Dec)
    df['month'] = df['date'].dt.month
    q4_boost = df['month'].apply(lambda x: 1.5 if x in [11, 12] else 1.0)
    
    # 3. Marketing Spend Generation (Random Walk with drift to look realistic)
    def generate_spend(mean_spend, volatility, n):
        return np.maximum(0, np.random.normal(mean_spend, volatility, n))

    # Channels
    df['spend_facebook'] = generate_spend(2000, 500, periods) * q4_boost  # Always on
    df['spend_google_search'] = generate_spend(1500, 300, periods) * q4_boost # Capture demand
    df['spend_tiktok'] = generate_spend(3000, 1000, periods) # Pulsing campaigns
    # Make TikTok more "pulsy" - set random days to 0
    df.loc[np.random.choice(df.index, size=int(periods*0.4)), 'spend_tiktok'] = 0
    
    df['spend_tv'] = 0.0
    # pulsed TV campaigns (e.g., 2 weeks on, 6 weeks off)
    for i in range(0, periods, 60):
        if i+14 < periods:
            df.loc[i:i+14, 'spend_tv'] = 15000 # High spend bursts

    # 4. Impact Simulation (The "True" Model)
    # We define simple Adstock and Saturation functions to generate "Sales"
    # This ensures the data is "recoverable" by the model later.
    
    def apply_adstock(series, alpha):
        """Simple geometric decay"""
        return series.ewm(alpha=1-alpha, adjust=False).mean()
    
    def apply_saturation(series, beta):
        """Hill function or simple exponent"""
        return (series ** beta)

    # Coefficients (True Contribution)
    # Facebook: Moderate lag, moderate saturation
    fb_contribution = apply_saturation(apply_adstock(df['spend_facebook'], alpha=0.3), 0.8) * 4.0
    
    # Google: Low lag (immediate), high saturation (capture demand limit)
    google_contribution = apply_saturation(apply_adstock(df['spend_google_search'], alpha=0.1), 0.9) * 5.0
    
    # TikTok: High lag (viral effect), linear-ish saturation
    tiktok_contribution = apply_saturation(apply_adstock(df['spend_tiktok'], alpha=0.6), 0.7) * 3.5
    
    # TV: Vary high lag (brand building), strong saturation
    tv_contribution = apply_saturation(apply_adstock(df['spend_tv'], alpha=0.7), 0.6) * 2.0  # Lower ROI short term
    
    # Base Sales (Baseline without marketing)
    baseline = seasonality * q4_boost
    
    # 5. Total Sales Calculation
    noise = np.random.normal(0, 1000, periods) # Random noise
    df['sales'] = baseline + fb_contribution + google_contribution + tiktok_contribution + tv_contribution + noise
    
    return df

if __name__ == "__main__":
    print("Generating synthetic data...")
    df = generate_beauty_mmm_data()
    
    # Ensure directory exists
    output_dir = os.path.join("data", "bronze")
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "beauty_brand_mmm.csv")
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")
    print(df.head())
