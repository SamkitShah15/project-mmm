import pandas as pd
import numpy as np
import os

def geometric_adstock(x, alpha):
    """
    Computes the Geometric Adstock (Lag) transformation.
    x: Input array (spend)
    alpha: Decay rate (0 to 1). Higher alpha = longer lag.
    """
    x_decayed = np.zeros_like(x)
    x_decayed[0] = x[0]
    for t in range(1, len(x)):
        x_decayed[t] = x[t] + alpha * x_decayed[t-1]
    return x_decayed

def hill_saturation(x, slope, kappa):
    """
    Computes the Hill Saturation transformation.
    x: Input array (spend/adstock)
    slope: Controls the steepness of the curve (S-shape).
    kappa: Half-saturation point (where impact is 50% of max).
    """
    return 1 / (1 + (kappa / (x + 1e-9))**slope)

def process_gold(input_path, output_path):
    """
    Processes Silver data into Gold data.
    - Applies feature engineering (Adstock, Saturation) for analysis.
    - Scales variables (optional but good for some models).
    - Prepares the final modeling dataset.
    """
    print(f"Reading Silver data from {input_path}...")
    df = pd.read_parquet(input_path)
    
    # Define channels
    channels = ['spend_facebook', 'spend_google_search', 'spend_tiktok', 'spend_tv']
    
    # --- Feature Engineering Demonstrations ---
    # We create "Pre-computed" features just to demonstrate the engineering skill.
    # In the actual Bayesian model, we let the model learn alpha/beta, 
    # but these are useful for "Heuristic" analysis or linear baselines.
    
    print("Applying Feature Engineering (Adstock & Saturation demos)...")
    
    # 1. Adstock (Lag)
    # Assumptions: TV has long memory (0.7), Digital has short memory (0.2-0.4)
    alphas = {
        'spend_facebook': 0.3,
        'spend_google_search': 0.1,
        'spend_tiktok': 0.5,
        'spend_tv': 0.7
    }
    
    for col, alpha in alphas.items():
        df[f'{col}_adstock'] = geometric_adstock(df[col].values, alpha)
        
    # 2. Saturation (Diminishing Returns)
    # We apply saturation on the adstocked variables
    for col in channels:
        adstock_col = f'{col}_adstock'
        # Heuristic: kappa is roughly the mean of the spend
        kappa = df[adstock_col].mean()
        slope = 1.0 # Linear-ish
        df[f'{col}_saturated'] = hill_saturation(df[adstock_col].values, slope=slope, kappa=kappa)
        
    # 3. Aggregations (e.g., Weekly)
    # MMM is often done at a weekly level to reduce noise.
    df_weekly = df.resample('W-MON', on='date').sum().reset_index()
    
    print(f"Gold data (Daily) shape: {df.shape}")
    print(f"Gold data (Weekly) shape: {df_weekly.shape}")
    
    # Save both Daily and Weekly versions
    df.to_parquet(output_path, index=False)
    
    weekly_path = output_path.replace('.parquet', '_weekly.parquet')
    df_weekly.to_parquet(weekly_path, index=False)
    
    print(f"Gold data saved to {output_path} and {weekly_path}")

if __name__ == "__main__":
    silver_path = os.path.join("data", "silver", "beauty_brand_mmm.parquet")
    gold_dir = os.path.join("data", "gold")
    os.makedirs(gold_dir, exist_ok=True)
    gold_path = os.path.join(gold_dir, "beauty_brand_mmm_features.parquet")
    
    process_gold(silver_path, gold_path)
