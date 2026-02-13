import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import os
import json
import matplotlib.pyplot as plt

def calibrate_mmm_model(data_path, experiment_path, output_dir):
    """
    Retrains the MMM model using the Geo-Experiment results as an INFORMATIVE PRIOR.
    """
    print("Loading data and experimental results...")
    df = pd.read_parquet(data_path)
    
    with open(experiment_path, 'r') as f:
        experiment_data = json.load(f)
    
    experiment_channel = experiment_data['channel'] # 'spend_tiktok'
    experiment_roas = experiment_data['experiment_roas']
    experiment_std = experiment_data['std_error']
    
    print(f"Calibrating model for {experiment_channel}...")
    print(f"Using Informative Prior: Normal(mu={experiment_roas}, sigma={experiment_std})")
    
    # --- Data Prep (Same as train_model.py) ---
    y = df['sales'].values
    
    # Needs to match the order in train_model.py for consistency
    media_cols = ['spend_facebook_saturated', 'spend_google_search_saturated', 
                  'spend_tiktok_saturated', 'spend_tv_saturated']
    
    X_media = df[media_cols].values
    
    # Find index of the experiment channel
    # Note: Our 'experiment_channel' is 'spend_tiktok', but media_cols has '_saturated'
    # In simulate_geo_experiment we saved 'spend_tiktok', but here we need to map it.
    # START HACK: For this demo, we know 'spend_tiktok_saturated' is the 3rd column (index 2)
    # in a real system we would map this dynamically.
    tiktok_idx = 2 
    
    print("Building Calibrated Bayesian Model...")
    
    with pm.Model() as calibrated_model:
        # --- Priors ---
        intercept = pm.Normal("intercept", mu=np.mean(y), sigma=np.std(y))
        sigma = pm.HalfNormal("sigma", sigma=np.std(y))
        
        # KEY CHANGE: We split the media coefficients into "Uncalibrated" and "Calibrated"
        # 1. Uncalibrated Channels (Facebook, Google, TV) -> Default HalfNormal
        # 2. Calibrated Channel (TikTok) -> Normal Prior focused on Experiment Result
        
        # We'll define them individually for clarity in the graph
        coef_fb = pm.HalfNormal("coef_fb", sigma=10)
        coef_google = pm.HalfNormal("coef_google", sigma=10)
        
        # THE CALIBRATION IS HERE:
        # We tighten the sigma (uncertainty) significantly around the experimental truth
        coef_tiktok = pm.Normal("coef_tiktok", mu=experiment_roas, sigma=experiment_std)
        
        coef_tv = pm.HalfNormal("coef_tv", sigma=10)
        
        # Stack them back for the dot product
        # Order: FB, Google, TikTok, TV
        media_coefs = pm.math.stack([coef_fb, coef_google, coef_tiktok, coef_tv])
        
        # --- Likelihood ---
        mu = intercept + pm.math.dot(X_media, media_coefs)
        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)
        
        # --- Sampling (ADVI) ---
        print("Running ADVI Calibration...")
        inference = pm.ADVI()
        approx = pm.fit(n=5000, method=inference, random_seed=42)
        trace = approx.sample(draws=1000)
        
    # --- Saving ---
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "calibrated_trace.nc")
    az.to_netcdf(trace, output_path)
    
    # Summary
    # We reconstruct the summary manually to include proper names
    summary = az.summary(trace, var_names=["coef_fb", "coef_google", "coef_tiktok", "coef_tv", "intercept"])
    print("\n--- Calibrated Model Summary ---")
    print(summary)
    
    summary_path = os.path.join(output_dir, "calibrated_summary.csv")
    summary.to_csv(summary_path)
    print(f"Calibrated trace saved to {output_path}")

if __name__ == "__main__":
    gold_path = os.path.join("data", "gold", "beauty_brand_mmm_features.parquet")
    models_dir = os.path.join("data", "models")
    experiment_file = os.path.join(models_dir, "experiment_results.json")
    
    calibrate_mmm_model(gold_path, experiment_file, models_dir)
