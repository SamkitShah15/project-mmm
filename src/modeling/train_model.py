import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import os
import pickle

def train_mmm_model(data_path, output_dir):
    """
    Trains a Bayesian Marketing Mix Model using PyMC.
    
    Parameters:
    - data_path: Path to the engineering 'Gold' data (parquet).
    - output_dir: Directory to save the model trace and artifacts.
    """
    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    
    # Filter for the last 2 years to ensure relevance
    # df = df[df['date'] >= '2023-01-01'] 
    
    # --- Data Preparation ---
    # We use the *pre-engineered* features from the Gold layer for this version.
    # In a more advanced "End-to-End" Bayesian model, we would learn the Alpha/Beta 
    # parameters inside the model. For this "MVP", we use the calculated features 
    # to demonstrate the regression component clearly.
    
    # Target
    y = df['sales'].values
    
    # Features (Media)
    # Note: We use the 'saturated' columns which already have Adstock + Saturation applied
    media_cols = ['spend_facebook_saturated', 'spend_google_search_saturated', 
                  'spend_tiktok_saturated', 'spend_tv_saturated']
    
    X_media = df[media_cols].values
    
    # Features (Control)
    # Seasonality (Sine/Cosine or Month/Week dummies could be added here)
    # For now, we'll keep it simple to the media effects.
    
    print("Building Bayesian Model...")
    
    with pm.Model() as mmm_model:
        # --- 1. Priors (The "Industry Knowledge") ---
        
        # Intercept (Baseline Sales)
        intercept = pm.Normal("intercept", mu=np.mean(y), sigma=np.std(y))
        
        # Media Coefficients (Impact)
        # We use HalfNormal because media cannot have a negative impact on sales.
        # This is a key advantage over OLS regression which often gives negative coefficients.
        media_coefs = pm.HalfNormal("media_coefs", sigma=10.0, shape=len(media_cols))
        
        # Noise (Error term)
        sigma = pm.HalfNormal("sigma", sigma=np.std(y))
        
        # --- 2. Likelihood (The "Data Fit") ---
        
        # Expected value formula: y = intercept + sum(media * coef)
        mu = intercept + pm.math.dot(X_media, media_coefs)
        
        # Likelihood
        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)
        
        # --- 3. Sampling (The "Computation") ---
        print("Running NUTS Sampler (MCMC)... this might take a minute.")
        # draws=1000, tune=1000 is standard for production. 
        # using smaller numbers for quick testing/demo purposes if needed.
        print("Running ADVI (Variational Inference) for speed on CPU...")
        try:
            inference = pm.ADVI()
            approx = pm.fit(n=1000, method=inference, random_seed=42)
            trace = approx.sample(draws=1000)
        except Exception as e:
            print(f"ADVI failed, falling back to very short NUTS: {e}")
            trace = pm.sample(draws=20, tune=20, chains=1, random_seed=42)
        
    # --- 4. Diagnostics & Saving ---
    print("Sampling complete. Saving trace...")
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "mmm_trace.nc")
    az.to_netcdf(trace, output_path)
    
    # Summary
    summary = az.summary(trace, var_names=["media_coefs", "intercept"])
    summary.index = ['intercept'] + media_cols # Rename indices for clarity
    print("\n--- Model Summary ---")
    print(summary)
    
    summary_path = os.path.join(output_dir, "model_summary.csv")
    summary.to_csv(summary_path)
    print(f"Model trace saved to {output_path}")
    print(f"Summary saved to {summary_path}")
    
    return trace

if __name__ == "__main__":
    gold_path = os.path.join("data", "gold", "beauty_brand_mmm_features.parquet")
    models_dir = os.path.join("data", "models")
    
    train_mmm_model(gold_path, models_dir)
