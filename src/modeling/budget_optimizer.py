import pandas as pd
import numpy as np
import scipy.optimize as optimize
import os

def geometric_adstock(x, alpha):
    """Re-implementation of adstock for optimization loop"""
    x_decayed = np.zeros_like(x)
    x_decayed[0] = x[0]
    for t in range(1, len(x)):
        x_decayed[t] = x[t] + alpha * x_decayed[t-1]
    return x_decayed

def hill_saturation(x, slope, kappa):
    """Re-implementation of saturation for optimization loop"""
    return 1 / (1 + (kappa / (x + 1e-9))**slope)

def budget_optimizer(model_path, data_path, total_budget=None):
    """
    Optimizes the marketing budget allocation to maximize Revenue.
    """
    print("Loading calibrated model and data...")
    summary = pd.read_csv(model_path, index_col=0)
    df = pd.read_parquet(data_path)
    
    # Extract Coefficients
    coefs = {
        'spend_facebook': summary.loc['coef_fb', 'mean'],
        'spend_google_search': summary.loc['coef_google', 'mean'],
        'spend_tiktok': summary.loc['coef_tiktok', 'mean'], # The calibrated one!
        'spend_tv': summary.loc['coef_tv', 'mean']
    }
    
    print("Optimization Parameters (Coefficients):")
    print(coefs)
    
    # Extract Hyperparameters (Hardcoded from process_gold.py for this MVP)
    alphas = {
        'spend_facebook': 0.3,
        'spend_google_search': 0.1,
        'spend_tiktok': 0.5,
        'spend_tv': 0.7
    }
    
    # Calculate Kappas from data (mean of adstock)
    kappas = {}
    for col, alpha in alphas.items():
        adstock = geometric_adstock(df[col].values, alpha)
        kappas[col] = adstock.mean()
        
    channels = list(coefs.keys())
    
    # Current Spend (Last 30 days usually, but here we take average daily * 7 for a weekly budget)
    # Let's optimize for a single "Average Day" spend allocation
    current_avg_spend = {col: df[col].mean() for col in channels}
    total_current_budget = sum(current_avg_spend.values())
    
    if total_budget is None:
        total_budget = total_current_budget
    
    print(f"\nTotal Budget to Optimize: ${total_budget:,.2f} (Daily Avg)")
    
    # Define Objective Function: Maximize Revenue
    # Revenue = Sum(Hill(Adstock(Spend)) * Coef)
    # Note: For single-day optimization, Adstock approximates to Spend / (1-alpha) in steady state? 
    # Actually, let's keep it simple: Optimize "Effective Spend" given saturation.
    # Assuming steady state: Adstock Level = Spend / (1 - Alpha)
    
    def objective_function(spends):
        total_revenue = 0
        for i, col in enumerate(channels):
            spend = spends[i]
            # Steady state adstock
            steady_adstock = spend / (1 - alphas[col]) 
            # Saturation
            saturation = hill_saturation(steady_adstock, slope=1.0, kappa=kappas[col])
            # Revenue
            revenue = saturation * coefs[col]
            total_revenue += revenue
        return -total_revenue # Minimize negative revenue
    
    # Constraints
    # 1. Sum of spend <= Total Budget
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - total_budget})
    
    # Bounds (Don't let any channel go to 0 or infinite)
    # +/- 50% of current spend to be realistic
    bounds = []
    for col in channels:
        avg = current_avg_spend[col]
        bounds.append((avg * 0.5, avg * 1.5))
        
    # Initial Guess
    x0 = [current_avg_spend[col] for col in channels]
    
    print(f"Initial Revenue (x0): {-objective_function(x0):,.2f}")
    
    print("Running SQP Optimization...")
    result = optimize.minimize(objective_function, x0, method='SLSQP', bounds=bounds, constraints=constraints, options={'disp': True})
    
    print(f"Optimization Status: {result.message}")
    print(f"Optimization Success: {result.success}")
    
    # Results
    print("\n--- Optimized Budget Allocation ---")
    optimized_spends = result.x
    
    print(f"{'Channel':<20} | {'Old Spend':<12} | {'New Spend':<12} | {'Delta':<8}")
    print("-" * 60)
    
    for i, col in enumerate(channels):
        old = x0[i]
        new = optimized_spends[i]
        delta = ((new - old) / old) * 100
        print(f"{col:<20} | ${old:,.2f}   | ${new:,.2f}   | {delta:>+5.1f}%")
        
    print("-" * 60)
    print(f"Projected Revenue Lift: {(-result.fun - -objective_function(x0)):,.2f}")

if __name__ == "__main__":
    models_dir = os.path.join("data", "models")
    model_path = os.path.join(models_dir, "calibrated_summary.csv")
    gold_path = os.path.join("data", "gold", "beauty_brand_mmm_features.parquet")
    
    if os.path.exists(model_path):
        budget_optimizer(model_path, gold_path)
    else:
        print("Model summary not found. Run calibrate_model.py first.")
