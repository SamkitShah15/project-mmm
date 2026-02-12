import pandas as pd
import numpy as np
import os

def process_silver(input_path, output_path):
    """
    Processes Bronze data into Silver data.
    - Validates schema
    - Handles missing values
    - Enforces data types
    """
    print(f"Reading Bronze data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # 1. Validation
    required_columns = ['date', 'sales', 'spend_facebook', 'spend_google_search', 'spend_tiktok', 'spend_tv']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # 2. Type Enforcing
    df['date'] = pd.to_datetime(df['date'])
    
    # 3. Data Cleaning (Example: Ensure no negative spend/sales)
    numeric_cols = ['sales', 'spend_facebook', 'spend_google_search', 'spend_tiktok', 'spend_tv']
    for col in numeric_cols:
        if (df[col] < 0).any():
            print(f"Warning: Negative values found in {col}. Clipping to 0.")
            df[col] = df[col].clip(lower=0)
            
    # 4. Feature Engineering (Basic date features for Silver)
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    
    print(f"Silver data shape: {df.shape}")
    print(df.head())
    
    # Save to Parquet for performance
    df.to_parquet(output_path, index=False)
    print(f"Silver data saved to {output_path}")

if __name__ == "__main__":
    bronze_path = os.path.join("data", "bronze", "beauty_brand_mmm.csv")
    silver_dir = os.path.join("data", "silver")
    os.makedirs(silver_dir, exist_ok=True)
    silver_path = os.path.join(silver_dir, "beauty_brand_mmm.parquet")
    
    process_silver(bronze_path, silver_path)
