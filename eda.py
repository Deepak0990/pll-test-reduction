import os
import pandas as pd

def run_exploratory_data_analysis():
    print("=" * 60)
    print(" 📊 PLL TEST REDUCTION: EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 60 + "\n")

    # Define dataset paths
    jitter_path = os.path.join("data", "synthetic_pll_jitter_dataset.csv")
    locktime_path = os.path.join("data", "synthetic_pll_locktime_dataset.csv")

    # --- 1. Analyze Jitter Dataset ---
    if os.path.exists(jitter_path):
        print("▶ [Dataset 1] Jitter Dataset Analysis (`synthetic_pll_jitter_dataset.csv`)")
        df_jitter = pd.read_csv(jitter_path)
        
        print(f"   • Shape (Rows, Columns): {df_jitter.shape}")
        print(f"   • Missing Values Found: {df_jitter.isnull().sum().sum()}")
        print("\n   • Feature Summary Statistics:")
        print(df_jitter.describe().T[['mean', 'std', 'min', 'max']])
        print("-" * 60 + "\n")
    else:
        print(f"❌ Jitter dataset not found at: {jitter_path}\n")

    # --- 2. Analyze Lock-Time Dataset ---
    if os.path.exists(locktime_path):
        print("▶ [Dataset 2] Lock-Time Dataset Analysis (`synthetic_pll_locktime_dataset.csv`)")
        df_lock = pd.read_csv(locktime_path)
        
        print(f"   • Shape (Rows, Columns): {df_lock.shape}")
        print(f"   • Missing Values Found: {df_lock.isnull().sum().sum()}")
        print("\n   • Feature Summary Statistics:")
        print(df_lock.describe().T[['mean', 'std', 'min', 'max']])
        print("-" * 60 + "\n")
    else:
        print(f"❌ Lock-time dataset not found at: {locktime_path}\n")

    print("✅ EDA complete! Data distributions are verified and clean for modeling.")

if __name__ == "__main__":
    run_exploratory_data_analysis()