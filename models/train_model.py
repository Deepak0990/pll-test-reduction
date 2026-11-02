import os
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
import pickle

def main():
    print("--- 1. Training Jitter Test Elimination Model ---")
    df_jitter = pd.read_csv("data/synthetic_pll_jitter_dataset.csv")
    
    features = [
        "vco_band_code", "ring_osc_ghz", "chargepump_ua", 
        "lock_time_us", "vdd_droop_norm", "wafer_radius_norm", "leak_screen_norm"
    ]
    target = "rms_jitter_ps"
    usl = 2.2
    
    train_df = df_jitter[df_jitter["split"] == "train"]
    cal_df = df_jitter[df_jitter["split"] == "calibration"]
    test_df = df_jitter[df_jitter["split"] == "test"]
    
    X_train, y_train = train_df[features], train_df[target]
    X_cal, y_cal = cal_df[features], cal_df[target]
    X_test, y_test = test_df[features], test_df[target]
    
    model = LGBMRegressor(
        n_estimators=250, max_depth=3, learning_rate=0.06, 
        subsample=0.8, random_state=42
    )
    model.fit(X_train, y_train)
    
    leak_gate = 2.0
    cal_preds = model.predict(X_cal)
    cal_mask = X_cal["leak_screen_norm"] < leak_gate
    residuals = np.abs(y_cal - cal_preds)[cal_mask]
    
    alpha = 0.008
    q_alpha = np.quantile(residuals, 1 - alpha)
    
    test_preds = model.predict(X_test)
    test_mask = X_test["leak_screen_norm"] < leak_gate
    upper_bounds = test_preds + q_alpha
    
    skipped = (upper_bounds < usl) & test_mask
    true_fails = y_test > usl
    escapes = (skipped & true_fails).sum()
    dppm = (escapes / len(X_test)) * 1e6
    skip_rate = skipped.mean() * 100
    
    print(f"  -> Jitter Skip Rate: {skip_rate:.2f}%")
    print(f"  -> Jitter Escapes: {dppm:.1f} DPPM (Target <= 20)")
    
    print("\n--- 2. Analyzing Adaptive Lock-Time Dataset ---")
    df_lock = pd.read_csv("data/synthetic_pll_locktime_dataset.csv")
    mean_fixed = df_lock["fixed_wait_us"].mean()
    mean_adaptive = df_lock["adaptive_wait_us"].mean()
    lock_savings_pct = (1 - (mean_adaptive / mean_fixed)) * 100
    print(f"  -> Mean Fixed Wait: {mean_fixed:.2f} µs")
    print(f"  -> Mean Adaptive Wait: {mean_adaptive:.2f} µs")
    print(f"  -> Lock-Time Reduction: {lock_savings_pct:.1f}%")
    
    os.makedirs("models", exist_ok=True)
    artifacts = {
        "model": model,
        "features": features,
        "q_alpha": q_alpha,
        "usl": usl,
        "leak_gate": leak_gate,
        "lock_stats": {
            "mean_fixed": mean_fixed,
            "mean_adaptive": mean_adaptive,
            "savings_pct": lock_savings_pct
        }
    }
    with open("models/pll_jitter_model.pkl", "wb") as f:
        pickle.dump(artifacts, f)
    print("\nModel & stats saved successfully to models/pll_jitter_model.pkl")

if __name__ == "__main__":
    main()