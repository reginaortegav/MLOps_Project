import pandas as pd
from scipy.stats import ks_2samp

def detect_drift(reference_df: pd.DataFrame, new_df: pd.DataFrame, threshold=0.05):
    """
    Compare feature distributions between reference and new data.
    Returns True if drift is detected in any feature.
    """
    drift_detected = False
    for col in reference_df.columns:
        if reference_df[col].dtype in ['float64', 'int64']:
            stat, p_value = ks_2samp(reference_df[col], new_df[col])
            if p_value < threshold:
                drift_detected = True
                print(f"Drift detected in {col} (p={p_value})")
    return drift_detected