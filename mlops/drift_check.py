# mlops/drift_check.py
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

def calculate_psi(expected, actual, buckets=10):
    """
    Calculate Population Stability Index (PSI) between expected and actual distributions.
    
    Args:
        expected: Reference distribution (pandas Series)
        actual: New distribution (pandas Series)
        buckets: Number of buckets for binning (default: 10)
    
    Returns:
        PSI value (float)
    """
    # Handle edge cases
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
    
    # Remove NaN values
    expected_clean = expected.dropna()
    actual_clean = actual.dropna()
    
    if len(expected_clean) == 0 or len(actual_clean) == 0:
        return 0.0
    
    # Create bins based on expected distribution quantiles
    try:
        # Use quantile-based binning for better distribution
        _, bin_edges = pd.qcut(expected_clean, q=buckets, retbins=True, duplicates='drop')
    except ValueError:
        try:
            # Fallback to equal-width binning
            _, bin_edges = pd.cut(expected_clean, bins=buckets, retbins=True, duplicates='drop')
        except ValueError:
            # If all values are the same, no drift
            return 0.0
    
    # Ensure we have valid bins
    if len(bin_edges) <= 1:
        return 0.0
    
    # Bin both distributions using the same edges
    expected_binned = pd.cut(expected_clean, bins=bin_edges, include_lowest=True)
    actual_binned = pd.cut(actual_clean, bins=bin_edges, include_lowest=True)
    
    # Calculate percentages for each bin
    expected_counts = expected_binned.value_counts(normalize=True, sort=False)
    actual_counts = actual_binned.value_counts(normalize=True, sort=False)
    
    # Align indices and fill missing bins with small value
    all_bins = expected_counts.index.union(actual_counts.index)
    expected_pct = expected_counts.reindex(all_bins, fill_value=0)
    actual_pct = actual_counts.reindex(all_bins, fill_value=0)
    
    # Add small epsilon to avoid log(0) and division by 0
    epsilon = 1e-6
    expected_pct = expected_pct + epsilon
    actual_pct = actual_pct + epsilon
    
    # Calculate PSI: Σ((actual% - expected%) * ln(actual% / expected%))
    psi = sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    
    return float(psi)

def select_drift_columns(df):
    """
    Select relevant columns for drift detection.
    Excludes target columns and non-numeric columns.
    """
    # Exclude common target column names
    exclude_cols = {'Class', 'class', 'target', 'label', 'y'}
    
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Filter out excluded columns
    drift_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    return drift_cols

def main():
    parser = argparse.ArgumentParser(
        description="Check dataset drift using Population Stability Index (PSI)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python mlops/drift_check.py new_data.csv
  python mlops/drift_check.py new_data.csv --ref-data reference.csv
  REF_DATA=ref.csv DRIFT_PSI_THRESHOLD=0.1 python mlops/drift_check.py new.csv
        """
    )
    parser.add_argument("new_data_path", help="Path to new dataset to check for drift")
    parser.add_argument("--ref-data", help="Path to reference dataset (overrides REF_DATA env var)")
    args = parser.parse_args()
    
    # Get configuration
    ref_data_path = args.ref_data or os.getenv("REF_DATA", "data/data_raw.csv")
    threshold = float(os.getenv("DRIFT_PSI_THRESHOLD", "0.2"))
    
    try:
        # Load datasets
        ref_df = pd.read_csv(ref_data_path)
        new_df = pd.read_csv(args.new_data_path)
        
        # Select columns for drift detection
        ref_cols = select_drift_columns(ref_df)
        new_cols = select_drift_columns(new_df)
        common_cols = list(set(ref_cols) & set(new_cols))
        
        if not common_cols:
            result = {
                "status": "error",
                "message": "No common numeric columns found for drift detection",
                "mean_psi": 0.0,
                "drift_exceeded": False,
                "threshold": threshold,
                "ref_data_path": ref_data_path,
                "new_data_path": args.new_data_path
            }
        else:
            # Calculate PSI for each column
            psi_values = {}
            valid_psi_count = 0
            
            for col in common_cols:
                try:
                    psi = calculate_psi(ref_df[col], new_df[col])
                    psi_values[col] = round(psi, 4)
                    if not np.isnan(psi) and not np.isinf(psi):
                        valid_psi_count += 1
                except Exception as e:
                    psi_values[col] = f"error: {str(e)}"
            
            # Calculate mean PSI from valid values only
            valid_psi_values = [v for v in psi_values.values() if isinstance(v, (int, float))]
            mean_psi = np.mean(valid_psi_values) if valid_psi_values else 0.0
            drift_exceeded = mean_psi > threshold
            
            result = {
                "status": "success",
                "mean_psi": round(float(mean_psi), 4),
                "drift_exceeded": drift_exceeded,
                "threshold": threshold,
                "columns_checked": len(common_cols),
                "valid_columns": valid_psi_count,
                "ref_data_path": ref_data_path,
                "new_data_path": args.new_data_path,
                "psi_by_column": psi_values
            }
    
    except FileNotFoundError as e:
        result = {
            "status": "error",
            "message": f"File not found: {str(e)}",
            "mean_psi": 0.0,
            "drift_exceeded": False,
            "threshold": threshold,
            "ref_data_path": ref_data_path,
            "new_data_path": args.new_data_path
        }
    
    except Exception as e:
        result = {
            "status": "error",
            "message": f"Unexpected error: {str(e)}",
            "mean_psi": 0.0,
            "drift_exceeded": False,
            "threshold": threshold,
            "ref_data_path": ref_data_path,
            "new_data_path": args.new_data_path
        }
    
    # Print one-line JSON summary to stdout
    print(json.dumps(result, separators=(',', ':')))
    
    # Write GitHub Actions output if running in CI
    github_output = os.getenv("GITHUB_OUTPUT")
    if github_output:
        try:
            with open(github_output, "a") as f:
                f.write(f"drift_exceeded={str(result['drift_exceeded']).lower()}\n")
                f.write(f"mean_psi={result['mean_psi']}\n")
        except Exception:
            # Silently ignore GitHub Actions output errors
            pass
    
    # Always exit with code 0 for conditional branching
    sys.exit(0)

if __name__ == "__main__":
    main()