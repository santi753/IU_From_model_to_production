# mlops/simulate_monthly_data.py
"""
Simulate 12 months of credit card transaction data with gradual drift.

This script reads the original dataset and generates monthly datasets with
progressively increasing drift in feature distributions while preserving
the original schema and Class labels.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

def apply_drift(df, month_num, seed=None):
    """
    Apply gradual drift to a dataset based on month number.
    
    Args:
        df: Original DataFrame
        month_num: Month number (1-12) - higher means more drift
        seed: Random seed for reproducibility
    
    Returns:
        DataFrame with drift applied
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Create a copy to avoid modifying original
    df_drifted = df.copy()
    
    # Calculate drift factor (0.0 for month 1, increasing to ~1.0 for month 12)
    drift_factor = (month_num - 1) / 11.0
    
    # Apply drift to Time column
    # Shift: move time values forward (simulating later transactions)
    time_shift = drift_factor * 86400  # Up to 24 hours shift by month 12
    # Scale: increase variance (transactions spread over wider time range)
    time_scale = 1.0 + (drift_factor * 0.3)  # Up to 30% increase in spread
    
    time_mean = df_drifted['Time'].mean()
    df_drifted['Time'] = (df_drifted['Time'] - time_mean) * time_scale + time_mean + time_shift
    df_drifted['Time'] = df_drifted['Time'].clip(lower=0)  # Ensure non-negative
    
    # Apply drift to Amount column
    # Shift: increase average transaction amount (inflation/behavior change)
    amount_shift_factor = 1.0 + (drift_factor * 0.15)  # Up to 15% increase
    # Scale: increase variance (more diverse transaction amounts)
    amount_scale_factor = 1.0 + (drift_factor * 0.2)  # Up to 20% increase in variance
    
    amount_mean = df_drifted['Amount'].mean()
    amount_std = df_drifted['Amount'].std()
    
    # Apply log-normal transformation to maintain realistic distribution
    log_amounts = np.log1p(df_drifted['Amount'])  # log1p to handle zeros
    log_mean = log_amounts.mean()
    log_std = log_amounts.std()
    
    # Apply drift in log space
    log_amounts = (log_amounts - log_mean) * amount_scale_factor + log_mean
    log_amounts = log_amounts + np.log(amount_shift_factor)
    
    # Transform back
    df_drifted['Amount'] = np.expm1(log_amounts).clip(lower=0)
    
    # Apply drift to V1-V28 columns (PCA components)
    for i in range(1, 29):
        col = f'V{i}'
        
        # Different drift patterns for different components
        if i <= 5:
            # Stronger drift for first few components (most important)
            # Shift mean
            shift = drift_factor * 0.3 * np.sin(i * np.pi / 6)  # Oscillating shifts
            # Scale variance
            scale = 1.0 + (drift_factor * 0.25 * (1 + 0.1 * i))
            # Add noise
            noise_std = drift_factor * 0.15
        elif i <= 15:
            # Moderate drift for middle components
            shift = drift_factor * 0.2 * np.cos(i * np.pi / 8)
            scale = 1.0 + (drift_factor * 0.15)
            noise_std = drift_factor * 0.1
        else:
            # Minimal drift for least important components
            shift = drift_factor * 0.1 * np.sin(i * np.pi / 10)
            scale = 1.0 + (drift_factor * 0.05)
            noise_std = drift_factor * 0.05
        
        # Apply transformations
        col_mean = df_drifted[col].mean()
        col_std = df_drifted[col].std()
        
        # Standardize, apply drift, then destandardize
        standardized = (df_drifted[col] - col_mean) / (col_std + 1e-8)
        drifted = standardized * scale + shift
        df_drifted[col] = drifted * col_std + col_mean
        
        # Add Gaussian noise
        noise = np.random.normal(0, noise_std, len(df_drifted))
        df_drifted[col] = df_drifted[col] + noise
    
    # Class column remains unchanged (as specified)
    # This simulates that fraud patterns stay similar but feature distributions change
    
    return df_drifted

def generate_monthly_datasets(input_path, output_dir, num_months=12, sample_size=None):
    """
    Generate monthly datasets with progressive drift.
    
    Args:
        input_path: Path to original dataset
        output_dir: Directory to save monthly datasets
        num_months: Number of months to generate (default: 12)
        sample_size: If specified, sample this many rows per month
    
    Returns:
        List of paths to generated datasets
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load original data
    print(f"Loading original data from: {input_path}")
    df_original = pd.read_csv(input_path)
    print(f"Original data shape: {df_original.shape}")
    print(f"Original columns: {list(df_original.columns)}")
    
    # Sample if requested (for faster testing)
    if sample_size and sample_size < len(df_original):
        print(f"Sampling {sample_size} rows from original data...")
        df_original = df_original.sample(n=sample_size, random_state=42)
    
    # Calculate statistics for the original data
    print("\nOriginal data statistics:")
    print(f"  Time: mean={df_original['Time'].mean():.2f}, std={df_original['Time'].std():.2f}")
    print(f"  Amount: mean={df_original['Amount'].mean():.2f}, std={df_original['Amount'].std():.2f}")
    print(f"  V1: mean={df_original['V1'].mean():.4f}, std={df_original['V1'].std():.4f}")
    print(f"  Class distribution: {df_original['Class'].value_counts().to_dict()}")
    
    generated_paths = []
    
    print(f"\nGenerating {num_months} monthly datasets with progressive drift...")
    print("-" * 60)
    
    for month in range(1, num_months + 1):
        # Apply drift based on month number
        df_month = apply_drift(df_original, month, seed=42 + month)
        
        # Generate filename
        month_str = f"{month:02d}"
        filename = f"month_{month_str}.csv"
        filepath = output_dir / filename
        
        # Save dataset
        df_month.to_csv(filepath, index=False)
        generated_paths.append(str(filepath))
        
        # Print statistics for this month
        print(f"\nMonth {month_str}:")
        print(f"  Saved to: {filepath}")
        print(f"  Shape: {df_month.shape}")
        print(f"  Drift factor: {((month - 1) / 11.0):.2%}")
        print(f"  Time: mean={df_month['Time'].mean():.2f}, std={df_month['Time'].std():.2f}")
        print(f"  Amount: mean={df_month['Amount'].mean():.2f}, std={df_month['Amount'].std():.2f}")
        print(f"  V1: mean={df_month['V1'].mean():.4f}, std={df_month['V1'].std():.4f}")
        
        # Calculate PSI preview (simplified)
        v1_drift = abs(df_month['V1'].mean() - df_original['V1'].mean()) / df_original['V1'].std()
        amount_drift = abs(df_month['Amount'].mean() - df_original['Amount'].mean()) / df_original['Amount'].std()
        print(f"  Drift indicators: V1_shift={v1_drift:.3f}, Amount_shift={amount_drift:.3f}")
    
    print("\n" + "=" * 60)
    print("Monthly data generation complete!")
    print(f"Generated {len(generated_paths)} datasets in: {output_dir}")
    
    return generated_paths

def main():
    """Main entry point for the script."""
    # Configuration
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    
    # Paths
    input_path = project_root / "data" / "data_raw.csv"
    output_dir = project_root / "data" / "simulated"
    
    # Check if input file exists
    if not input_path.exists():
        print(f"Error: Input file not found at {input_path}")
        print("Please ensure data/data_raw.csv exists.")
        sys.exit(1)
    
    # Parse command-line arguments (optional)
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate monthly datasets with progressive drift",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--sample-size", 
        type=int, 
        default=None,
        help="Sample size per month (default: use full dataset)"
    )
    parser.add_argument(
        "--num-months",
        type=int,
        default=12,
        help="Number of months to generate (default: 12)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(output_dir),
        help=f"Output directory (default: {output_dir})"
    )
    
    args = parser.parse_args()
    
    # Generate monthly datasets
    generated_paths = generate_monthly_datasets(
        input_path=input_path,
        output_dir=args.output_dir,
        num_months=args.num_months,
        sample_size=args.sample_size
    )
    
    # Print summary for easy reference in workflows
    print("\n" + "=" * 60)
    print("GENERATED FILES (for workflow reference):")
    print("-" * 60)
    for path in generated_paths:
        print(path)
    print("=" * 60)
    
    # Also write paths to a manifest file for programmatic access
    manifest_path = Path(args.output_dir) / "manifest.txt"
    with open(manifest_path, "w") as f:
        for path in generated_paths:
            # Convert to relative path from project root for cross-platform compatibility
            relative_path = Path(path).relative_to(project_root)
            # Use forward slashes for cross-platform compatibility
            f.write(f"{relative_path.as_posix()}\n")
    print(f"\nManifest written to: {manifest_path}")
    
    # Test drift detection on a sample month
    print("\n" + "=" * 60)
    print("DRIFT DETECTION TEST (Month 6 vs Original):")
    print("-" * 60)
    
    test_month = 6
    test_path = Path(args.output_dir) / f"month_{test_month:02d}.csv"
    
    if test_path.exists():
        # Run drift check
        import subprocess
        result = subprocess.run([
            sys.executable,
            str(script_dir / "drift_check.py"),
            str(test_path),
            "--ref-data", str(input_path)
        ], capture_output=True, text=True)
        
        try:
            import json
            drift_result = json.loads(result.stdout.strip())
            print(f"Mean PSI: {drift_result.get('mean_psi', 'N/A'):.4f}")
            print(f"Drift exceeded: {drift_result.get('drift_exceeded', 'N/A')}")
            print(f"Threshold: {drift_result.get('threshold', 'N/A')}")
        except:
            print("Could not parse drift check results")
            print(f"Output: {result.stdout}")
    
    print("=" * 60)
    print("\nDone! Monthly datasets are ready for testing drift detection and model monitoring.")

if __name__ == "__main__":
    main()