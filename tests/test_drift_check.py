# tests/test_drift_check.py
import sys
import json
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).resolve().parents[1]))

from mlops.drift_check import calculate_psi, select_drift_columns

def test_calculate_psi():
    """Test PSI calculation with known distributions"""
    # Test identical distributions (should have PSI ≈ 0)
    expected = pd.Series(np.random.normal(0, 1, 1000))
    actual = expected.copy()
    psi = calculate_psi(expected, actual)
    assert abs(psi) < 0.01, f"PSI for identical distributions should be ~0, got {psi}"
    
    # Test shifted distributions (should have higher PSI)
    shifted = expected + 2  # Shift mean by 2
    psi_shifted = calculate_psi(expected, shifted)
    assert psi_shifted > 0.1, f"PSI for shifted distribution should be >0.1, got {psi_shifted}"
    
    print(f"✓ PSI tests passed: identical={psi:.4f}, shifted={psi_shifted:.4f}")

def test_select_drift_columns():
    """Test column selection for drift detection"""
    df = pd.DataFrame({
        'Time': [1, 2, 3],
        'Amount': [10.0, 20.0, 30.0],
        'V1': [0.1, 0.2, 0.3],
        'Class': [0, 1, 0],  # Should be excluded
        'text_col': ['a', 'b', 'c']  # Should be excluded
    })
    
    drift_cols = select_drift_columns(df)
    expected_cols = ['Time', 'Amount', 'V1']
    assert set(drift_cols) == set(expected_cols), f"Expected {expected_cols}, got {drift_cols}"
    print(f"✓ Column selection test passed: {drift_cols}")

def create_test_datasets():
    """Create reference and new datasets for testing"""
    np.random.seed(42)
    
    # Reference dataset (similar to fraud detection features)
    n_ref = 1000
    ref_data = {
        'Time': np.random.uniform(0, 172800, n_ref),  # 48 hours in seconds
        'Amount': np.random.lognormal(3, 1.5, n_ref),  # Log-normal for realistic amounts
        'V1': np.random.normal(0, 1, n_ref),
        'V2': np.random.normal(0, 1, n_ref),
        'V3': np.random.normal(0, 1, n_ref),
        'Class': np.random.choice([0, 1], n_ref, p=[0.998, 0.002])  # Imbalanced target
    }
    ref_df = pd.DataFrame(ref_data)
    
    # New dataset with some drift in Amount and V1
    n_new = 500
    new_data = {
        'Time': np.random.uniform(0, 172800, n_new),
        'Amount': np.random.lognormal(3.2, 1.6, n_new),  # Slightly higher mean and variance
        'V1': np.random.normal(0.3, 1.1, n_new),  # Shifted mean and increased variance
        'V2': np.random.normal(0, 1, n_new),  # No drift
        'V3': np.random.normal(0, 1, n_new),  # No drift
        'Class': np.random.choice([0, 1], n_new, p=[0.995, 0.005])  # Different class distribution
    }
    new_df = pd.DataFrame(new_data)
    
    return ref_df, new_df

def test_end_to_end():
    """Test the complete drift check workflow"""
    print("\n=== End-to-End Drift Check Test ===")
    
    # Create test datasets
    ref_df, new_df = create_test_datasets()
    
    # Save to temporary files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as ref_file:
        ref_df.to_csv(ref_file.name, index=False)
        ref_path = ref_file.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as new_file:
        new_df.to_csv(new_file.name, index=False)
        new_path = new_file.name
    
    try:
        # Import and run drift check
        from mlops.drift_check import main
        import subprocess
        
        # Run the script as a subprocess to test the CLI interface
        result = subprocess.run([
            sys.executable, '-c', f'''
import sys
sys.path.append("{str(Path(__file__).resolve().parents[1])}")
from mlops.drift_check import main
import sys
sys.argv = ["drift_check.py", "{new_path}", "--ref-data", "{ref_path}"]
main()
            '''
        ], capture_output=True, text=True)
        
        # Parse JSON output
        try:
            drift_result = json.loads(result.stdout.strip())
            print(f"Drift check result: {json.dumps(drift_result, indent=2)}")
            
            # Verify expected structure
            assert drift_result['status'] == 'success'
            assert 'mean_psi' in drift_result
            assert 'drift_exceeded' in drift_result
            assert 'psi_by_column' in drift_result
            assert len(drift_result['psi_by_column']) > 0
            
            print(f"✓ End-to-end test passed!")
            print(f"  Mean PSI: {drift_result['mean_psi']}")
            print(f"  Drift exceeded: {drift_result['drift_exceeded']}")
            print(f"  Columns checked: {drift_result['columns_checked']}")
            
        except json.JSONDecodeError:
            print(f"Failed to parse JSON output: {result.stdout}")
            print(f"Error output: {result.stderr}")
            raise
            
    finally:
        # Clean up temporary files
        Path(ref_path).unlink()
        Path(new_path).unlink()

if __name__ == "__main__":
    print("Testing drift check functionality...")
    
    test_calculate_psi()
    test_select_drift_columns()
    test_end_to_end()
    
    print("\n✅ All tests passed!")
    print("\nUsage examples:")
    print("  python mlops/drift_check.py new_data.csv")
    print("  python mlops/drift_check.py new_data.csv --ref-data reference.csv")
    print("  REF_DATA=ref.csv DRIFT_PSI_THRESHOLD=0.1 python mlops/drift_check.py new.csv")