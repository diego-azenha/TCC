"""Script to remove problematic columns from parquet files."""

import pandas as pd
from pathlib import Path

def main():
    print("="*80)
    print("Fixing Parquet Files - Removing SW002766 column")
    print("="*80)
    
    parquet_dir = Path("data/parquets")
    x_ts_path = parquet_dir / "x_ts.parquet"
    
    if not x_ts_path.exists():
        print(f"❌ File not found: {x_ts_path}")
        return
    
    # Load parquet
    print(f"\nLoading {x_ts_path}...")
    df = pd.read_parquet(x_ts_path)
    
    print(f"Original shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Check if SW002766 exists
    if 'SW002766' not in df.columns:
        print("\n✓ Column SW002766 not found in dataframe. Nothing to fix.")
        return
    
    # Check NaN count
    nan_count = df['SW002766'].isna().sum()
    total_rows = len(df)
    print(f"\nColumn SW002766 has {nan_count}/{total_rows} NaN values ({100*nan_count/total_rows:.2f}%)")
    
    # Remove column
    print("\nRemoving column SW002766...")
    df_fixed = df.drop(columns=['SW002766'])
    
    print(f"New shape: {df_fixed.shape}")
    print(f"New columns: {df_fixed.columns.tolist()}")
    
    # Backup original
    backup_path = parquet_dir / "x_ts.parquet.backup"
    if not backup_path.exists():
        print(f"\nCreating backup: {backup_path}")
        df.to_parquet(backup_path, index=False)
    
    # Save fixed version
    print(f"\nSaving fixed parquet to {x_ts_path}...")
    df_fixed.to_parquet(x_ts_path, index=False)
    
    # Verify
    print("\nVerifying fix...")
    df_verify = pd.read_parquet(x_ts_path)
    print(f"Verified shape: {df_verify.shape}")
    print(f"SW002766 in columns: {'SW002766' in df_verify.columns}")
    
    print("\n✓ Done! Parquet file fixed successfully.")
    print(f"  - Original file backed up to: {backup_path}")
    print(f"  - Fixed file saved to: {x_ts_path}")
    print(f"  - Removed columns: SW002766")
    print(f"  - New feature count: {len([c for c in df_fixed.columns if c not in ['date', 'ticker']])}")


if __name__ == "__main__":
    main()
