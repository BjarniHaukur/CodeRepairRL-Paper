#!/usr/bin/env python3
"""
Check the number of rows per table in the original RUN_ID to verify it's 64.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wandb
import tempfile
from bs4 import BeautifulSoup
from tqdm import tqdm
from plot_config import ENTITY, PROJECT, RUN_ID


def check_table_rows(run_id, sample_size=10):
    """
    Check the number of rows per table in HTML files.
    """
    print(f"Checking table rows for run: {run_id}")
    
    api = wandb.Api()
    run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
    
    # Get HTML files
    files = run.files()
    html_files = [f for f in files if f.name.endswith('.html') and 'table' in f.name]
    
    print(f"Found {len(html_files)} HTML files")
    
    # Check sample of files
    sample_files = html_files[:sample_size]
    print(f"Analyzing {len(sample_files)} sample files for row counts...")
    
    row_counts = []
    
    with tempfile.TemporaryDirectory() as temp_dir:
        for file in tqdm(sample_files, desc="Processing files"):
            try:
                # Download file
                file.download(root=temp_dir, replace=True)
                
                # Find downloaded file
                downloaded_path = None
                for root, dirs, files_in_dir in os.walk(temp_dir):
                    for fname in files_in_dir:
                        if fname.endswith('.html') and 'table' in fname:
                            downloaded_path = os.path.join(root, fname)
                            break
                    if downloaded_path:
                        break
                
                if not downloaded_path:
                    continue
                
                # Parse HTML
                with open(downloaded_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                soup = BeautifulSoup(content, 'html.parser')
                table = soup.find('table')
                
                if not table:
                    continue
                
                # Count rows
                rows = table.find_all('tr')
                total_rows = len(rows)
                data_rows = total_rows - 1  # Subtract header row
                
                row_counts.append({
                    'file': file.name,
                    'total_rows': total_rows,
                    'data_rows': data_rows
                })
                
                print(f"  {file.name}: {data_rows} data rows ({total_rows} total)")
                
            except Exception as e:
                print(f"  Error processing {file.name}: {e}")
                continue
    
    return row_counts


def main():
    """Main function."""
    print("="*80)
    print(f"CHECKING TABLE ROWS FOR RUN_ID: {RUN_ID}")
    print("="*80)
    
    row_counts = check_table_rows(RUN_ID)
    
    if not row_counts:
        print("No table data found!")
        return
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    data_rows = [rc['data_rows'] for rc in row_counts]
    
    print(f"\nFiles analyzed: {len(row_counts)}")
    print(f"Row counts (data rows only):")
    print(f"  Min: {min(data_rows)}")
    print(f"  Max: {max(data_rows)}")
    print(f"  Average: {sum(data_rows) / len(data_rows):.1f}")
    print(f"  Most common: {max(set(data_rows), key=data_rows.count)}")
    
    # Check if 64 is the expected value
    rows_64 = sum(1 for rc in data_rows if rc == 64)
    print(f"\nTables with exactly 64 rows: {rows_64}/{len(data_rows)} ({rows_64/len(data_rows)*100:.1f}%)")
    
    if rows_64 == len(data_rows):
        print("✅ ALL tables have exactly 64 rows as expected!")
    elif rows_64 > len(data_rows) * 0.8:
        print("⚠️  Most tables have 64 rows, but some variation exists")
    else:
        print("❌ Tables do not consistently have 64 rows")
    
    # Show distribution
    from collections import Counter
    count_distribution = Counter(data_rows)
    print(f"\nRow count distribution:")
    for count, freq in sorted(count_distribution.items()):
        print(f"  {count} rows: {freq} files")


if __name__ == "__main__":
    main()