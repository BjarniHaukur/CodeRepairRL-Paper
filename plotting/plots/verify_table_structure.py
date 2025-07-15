#!/usr/bin/env python3
"""
Verify the table structure and parsing for run c1mr1lgd.
Should have 32 rows per table.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wandb_utils import get_run, get_table_data, parse_rollout_data
from bs4 import BeautifulSoup


# Configuration 
ENTITY = "assert-kth"
PROJECT = "SWE-Gym-GRPO"
RUN_ID = "c1mr1lgd"  # https://wandb.ai/assert-kth/SWE-Gym-GRPO/runs/c1mr1lgd


def debug_table_parsing():
    """Debug the table parsing process step by step."""
    
    print("="*60)
    print("Debugging Table Structure for c1mr1lgd")
    print("="*60)
    
    # Get run
    run = get_run(ENTITY, PROJECT, RUN_ID)
    
    # Check table info
    if "table" not in run.summary:
        print("❌ No table found in run summary")
        return
    
    table_info = dict(run.summary["table"])
    print(f"Table info: {table_info}")
    
    # Download and examine raw HTML
    table_path = table_info["path"]
    print(f"Table path: {table_path}")
    
    try:
        # Download the HTML file
        table_file = run.file(table_path)
        html_content = table_file.download(replace=True, exist_ok=True)
        
        print(f"Downloaded HTML file: {html_content.name}")
        
        # Read raw HTML
        with open(html_content.name, 'r', encoding='utf-8') as f:
            html_data = f.read()
        
        print(f"HTML file size: {len(html_data)} characters")
        
        # Parse with Beautiful Soup
        soup = BeautifulSoup(html_data, 'html.parser')
        
        # Find all tables
        tables = soup.find_all('table')
        print(f"Number of tables found: {len(tables)}")
        
        for i, table in enumerate(tables):
            print(f"\n--- Table {i+1} ---")
            
            # Count rows
            rows = table.find_all('tr')
            print(f"Total rows (including header): {len(rows)}")
            
            # Check for header
            thead = table.find('thead')
            tbody = table.find('tbody')
            
            if thead:
                header_rows = thead.find_all('tr')
                print(f"Header rows: {len(header_rows)}")
            
            if tbody:
                body_rows = tbody.find_all('tr')
                print(f"Body rows: {len(body_rows)}")
            else:
                print("No tbody found - counting all rows")
                # If no tbody, assume first row is header
                data_rows = rows[1:] if len(rows) > 1 else rows
                print(f"Data rows (excluding first): {len(data_rows)}")
            
            # Show first few row structures
            print(f"First 3 rows structure:")
            for j, row in enumerate(rows[:3]):
                cells = row.find_all(['td', 'th'])
                print(f"  Row {j}: {len(cells)} cells")
                if cells:
                    first_cell = cells[0].get_text(strip=True)[:50]
                    print(f"    First cell: {first_cell}...")
        
    except Exception as e:
        print(f"Error: {e}")


def test_improved_parsing():
    """Test the current parsing and see what we get."""
    
    print("\n" + "="*60)
    print("Testing Current Parsing")
    print("="*60)
    
    # Use existing function
    run = get_run(ENTITY, PROJECT, RUN_ID)
    df = get_table_data(run)
    
    if df is not None:
        print(f"✓ Parsed DataFrame shape: {df.shape}")
        print(f"✓ Columns: {list(df.columns)}")
        
        # Check if we have expected 32 rows
        expected_rows = 32
        actual_rows = len(df)
        
        if actual_rows == expected_rows:
            print(f"✅ Correct row count: {actual_rows}")
        else:
            print(f"❌ Expected {expected_rows} rows, got {actual_rows}")
        
        # Show sample data
        print(f"\nFirst row data lengths:")
        for col in df.columns:
            value = df.iloc[0][col]
            length = len(str(value)) if value else 0
            print(f"  {col}: {length} chars")
        
        return df
    else:
        print("❌ Failed to parse table")
        return None


def main():
    """Main verification function."""
    
    # Debug the parsing process
    debug_table_parsing()
    
    # Test current parsing
    df = test_improved_parsing()
    
    if df is not None and len(df) != 32:
        print(f"\n🔧 Need to fix parsing - expected 32 rows, got {len(df)}")
    elif df is not None:
        print(f"\n✅ Table parsing looks correct!")
    else:
        print(f"\n❌ Table parsing failed completely")


if __name__ == "__main__":
    main()