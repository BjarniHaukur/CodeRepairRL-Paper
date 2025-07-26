#!/usr/bin/env python3
"""Test HTML parsing to debug the issue"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from individual_problem_analysis import parse_table_html

# Test with a local HTML file
test_file = "../media/html/table_1004_f7148f6f8e823f6922dd.html"
print(f"Testing parse_table_html with: {test_file}")

rows = parse_table_html(test_file)
print(f"\nNumber of rows parsed: {len(rows)}")

if rows:
    print(f"\nFirst row keys: {list(rows[0].keys())}")
    print(f"\nFirst row (truncated):")
    for key, value in rows[0].items():
        print(f"  {key}: {str(value)[:100]}...")
else:
    print("\nNo rows parsed! Checking HTML structure...")
    
    # Read and inspect the HTML
    with open(test_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    print(f"\nFile size: {len(content)} bytes")
    print(f"Contains <table>: {'<table' in content}")
    print(f"Contains <thead>: {'<thead' in content}")
    print(f"Contains <tbody>: {'<tbody' in content}")
    print(f"Contains <tr>: {'<tr' in content}")
    print(f"Contains <td>: {'<td' in content}")