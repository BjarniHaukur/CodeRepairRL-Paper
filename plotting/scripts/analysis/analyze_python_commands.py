#!/usr/bin/env python3
"""
Analyze Python commands executed by the model throughout training.
Extract python/python3 commands and their outputs to understand what the model is trying to do.
"""

import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import json
from collections import defaultdict, Counter
from tqdm import tqdm

# Import from command_evolution_sankey
from command_evolution_sankey import (
    get_run, extract_all_training_tables
)

# Import from wandb_utils
from wandb_utils import extract_shell_commands
from plot_config import ENTITY, PROJECT, RUN_ID, get_output_filename

def extract_python_commands_with_context(completion_text):
    """
    Extract full Python commands and their context from completion text.
    Custom implementation to get full command with arguments.
    """
    python_executions = []
    
    # Use the same patterns as wandb_utils but extract full commands
    from wandb_utils import SHELL_PATTERN
    
    # Find all shell commands
    shell_matches = [(m.start(), m.group(1)) for m in SHELL_PATTERN.finditer(completion_text)]
    
    # Process each shell command
    for start_pos, full_command in shell_matches:
        full_command = full_command.strip()
        
        # Check if it's a Python command
        if full_command.startswith(('python', 'python3')):
            # Extract context/output by looking at text after the command
            # Find the end of the tool call
            tool_end = completion_text.find('</tool_call>', start_pos)
            if tool_end == -1:
                continue
                
            # Look for output after the tool call
            output_start = tool_end + len('</tool_call>')
            context_window = completion_text[output_start:output_start + 1000]
            
            # Extract meaningful output (skip empty lines and tool syntax)
            output_lines = []
            for line in context_window.split('\n'):
                line = line.strip()
                if line and not line.startswith('<') and not line.startswith('```') and not line.startswith('{{'):
                    output_lines.append(line)
                elif line.startswith('<') or line.startswith('```'):
                    break
                elif len(output_lines) > 0:  # Stop after some output is captured
                    break
            
            output = '\n'.join(output_lines[:15])  # First 15 lines
            
            # Get some context around the command
            context_start = max(0, start_pos - 100)
            context_end = min(len(completion_text), tool_end + 300)
            context = completion_text[context_start:context_end]
            
            python_executions.append({
                'command': full_command,
                'output': output,
                'full_context': context
            })
    
    return python_executions

def categorize_python_command(cmd):
    """Categorize Python commands by their apparent purpose."""
    cmd_lower = cmd.lower()
    
    # Check for common patterns
    if '-m pip' in cmd_lower or 'pip install' in cmd_lower or 'pip show' in cmd_lower:
        return 'pip_operations'
    elif '-m pytest' in cmd_lower or 'pytest' in cmd_lower:
        return 'testing'
    elif '-c' in cmd_lower:
        # Further categorize inline execution
        if 'import' in cmd_lower:
            return 'import_check'
        elif 'print' in cmd_lower:
            return 'inline_print'
        else:
            return 'inline_other'
    elif '.py' in cmd_lower and not any(x in cmd_lower for x in ['-m', '-c']):
        return 'script_execution'
    elif '-m' in cmd_lower and 'pip' not in cmd_lower and 'pytest' not in cmd_lower:
        return 'module_execution'
    elif '--version' in cmd_lower:
        return 'version_check'
    else:
        return 'other'

def extract_inline_code(cmd):
    """Extract code from python -c commands."""
    # Try different quote patterns
    patterns = [
        r'-c\s+"([^"]+)"',
        r"-c\s+'([^']+)'",
        r'-c\s+([^\s]+)'  # Unquoted single word
    ]
    
    for pattern in patterns:
        match = re.search(pattern, cmd)
        if match:
            return match.group(1)
    
    return None

def analyze_python_usage_from_tables(tables):
    """Analyze Python commands from parsed tables."""
    print("="*80)
    print("Python Command Analysis from Parsed Tables")
    print("="*80)
    
    all_python_data = []
    
    # Process each table
    for table_idx, df in enumerate(tqdm(tables, desc="Analyzing Python commands")):
        if 'Completion' not in df.columns:
            continue
            
        # Get global step if available
        global_step = df['global_step'].iloc[0] if 'global_step' in df.columns else table_idx
        
        # Extract Python commands from each rollout
        for rollout_idx, completion in enumerate(df['Completion']):
            if not completion:
                continue
                
            python_cmds = extract_python_commands_with_context(completion)
            
            for cmd_data in python_cmds:
                all_python_data.append({
                    'table_index': table_idx,
                    'global_step': global_step,
                    'rollout_index': rollout_idx,
                    'command': cmd_data['command'],
                    'output': cmd_data['output'],
                    'category': categorize_python_command(cmd_data['command']),
                    'inline_code': extract_inline_code(cmd_data['command']) if '-c' in cmd_data['command'] else None
                })
    
    print(f"\nTotal Python commands found: {len(all_python_data)}")
    
    if not all_python_data:
        print("No Python commands found!")
        return
    
    # Analyze by training phase
    total_tables = len(tables)
    early_cutoff = int(total_tables * 0.2)
    late_cutoff = int(total_tables * 0.8)
    
    early_data = [d for d in all_python_data if d['table_index'] < early_cutoff]
    late_data = [d for d in all_python_data if d['table_index'] >= late_cutoff]
    
    print(f"\n{'='*80}")
    print("TRAINING PHASE COMPARISON")
    print(f"{'='*80}")
    
    # Early training analysis
    print(f"\nEarly Training (First 20% of tables):")
    print(f"  Total Python commands: {len(early_data)}")
    if early_data:
        early_categories = Counter(d['category'] for d in early_data)
        print(f"  Command categories:")
        for cat, count in early_categories.most_common():
            percentage = (count / len(early_data)) * 100
            print(f"    {cat}: {count} ({percentage:.1f}%)")
    
    # Late training analysis
    print(f"\nLate Training (Last 20% of tables):")
    print(f"  Total Python commands: {len(late_data)}")
    if late_data:
        late_categories = Counter(d['category'] for d in late_data)
        print(f"  Command categories:")
        for cat, count in late_categories.most_common():
            percentage = (count / len(late_data)) * 100
            print(f"    {cat}: {count} ({percentage:.1f}%)")
    
    # Most common commands
    print(f"\n{'='*80}")
    print("TOP PYTHON COMMANDS")
    print(f"{'='*80}")
    
    command_counts = Counter(d['command'] for d in all_python_data)
    print("\nTop 20 most frequent Python commands:")
    for i, (cmd, count) in enumerate(command_counts.most_common(20), 1):
        print(f"  {i:2d}. [{count:3d}] {cmd[:100]}{'...' if len(cmd) > 100 else ''}")
    
    # Analyze inline code patterns
    print(f"\n{'='*80}")
    print("INLINE CODE PATTERNS (-c commands)")
    print(f"{'='*80}")
    
    inline_data = [d for d in all_python_data if d['category'].startswith('inline_')]
    if inline_data:
        inline_patterns = Counter(d['inline_code'] for d in inline_data if d['inline_code'])
        print(f"\nTop 15 inline code patterns:")
        for i, (code, count) in enumerate(inline_patterns.most_common(15), 1):
            if code:
                print(f"  {i:2d}. [{count:3d}] {code[:80]}{'...' if len(code) > 80 else ''}")
    
    # Show specific examples with outputs
    print(f"\n{'='*80}")
    print("EXAMPLE COMMANDS WITH OUTPUTS")
    print(f"{'='*80}")
    
    # Show some actual examples to understand what's happening
    print("\n--- SAMPLE PYTHON COMMANDS ---")
    for i, data in enumerate(all_python_data[:10]):  # First 10 examples
        print(f"\nExample {i+1}:")
        print(f"  Command: '{data['command']}'")
        print(f"  Category: {data['category']}")
        if data['output']:
            output = data['output'].strip()
            if len(output) > 150:
                output = output[:150] + "..."
            print(f"  Output: {output}")
        else:
            print("  Output: [No output captured]")
        
        # Show context to understand what's happening
        if data.get('full_context'):
            context = data['full_context'][:200] + "..." if len(data['full_context']) > 200 else data['full_context']
            print(f"  Context: {context}")
        print("-" * 60)
    
    # Evolution over training
    print(f"\n{'='*80}")
    print("COMMAND EVOLUTION OVER TRAINING")
    print(f"{'='*80}")
    
    # Divide into quarters
    quarter_size = total_tables // 4
    quarters = [
        ("Q1 (0-25%)", 0, quarter_size),
        ("Q2 (25-50%)", quarter_size, 2*quarter_size),
        ("Q3 (50-75%)", 2*quarter_size, 3*quarter_size),
        ("Q4 (75-100%)", 3*quarter_size, total_tables)
    ]
    
    for quarter_name, start, end in quarters:
        quarter_data = [d for d in all_python_data if start <= d['table_index'] < end]
        if quarter_data:
            print(f"\n{quarter_name}:")
            print(f"  Total commands: {len(quarter_data)}")
            print(f"  Commands per table: {len(quarter_data) / (end - start):.2f}")
            
            # Top categories
            quarter_cats = Counter(d['category'] for d in quarter_data)
            print(f"  Top categories:")
            for cat, count in quarter_cats.most_common(3):
                percentage = (count / len(quarter_data)) * 100
                print(f"    {cat}: {percentage:.1f}%")
    
    # Special analysis for pip operations
    print(f"\n{'='*80}")
    print("PIP OPERATIONS ANALYSIS")
    print(f"{'='*80}")
    
    pip_data = [d for d in all_python_data if d['category'] == 'pip_operations']
    if pip_data:
        pip_commands = Counter(d['command'] for d in pip_data)
        print(f"\nTotal pip operations: {len(pip_data)}")
        print("Most common pip commands:")
        for cmd, count in pip_commands.most_common(10):
            print(f"  [{count:3d}] {cmd}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Analyze Python commands executed by the model throughout training')
    parser.add_argument('--run-id', type=str, default=RUN_ID, 
                        help=f'WandB run ID (default: {RUN_ID})')
    args = parser.parse_args()
    
    # Get run
    run = get_run(ENTITY, PROJECT, args.run_id)
    print(f"Analyzing run: {run.name} (ID: {run.id})")
    
    # Extract all training tables using the existing function
    tables = extract_all_training_tables(run)
    
    if not tables:
        print("No tables found!")
        return
    
    print(f"\nLoaded {len(tables)} tables")
    
    # Analyze Python usage
    analyze_python_usage_from_tables(tables)

if __name__ == "__main__":
    main()