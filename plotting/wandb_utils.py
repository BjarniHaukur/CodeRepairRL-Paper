"""
Simple utility functions for fetching data from Weights & Biases API.
"""

import wandb
import pandas as pd
import numpy as np
import requests
import json
import re
from bs4 import BeautifulSoup
from typing import List, Dict, Optional, Tuple
from collections import defaultdict, Counter


def get_run(entity: str, project: str, run_id: str) -> wandb.apis.public.Run:
    """
    Get a specific run from W&B.
    
    Args:
        entity: W&B entity (username or team)
        project: W&B project name
        run_id: Run ID
    
    Returns:
        wandb Run object
    """
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    print(f"Loaded run: {run.name} (ID: {run.id})")
    print(f"Tags: {run.tags}")
    print(f"State: {run.state}")
    return run


def get_history(run: wandb.apis.public.Run, 
                keys: Optional[List[str]] = None,
                x_axis: str = "_step") -> pd.DataFrame:
    """
    Get history (metrics over time) from a run.
    
    Args:
        run: W&B run object
        keys: List of metric keys to fetch (None = all)
        x_axis: X-axis key (default: "_step")
    
    Returns:
        DataFrame with run history
    """
    if keys and x_axis not in keys:
        keys = keys + [x_axis]
    
    history = run.history(keys=keys)
    
    print(f"\nHistory shape: {history.shape}")
    print(f"Columns: {list(history.columns)}")
    
    # Print summary stats for numeric columns
    numeric_cols = history.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != x_axis and not history[col].isna().all():
            print(f"{col}: mean={history[col].mean():.4f}, "
                  f"std={history[col].std():.4f}, "
                  f"last={history[col].iloc[-1]:.4f}")
    
    return history


def get_summary_metrics(run: wandb.apis.public.Run) -> Dict:
    """
    Get final/summary metrics from a run.
    
    Args:
        run: W&B run object
    
    Returns:
        Dict of summary metrics
    """
    summary = dict(run.summary)
    
    print(f"\nSummary metrics:")
    for key, value in summary.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    return summary


def get_all_runs(entity: str, project: str) -> List[wandb.apis.public.Run]:
    """
    Get all runs from a project for analysis.
    
    Args:
        entity: W&B entity
        project: W&B project name
    
    Returns:
        List of all runs in the project
    """
    api = wandb.Api()
    runs = list(api.runs(f"{entity}/{project}"))
    
    print(f"Found {len(runs)} total runs in {entity}/{project}")
    return runs


def analyze_runs(runs: List[wandb.apis.public.Run], min_steps: int = 100) -> Dict:
    """
    Analyze runs to identify which are worth plotting.
    
    Args:
        runs: List of wandb runs
        min_steps: Minimum steps to consider a run viable
    
    Returns:
        Dict with analysis results
    """
    analysis = {
        "viable_runs": [],
        "short_runs": [],
        "crashed_runs": [],
        "summary": {}
    }
    
    print(f"\nAnalyzing runs (min_steps={min_steps}):")
    print("=" * 60)
    
    for run in runs:
        # Get basic info
        steps = run.summary.get("_step", 0)
        global_steps = run.summary.get("train/global_step", 0)
        state = run.state
        
        # Check if viable
        actual_steps = max(steps, global_steps)
        
        print(f"{run.name[:30]:30} | Steps: {actual_steps:4.0f} | State: {state:10} | ID: {run.id}")
        
        if actual_steps >= min_steps:
            analysis["viable_runs"].append({
                "run": run,
                "steps": actual_steps,
                "url": f"https://wandb.ai/{run.entity}/{run.project}/runs/{run.id}"
            })
        else:
            analysis["short_runs"].append({
                "run": run,
                "steps": actual_steps,
                "url": f"https://wandb.ai/{run.entity}/{run.project}/runs/{run.id}"
            })
        
        if state in ["crashed", "failed"]:
            analysis["crashed_runs"].append({
                "run": run,
                "steps": actual_steps,
                "url": f"https://wandb.ai/{run.entity}/{run.project}/runs/{run.id}"
            })
    
    # Summary
    analysis["summary"] = {
        "total_runs": len(runs),
        "viable_runs": len(analysis["viable_runs"]),
        "short_runs": len(analysis["short_runs"]),
        "crashed_runs": len(analysis["crashed_runs"])
    }
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"  Total runs: {analysis['summary']['total_runs']}")
    print(f"  Viable runs (>{min_steps} steps): {analysis['summary']['viable_runs']}")
    print(f"  Short runs (<={min_steps} steps): {analysis['summary']['short_runs']}")
    print(f"  Crashed/failed runs: {analysis['summary']['crashed_runs']}")
    
    return analysis


def print_data_info(df: pd.DataFrame, name: str = "Data"):
    """
    Print useful information about a DataFrame for debugging.
    
    Args:
        df: DataFrame to inspect
        name: Name for the printout
    """
    print(f"\n{name} Info:")
    print(f"  Shape: {df.shape}")
    print(f"  Memory: {df.memory_usage().sum() / 1024**2:.2f} MB")
    print(f"  Columns: {list(df.columns)}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    print(f"\nLast 5 rows:")
    print(df.tail())


def get_table_data(run: wandb.apis.public.Run) -> Optional[pd.DataFrame]:
    """
    Extract and parse the HTML table from a W&B run containing rollout data.
    
    Args:
        run: W&B run object
    
    Returns:
        DataFrame with parsed rollout data, or None if no table found
    """
    # Check if run has table data
    if "table" not in run.summary:
        print(f"No table found in run {run.name}")
        return None
    
    table_info = run.summary["table"]
    # Convert to regular dict to handle W&B's SummarySubDict
    table_dict = dict(table_info)
    
    if "path" not in table_dict:
        print(f"No path in table info for run {run.name}: {table_dict}")
        return None
    
    # Get the table file path
    table_path = table_dict["path"]
    print(f"Found table at: {table_path}")
    
    try:
        # Download the HTML table file from W&B
        table_file = run.file(table_path)
        html_content = table_file.download(replace=True, exist_ok=True)
        
        # Read the HTML content
        with open(html_content.name, 'r', encoding='utf-8') as f:
            html_data = f.read()
        
        # Parse with Beautiful Soup
        soup = BeautifulSoup(html_data, 'html.parser')
        
        # Find the table
        table = soup.find('table')
        if not table:
            print("No table element found in HTML")
            return None
        
        # Extract headers
        headers = []
        header_row = table.find('thead')
        if header_row:
            for th in header_row.find_all('th'):
                headers.append(th.get_text(strip=True))
        else:
            # Fallback: use first row as headers
            first_row = table.find('tr')
            if first_row:
                for td in first_row.find_all(['td', 'th']):
                    headers.append(td.get_text(strip=True))
        
        print(f"Table headers: {headers}")
        
        # Extract rows
        rows = []
        tbody = table.find('tbody')
        if tbody:
            for tr in tbody.find_all('tr'):
                row = []
                for td in tr.find_all('td'):
                    # Get text content, preserving some formatting
                    cell_text = td.get_text(separator='\n', strip=True)
                    row.append(cell_text)
                if row:  # Only add non-empty rows
                    rows.append(row)
        else:
            # Fallback: all rows except first (if used as header)
            all_rows = table.find_all('tr')
            start_idx = 1 if not table.find('thead') else 0
            for tr in all_rows[start_idx:]:
                row = []
                for td in tr.find_all('td'):
                    cell_text = td.get_text(separator='\n', strip=True)
                    row.append(cell_text)
                if row:
                    rows.append(row)
        
        print(f"Extracted {len(rows)} rows")
        
        # Create DataFrame
        if headers and rows:
            # Ensure all rows have same length as headers
            max_cols = len(headers)
            padded_rows = []
            for row in rows:
                padded_row = row[:max_cols]  # Truncate if too long
                padded_row.extend([''] * (max_cols - len(padded_row)))  # Pad if too short
                padded_rows.append(padded_row)
            
            df = pd.DataFrame(padded_rows, columns=headers)
            print(f"Created DataFrame with shape: {df.shape}")
            return df
        else:
            print("No valid data found in table")
            return None
            
    except Exception as e:
        print(f"Error parsing table: {e}")
        return None


def parse_rollout_data(df: pd.DataFrame, rollout_index: int = 0) -> Dict:
    """
    Parse a specific rollout from the table data.
    
    Args:
        df: DataFrame from get_table_data()
        rollout_index: Which rollout to extract (0-based)
    
    Returns:
        Dict with parsed rollout data
    """
    if df is None or len(df) == 0:
        return {}
    
    if rollout_index >= len(df):
        print(f"Rollout index {rollout_index} out of range (max: {len(df)-1})")
        return {}
    
    row = df.iloc[rollout_index]
    
    # Try to identify key columns
    rollout_data = {}
    
    for col in df.columns:
        value = row[col]
        col_lower = col.lower()
        
        if any(keyword in col_lower for keyword in ['prompt', 'input', 'query']):
            rollout_data['prompt'] = value
        elif any(keyword in col_lower for keyword in ['completion', 'response', 'output']):
            rollout_data['completion'] = value
        elif any(keyword in col_lower for keyword in ['reward', 'score']):
            rollout_data['reward'] = value
        elif any(keyword in col_lower for keyword in ['episode', 'step', 'id']):
            rollout_data['episode_id'] = value
        else:
            # Store other columns as-is
            rollout_data[col] = value
    
    return rollout_data


# Pre-compile regex patterns for better performance
SHELL_PATTERN = re.compile(r'\{"name"\s*:\s*"shell"\s*,\s*"arguments"\s*:\s*\{"cmd"\s*:\s*"([^"]+)"\}\}')
APPLY_PATCH_PATTERN = re.compile(r'\{"name"\s*:\s*"apply_patch"\s*,\s*"arguments"\s*:\s*\{[^}]*\}\}')


def extract_shell_commands(completion_text: str, max_steps: int = 20) -> List[str]:
    """
    Extract shell commands and apply_patch tool calls from conversation completion text.
    
    Args:
        completion_text: Full completion text with conversation
        max_steps: Maximum number of command steps to extract
        
    Returns:
        List of commands/tools in order (shell commands return first word, apply_patch returns "apply_patch")
    """
    # Valid command whitelist - only common shell commands
    VALID_COMMANDS = {
        # File operations
        'ls', 'cat', 'head', 'tail', 'find', 'grep', 'rg', 'mkdir', 'rm', 'mv', 'cp', 'touch', 'chmod', 'pwd', 'cd',
        # Development tools
        'git', 'python', 'python3', 'pip', 'npm', 'node', 'curl', 'wget', 'make', 'cmake', 'gcc', 'g++',
        # Text processing
        'sed', 'awk', 'diff', 'sort', 'uniq', 'wc', 'tr', 'cut', 'echo', 'printf',
        # System tools
        'ps', 'top', 'htop', 'kill', 'killall', 'which', 'whereis', 'lsof', 'mount', 'umount', 'df', 'du', 'uname', 'whoami', 'id', 'uptime',
        # Package managers
        'apt', 'apt-get', 'yum', 'dnf', 'brew', 'conda', 'mamba',
        # Archive tools
        'tar', 'zip', 'unzip', 'gzip', 'gunzip',
        # Network tools
        'ssh', 'scp', 'rsync', 'ping', 'netstat',
        # Container tools
        'docker', 'podman', 'kubectl',
        # Editors
        'vim', 'nano', 'emacs',
        # Other common tools
        'bash', 'sh', 'zsh', 'fish', 'tmux', 'screen', 'sudo', 'su', 'env', 'export', 'source',
        'sqlite3', 'mysql', 'psql', 'redis-cli', 'mongo',
        # Testing/build tools
        'pytest', 'jest', 'mocha', 'cargo', 'go', 'rustc', 'javac', 'java',
        # Data tools
        'jq', 'yq', 'csvkit', 'sqlfluff', 'dask', 'dvc', 'mlflow'
    }
    
    commands = []
    
    # Find all tool calls (both shell and apply_patch)
    # First, find all positions of each type
    shell_matches = [(m.start(), 'shell', m.group(1)) for m in SHELL_PATTERN.finditer(completion_text)]
    patch_matches = [(m.start(), 'apply_patch', None) for m in APPLY_PATCH_PATTERN.finditer(completion_text)]
    
    # Combine and sort by position to maintain order
    all_matches = shell_matches + patch_matches
    all_matches.sort(key=lambda x: x[0])
    
    # Process matches in order
    for _, tool_type, match_data in all_matches[:max_steps]:
        if tool_type == 'apply_patch':
            commands.append('apply_patch')
        elif tool_type == 'shell' and match_data:
            # Extract the full command
            full_command = match_data.strip()
            
            # Extract just the first word (the actual command)
            cmd_parts = full_command.split()
            if cmd_parts:
                command = cmd_parts[0]
                
                # Clean common variations
                if command.startswith('./'):
                    command = command[2:]
                if command.endswith(';'):
                    command = command[:-1]
                
                # Remove any path prefixes
                if '/' in command:
                    command = command.split('/')[-1]
                
                # Only include if it's a valid command
                if command in VALID_COMMANDS:
                    commands.append(command)
    
    return commands