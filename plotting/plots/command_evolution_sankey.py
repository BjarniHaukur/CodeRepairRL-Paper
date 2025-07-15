#!/usr/bin/env python3
"""
Create proper time-evolution Sankey diagrams using the provided plotly template.
Generate two separate PNG files for early vs late training command evolution.

Uses the exact template structure with command states and actual transition data.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from wandb_utils import get_run, get_history, extract_shell_commands
from plot_config import get_command_color


# Configuration 
ENTITY = "assert-kth"
PROJECT = "SWE-Gym-GRPO"
RUN_ID = "nz1r7ml3"  # https://wandb.ai/assert-kth/SWE-Gym-GRPO/runs/nz1r7ml3
# RUN_ID = "c1mr1lgd"  # https://wandb.ai/assert-kth/SWE-Gym-GRPO/runs/c1mr1lgd

# Time steps for evolution
T = 25 # Number of time steps - good tradeoff point

# We'll build the states list dynamically based on actual command usage


def process_single_table(run, table_index, row_data):
    """Process a single table - used for parallel processing."""
    global_step = int(row_data['train/global_step'])
    table_info = row_data['table']
    table_path = table_info['path']
    
    try:
        # Download and parse table (use replace=False to cache)
        table_file = run.file(table_path)
        html_content = table_file.download(replace=False, exist_ok=True)
        
        with open(html_content.name, 'r', encoding='utf-8') as f:
            html_data = f.read()
        
        soup = BeautifulSoup(html_data, 'html.parser')
        table = soup.find('table')
        
        if not table:
            return None
        
        # Extract headers
        headers = []
        header_row = table.find('thead')
        if header_row:
            headers = [th.get_text(strip=True) for th in header_row.find_all('th')]
        else:
            first_row = table.find('tr')
            if first_row:
                headers = [td.get_text(strip=True) for td in first_row.find_all(['td', 'th'])]
        
        # Extract rows
        rows = []
        tbody = table.find('tbody')
        if tbody:
            for tr in tbody.find_all('tr'):
                row = [td.get_text(separator='\n', strip=True) for td in tr.find_all('td')]
                if row:
                    rows.append(row)
        
        if headers and rows:
            # Create DataFrame
            max_cols = len(headers)
            padded_rows = [row[:max_cols] + [''] * (max_cols - len(row)) for row in rows]
            
            df = pd.DataFrame(padded_rows, columns=headers)
            df['global_step'] = global_step
            df['table_index'] = table_index
            return df
            
    except Exception as e:
        return None


def extract_all_training_tables(run, max_tables=None):
    """Extract all tables from the training history using parallel processing."""
    print("="*60)
    print("Extracting All Training Tables (Parallel)")
    print("="*60)
    
    history = get_history(run, keys=['train/global_step', 'table'])
    table_history = history[~history['table'].isna()]
    
    print(f"Found {len(table_history)} table entries in training history")
    
    if max_tables:
        table_history = table_history.head(max_tables)
        print(f"Limited to {len(table_history)} tables for testing")
    
    # Prepare data for parallel processing
    table_data = [(i, row) for i, (_, row) in enumerate(table_history.iterrows())]
    
    tables = []
    failed_count = 0
    
    # Use ThreadPoolExecutor for parallel downloads
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(process_single_table, run, i, row): i 
            for i, row in table_data
        }
        
        # Process completed tasks with progress bar
        with tqdm(total=len(table_data), desc="Processing tables") as pbar:
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    if result is not None:
                        tables.append((index, result))
                    else:
                        failed_count += 1
                except Exception as e:
                    failed_count += 1
                pbar.update(1)
    
    # Sort tables by index to maintain order
    tables.sort(key=lambda x: x[0])
    tables = [df for _, df in tables]
    
    print(f"\nSuccessfully extracted {len(tables)} tables ({failed_count} failed)")
    total_rollouts = sum(len(df) for df in tables)
    print(f"Total rollouts: {total_rollouts}")
    
    return tables


def build_dynamic_transitions_data(tables, training_phase):
    """
    Build transitions data with dynamic symbol set that grows per timestep.
    
    Args:
        tables: List of DataFrames
        training_phase: "Early" or "Late" for logging
        
    Returns:
        Tuple of (states_by_timestep, transitions, all_states)
    """
    print(f"\nBuilding dynamic transitions for {training_phase} training:")
    
    # Extract all command sequences from rollouts (no length constraint)
    all_sequences = []
    total_rollouts = sum(len(df) for df in tables)
    
    for table_idx, df in enumerate(tables):
        for rollout_idx in range(len(df)):
            completion = df.iloc[rollout_idx]['Completion']
            commands = extract_shell_commands(completion, max_steps=20)
            
            # Include all sequences, pad shorter ones with None to show natural termination
            if len(commands) > 0:  # Only include sequences with at least one command
                padded_commands = commands[:T] + [None] * (T - len(commands))
                all_sequences.append(padded_commands[:T])  # Ensure exactly T elements
    
    print(f"  Total rollouts: {total_rollouts}")
    print(f"  Valid sequences: {len(all_sequences)} (all with ≥1 command)")
    
    # Build dynamic state set: track which states appear at each timestep
    states_by_timestep = [set() for _ in range(T)]
    
    for sequence in all_sequences:
        for t, cmd in enumerate(sequence):
            if cmd is not None:  # Skip None values when building state sets
                states_by_timestep[t].add(cmd)
    
    # Get all unique states across all timesteps (excluding None)
    all_states = set()
    for states_set in states_by_timestep:
        all_states.update(states_set)
    
    # Remove None if it somehow got in there
    all_states.discard(None)
    
    # Sort states for consistent ordering
    all_states = sorted(list(all_states))
    
    print(f"  Total unique commands found: {len(all_states)}")
    for t in range(T):
        valid_commands = len([cmd for cmd in states_by_timestep[t] if cmd is not None])
        print(f"    Timestep {t}: {valid_commands} unique commands")
    
    # Build state index mapping
    state_to_idx = {state: i for i, state in enumerate(all_states)}
    N = len(all_states)
    
    # Build transitions matrix
    transitions = np.zeros((T-1, N, N), dtype=int)
    
    for sequence in all_sequences:
        for t in range(T-1):
            from_cmd = sequence[t]
            to_cmd = sequence[t+1]
            
            # Only record transitions between actual commands (skip None)
            if from_cmd is not None and to_cmd is not None:
                from_idx = state_to_idx[from_cmd]
                to_idx = state_to_idx[to_cmd]
                transitions[t, from_idx, to_idx] += 1
    
    # Print statistics
    total_transitions = transitions.sum()
    print(f"  Total transitions recorded: {total_transitions}")
    
    for t in range(T-1):
        step_transitions = transitions[t].sum()
        print(f"    Step {t} → {t+1}: {step_transitions} transitions")
    
    return states_by_timestep, transitions, all_states


def create_sankey_diagram(states_by_timestep, transitions, all_states, title, filename):
    """
    Create Sankey diagram with dynamic states using the provided plotly template.
    
    Args:
        states_by_timestep: List of sets containing states active at each timestep
        transitions: (T-1, N, N) numpy array
        all_states: List of all unique states
        title: Plot title
        filename: Output filename (without extension)
    """
    print(f"\nCreating Sankey diagram: {title}")
    
    N = len(all_states)
    
    # ── 2. Flatten to Plotly's 1-D source/target/value lists ─────────────────
    labels, x, y, node_colors, node_sizes = [], [], [], [], []
    
    # Use semantic color mapping from plot_config
    state_color_map = {}
    for s in all_states:
        state_color_map[s] = get_command_color(s)
    
    # Track which states are actually used at each timestep to position nodes correctly
    node_mapping = {}  # Maps (timestep, state) to node index
    node_count = 0
    
    for t in range(T):
        # Only add nodes for states that actually appear at this timestep
        active_states = sorted([s for s in all_states if s in states_by_timestep[t]])
        
        for k, s in enumerate(active_states):
            # Special formatting for apply_patch
            if s == 'apply_patch':
                labels.append(f'<b>APPLY_PATCH</b>')  # Bold and uppercase
                node_sizes.append(28)  # Larger node
            else:
                labels.append(s)
                node_sizes.append(18)  # Regular size
                
            x.append(t / (T-1))            # fixed column positions: 0 … 1
            y.append(k / (len(active_states)-1) if len(active_states) > 1 else 0.5)  # spacing based on active states
            node_colors.append(state_color_map[s])
            node_mapping[(t, s)] = node_count
            node_count += 1

    sources, targets, values, link_colors = [], [], [], []
    
    # Build transitions based on actual flows in the data
    state_to_idx = {state: i for i, state in enumerate(all_states)}
    
    for t in range(T-1):
        for i, from_state in enumerate(all_states):
            for j, to_state in enumerate(all_states):
                v = transitions[t, i, j]
                if v == 0:  # prune empty flows
                    continue
                
                # Only add flow if both states exist at their respective timesteps
                if from_state in states_by_timestep[t] and to_state in states_by_timestep[t+1]:
                    if (t, from_state) in node_mapping and (t+1, to_state) in node_mapping:
                        sources.append(node_mapping[(t, from_state)])
                        targets.append(node_mapping[(t+1, to_state)])
                        values.append(int(v))
                        # Color links by their source state for easy tracing
                        # Add transparency to reduce visual clutter
                        base_color = state_color_map[from_state]
                        if base_color.startswith('#'):
                            # Convert hex to rgba with reduced opacity for cleaner look
                            r = int(base_color[1:3], 16)
                            g = int(base_color[3:5], 16)
                            b = int(base_color[5:7], 16)
                            link_colors.append(f'rgba({r}, {g}, {b}, 0.5)')  # 50% opacity for better clarity
                        else:
                            link_colors.append(base_color)

    print(f"Total flows in diagram: {len(values)}")
    if values:
        print(f"Flow range: {min(values)} - {max(values)}")

    # ── 3. Plot ───────────────────────────────────────────────────────────────
    # Create border styling (thicker for apply_patch)
    node_line_colors = []
    node_line_widths = []
    
    for label in labels:
        if 'APPLY_PATCH' in label:
            node_line_colors.append('#8B0000')  # Dark red border
            node_line_widths.append(2.5)        # Thick border
        else:
            node_line_colors.append('black')
            node_line_widths.append(0.5)        # Thin border
    
    fig = go.Figure(go.Sankey(
        arrangement = "snap",              # keeps nodes where we placed them
        node = dict(
            pad       = 25,  # More padding between nodes
            thickness = 20,  # Slightly thinner nodes for cleaner look
            label     = labels,
            x         = x,
            y         = y,
            color     = node_colors,
            line      = dict(color=node_line_colors, width=node_line_widths),
        ),
        link = dict(
            source = sources,
            target = targets,
            value  = values,
            color  = link_colors,
        ),
        textfont = dict(size=22, family="Arial", color="black")
    ))
    
    fig.update_layout(
        width  = 2200,  # Back to reasonable width for readable text
        height = 1400,  # Back to reasonable height
        font   = dict(size=24),
        margin = dict(l=150, r=150, t=120, b=300),  # Huge bottom margin to fix cutoff
        title  = dict(
            text=f"{title}<br><sub>Command Evolution Across {T} Time Steps in Rollouts</sub>",
            x=0.5,
            font=dict(size=32, family="Arial")
        ),
    )

    # ── 4. Export ─────────────────────────────────────────────────────────────
    html_path = f"../figures/plots/{filename}.html"
    
    fig.write_html(html_path)
    print(f"Saved HTML: {html_path}")
    
    return fig


def main():
    """Main function."""
    
    print("="*60)
    print("Proper Sankey Diagrams: Command Time Evolution")
    print("="*60)
    
    # Get run
    run = get_run(ENTITY, PROJECT, RUN_ID)
    print(f"Loaded run: {run.name} (ID: {run.id})")
    print(f"Tags: {run.tags}")
    print(f"State: {run.state}")
    
    # Extract all training tables
    tables = extract_all_training_tables(run)
    
    if not tables:
        print("❌ No tables extracted")
        return
    
    # Split into first 20% and last 20% for more disjoint comparison
    first_20_percent = int(len(tables) * 0.2)
    last_20_percent_start = int(len(tables) * 0.8)
    
    first_half_tables = tables[:first_20_percent]
    second_half_tables = tables[last_20_percent_start:]
    
    print(f"\nSplitting {len(tables)} tables:")
    print(f"First 20%: {len(first_half_tables)} tables ({sum(len(df) for df in first_half_tables)} rollouts)")
    print(f"Last 20%: {len(second_half_tables)} tables ({sum(len(df) for df in second_half_tables)} rollouts)")
    
    # Build dynamic transitions data for each half
    early_states, early_transitions, early_all_states = build_dynamic_transitions_data(first_half_tables, "Early")
    late_states, late_transitions, late_all_states = build_dynamic_transitions_data(second_half_tables, "Late")
    
    # Create separate Sankey diagrams
    print("\n" + "="*60)
    print("Creating Dynamic Time-Evolution Sankey Diagrams")
    print("="*60)
    
    # Early training Sankey
    early_fig = create_sankey_diagram(
        early_states,
        early_transitions,
        early_all_states,
        f"{run.name}: Early Training Command Evolution (First 20%)",
        f"early_training_sankey_{RUN_ID}"
    )
    
    # Late training Sankey  
    late_fig = create_sankey_diagram(
        late_states,
        late_transitions,
        late_all_states,
        f"{run.name}: Late Training Command Evolution (Last 20%)",
        f"late_training_sankey_{RUN_ID}"
    )
    
    print(f"\n✅ Created proper time-evolution Sankey diagrams!")
    print(f"📊 Two separate files with {T} time steps each")
    print(f"🎯 Shows how command sequences evolve within rollouts during training")


if __name__ == "__main__":
    main()