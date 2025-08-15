#!/usr/bin/env python3
"""
Analyze terminal_exploration_depth_reward by examining specific completions and their rewards.
This script will extract tables, look for the deep reward columns, and print detailed examples.
"""

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

import re
import json
from typing import List, Tuple
import pandas as pd
import numpy as np
from wandb_utils import get_run
from utils.table_parser import TableExtractor
from collections import defaultdict

# Configuration
ENTITY = "assert-kth"
PROJECT = "SWE-Gym-GRPO"
RUN_ID = "z5iaa297"

def parse_calls(text: str) -> List[Tuple[str, bool, str]]:
    """Parse tool calls from completion text (matches terminal.py logic)."""
    if not isinstance(text, str) or not text:
        return []
    
    calls = []
    call_iter = list(re.finditer(r"<tool_call>\s*(\{[\s\S]*?\})\s*</tool_call>", text))
    resp_iter = list(re.finditer(r"<tool_response>\s*([\s\S]*?)\s*</tool_response>", text))
    responses = [(m.start(), m.group(1) or "") for m in resp_iter]
    
    for call_m in call_iter:
        cmd = None
        try:
            parsed = json.loads(call_m.group(1))
            args = parsed.get("arguments", {}) if isinstance(parsed, dict) else {}
            cmd = args.get("cmd") if isinstance(args, dict) else None
        except Exception:
            cmd = None
            
        if not isinstance(cmd, str) or not cmd.strip():
            continue
            
        success = False
        resp_text = ""
        for (pos, content) in responses:
            if pos > call_m.end():
                cn = (content or "").strip().lower()
                if not ("command failed with exit code" in cn or "command timed out" in cn or "shell execution failed" in cn):
                    success = True
                resp_text = content or ""
                break
        calls.append((cmd.strip(), success, resp_text))
    
    return calls

def analyze_deep_reward_components(completion: str) -> dict:
    """Analyze what bonuses/penalties should be triggered based on the completion."""
    triples = parse_calls(completion)
    cmds = [c for c, _, _ in triples]
    oks = [ok for _, ok, _ in triples]
    resps = [r for _, _, r in triples]
    
    analysis = {
        'total_commands': len(cmds),
        'successful_commands': sum(oks),
        'commands': cmds[:10],  # First 10 commands
        'successes': oks[:10],
        'bonuses': {},
        'penalties': {},
        'expected_reward': 0.0
    }
    
    # Regex patterns from terminal.py
    re_search = re.compile(r"\b(rg|grep|find)\b")
    re_slice = re.compile(r"\b(sed|head|tail)\b")
    re_precise_slice = re.compile(r"\b(sed\s+-n\s+'?\d+\s*,\s*\d+\s*p'?|head\s+-n\s+\d+|tail\s+-n\s+\d+)\b")
    re_search_flags = re.compile(r"(?:^|\s)-(?:n)\b|\b--type=\w+\b")
    re_path_token = re.compile(r"[\w./-]+\.(py|ts|js|java|go|rs|md|txt|json|yaml|yml|toml|ini)\b|\b(?:src|lib|tests?|docs)/[\w./-]+\b")
    
    # Check for chain bonus (+0.25)
    bonus_chain = 0.0
    for idx, (cmd, ok, resp) in enumerate(triples[:10]):
        if ok and re_search.search(cmd):
            raw_paths = re_path_token.findall(resp)
            norm_paths = set(p if isinstance(p, str) else ''.join(p) for p in raw_paths)
            if norm_paths:
                later = triples[idx + 1:]
                if any(ok2 and re_slice.search(cmd2) and any(p in cmd2 for p in norm_paths) for cmd2, ok2, _ in later):
                    bonus_chain = 0.25
                    analysis['bonuses']['chain'] = f"+0.25 (search→path→read chain found)"
            break
    
    # Check success ratio (+0.20)
    first6 = oks[:6]
    if len(first6) == 6:
        bonus_success = 0.20 * (sum(1 for x in first6 if x) / len(first6))
        analysis['bonuses']['success_ratio'] = f"+{bonus_success:.3f} ({sum(first6)}/6 success)"
    else:
        bonus_success = 0.0
    
    # Check precise reads (+0.20)
    bonus_precise = 0.0
    if bonus_chain == 0.0:  # Don't double count
        if any(ok and re_precise_slice.search(cmd) for cmd, ok in zip(cmds, oks)):
            bonus_precise = 0.20
            analysis['bonuses']['precise_reads'] = "+0.20 (precise sed/head/tail found)"
    
    # Check scoped search (+0.15)
    bonus_scoped = 0.0
    for (cmd, ok, resp) in triples[:10]:
        if ok and re_search.search(cmd):
            if re_search_flags.search(cmd) or "/" in cmd or re_path_token.search(resp):
                bonus_scoped = 0.15
                analysis['bonuses']['scoped_search'] = "+0.15 (scoped search with flags/paths)"
                break
    
    # Don't combine scoped with chain
    if bonus_chain > 0.0:
        bonus_scoped = 0.0
        if 'scoped_search' in analysis['bonuses']:
            del analysis['bonuses']['scoped_search']
    
    # Check penalties (up to -0.20)
    penalty = 0.0
    
    # Duplicate commands
    seen = set()
    for cmd in cmds[:10]:
        if cmd in seen:
            penalty += 0.07
            analysis['penalties']['duplicate'] = "-0.07 (duplicate command)"
            break
        seen.add(cmd)
    
    # Consecutive failures
    for i in range(1, min(10, len(triples))):
        if not oks[i] and not oks[i-1] and cmds[i].strip() == cmds[i-1].strip():
            penalty += 0.07
            analysis['penalties']['consecutive_fail'] = "-0.07 (consecutive failures)"
            break
    
    # Truncated outputs
    if bonus_precise == 0.0 and any("truncated" in r.lower() for r in resps[:10]):
        penalty += 0.10
        analysis['penalties']['truncated'] = "-0.10 (truncated output before precise read)"
    
    # Calculate total
    total = max(0.0, min(1.0, bonus_chain + bonus_success + bonus_precise + bonus_scoped - min(penalty, 0.20)))
    analysis['expected_reward'] = total
    
    return analysis

def main():
    """Main analysis function."""
    print("="*80)
    print("Deep Reward Analysis - Detailed Completion Inspection")
    print("="*80)
    
    # Get run
    try:
        run = get_run(ENTITY, PROJECT, RUN_ID)
    except Exception as e:
        print(f"❌ Failed to load run: {e}")
        return
    
    # Extract tables
    print("\nExtracting tables from run...")
    extractor = TableExtractor(max_workers=5)
    tables = extractor.extract_all_training_tables(run, max_tables=50)  # Limit for faster analysis
    
    if not tables:
        print("❌ No tables found")
        return
    
    print(f"✅ Extracted {len(tables)} tables")
    
    # Find and analyze deep rewards
    all_rewards = []
    examples = {
        'zero': [],
        'low': [],
        'medium': [],
        'high': []
    }
    
    print("\n" + "="*80)
    print("Analyzing Deep Rewards Across Tables")
    print("="*80)
    
    for table_idx, df in enumerate(tables):
        # Look for deep reward column
        deep_reward_cols = [col for col in df.columns if 'terminal_exploration_depth' in col.lower()]
        
        if not deep_reward_cols:
            continue
        
        reward_col = deep_reward_cols[0]
        
        for rollout_idx in range(min(len(df), 10)):  # Analyze first 10 rollouts per table
            try:
                row = df.iloc[rollout_idx]
                completion = row.get('Completion', '')
                
                # Get the actual reward
                reward_str = row.get(reward_col, '0')
                try:
                    actual_reward = float(reward_str) if reward_str else 0.0
                except:
                    actual_reward = 0.0
                
                all_rewards.append(actual_reward)
                
                # Analyze what the reward should be
                analysis = analyze_deep_reward_components(completion)
                
                # Categorize and store examples
                if actual_reward == 0:
                    category = 'zero'
                elif actual_reward <= 0.15:
                    category = 'low'
                elif actual_reward <= 0.35:
                    category = 'medium'
                else:
                    category = 'high'
                
                if len(examples[category]) < 3:  # Keep 3 examples per category
                    examples[category].append({
                        'table': table_idx,
                        'rollout': rollout_idx,
                        'actual_reward': actual_reward,
                        'analysis': analysis,
                        'completion': completion[:500]  # First 500 chars
                    })
                    
            except Exception as e:
                continue
    
    # Print statistics
    if all_rewards:
        rewards_array = np.array(all_rewards)
        print(f"\nOverall Statistics ({len(rewards_array)} samples):")
        print(f"  Mean: {rewards_array.mean():.4f}")
        print(f"  Std: {rewards_array.std():.4f}")
        print(f"  Min: {rewards_array.min():.4f}")
        print(f"  Max: {rewards_array.max():.4f}")
        print(f"  Zeros: {(rewards_array == 0).sum()} ({(rewards_array == 0).mean()*100:.1f}%)")
        
        # Distribution
        print(f"\nDistribution:")
        print(f"  Zero (0.0): {(rewards_array == 0).sum()} ({(rewards_array == 0).mean()*100:.1f}%)")
        print(f"  Low (0-0.15): {((rewards_array > 0) & (rewards_array <= 0.15)).sum()} ({((rewards_array > 0) & (rewards_array <= 0.15)).mean()*100:.1f}%)")
        print(f"  Medium (0.15-0.35): {((rewards_array > 0.15) & (rewards_array <= 0.35)).sum()} ({((rewards_array > 0.15) & (rewards_array <= 0.35)).mean()*100:.1f}%)")
        print(f"  High (>0.35): {(rewards_array > 0.35).sum()} ({(rewards_array > 0.35).mean()*100:.1f}%)")
    
    # Print detailed examples
    print("\n" + "="*80)
    print("DETAILED EXAMPLES")
    print("="*80)
    
    for category, cat_examples in examples.items():
        if not cat_examples:
            continue
            
        print(f"\n{'='*60}")
        print(f"{category.upper()} REWARDS")
        print(f"{'='*60}")
        
        for i, ex in enumerate(cat_examples, 1):
            print(f"\n--- Example {i} (Table {ex['table']}, Rollout {ex['rollout']}) ---")
            print(f"Actual Reward: {ex['actual_reward']:.3f}")
            print(f"Expected Reward: {ex['analysis']['expected_reward']:.3f}")
            print(f"Commands: {ex['analysis']['total_commands']} total, {ex['analysis']['successful_commands']} successful")
            
            if ex['analysis']['commands']:
                print(f"First commands: {ex['analysis']['commands'][:5]}")
                print(f"Success pattern: {ex['analysis']['successes'][:5]}")
            
            if ex['analysis']['bonuses']:
                print("Bonuses triggered:")
                for bonus_name, bonus_desc in ex['analysis']['bonuses'].items():
                    print(f"  • {bonus_name}: {bonus_desc}")
            
            if ex['analysis']['penalties']:
                print("Penalties triggered:")
                for penalty_name, penalty_desc in ex['analysis']['penalties'].items():
                    print(f"  • {penalty_name}: {penalty_desc}")
            
            # Check for discrepancy
            if abs(ex['actual_reward'] - ex['analysis']['expected_reward']) > 0.05:
                print(f"⚠️  DISCREPANCY: Expected {ex['analysis']['expected_reward']:.3f} but got {ex['actual_reward']:.3f}")
    
    # Analysis summary
    print("\n" + "="*80)
    print("ANALYSIS SUMMARY")
    print("="*80)
    
    if all_rewards:
        zero_pct = (rewards_array == 0).mean() * 100
        if zero_pct > 60:
            print(f"⚠️  Very high zero rate ({zero_pct:.1f}%) - reward may be too strict")
        elif zero_pct > 40:
            print(f"⚠️  High zero rate ({zero_pct:.1f}%) - agents rarely exhibit target behaviors")
        else:
            print(f"✅ Reasonable zero rate ({zero_pct:.1f}%)")
        
        if rewards_array.max() < 0.4:
            print(f"⚠️  Low maximum reward ({rewards_array.max():.3f}) - full bonuses rarely achieved")
        else:
            print(f"✅ Good maximum reward ({rewards_array.max():.3f}) - some agents achieve multiple bonuses")
        
        mean_reward = rewards_array.mean()
        if mean_reward < 0.05:
            print(f"❌ Very poor mean reward ({mean_reward:.3f}) - insufficient learning signal")
        elif mean_reward < 0.15:
            print(f"⚠️  Low mean reward ({mean_reward:.3f}) - weak learning signal")
        else:
            print(f"✅ Good mean reward ({mean_reward:.3f}) - meaningful learning signal")
    
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()