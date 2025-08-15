#!/usr/bin/env python3
"""
Deep analysis of whether terminal_exploration_depth_reward correlates with actual
problem-solving quality and debugging strategies, not just technical execution.
"""

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

import re
import json
from typing import List, Tuple, Dict
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
    """Parse tool calls from completion text."""
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

def extract_problem_context(prompt: str) -> str:
    """Extract the actual problem description from the prompt."""
    # Split on user marker and get the actual problem
    parts = prompt.split("<|im_start|>user")
    if len(parts) > 1:
        problem = parts[-1].strip()
        # Take first 200 chars for problem summary
        return problem[:200] + "..." if len(problem) > 200 else problem
    return prompt[:200] + "..." if len(prompt) > 200 else prompt

def analyze_strategy_quality(completion: str, prompt: str) -> Dict:
    """Analyze the actual problem-solving strategy quality."""
    triples = parse_calls(completion)
    cmds = [c for c, _, _ in triples]
    oks = [ok for _, ok, _ in triples]
    resps = [r for _, _, r in triples]
    
    analysis = {
        'problem_focus': 0,      # How well does agent focus on the problem?
        'exploration_quality': 0, # Quality of exploration strategy
        'progress_made': 0,       # Did agent make meaningful progress?
        'strategic_thinking': 0,  # Evidence of strategic approach
        'efficiency': 0,          # Efficient use of commands
        'total_score': 0
    }
    
    # Extract key terms from problem to check if agent focuses on them
    problem_text = extract_problem_context(prompt).lower()
    key_terms = set()
    
    # Extract likely important terms (function names, class names, etc.)
    for word in problem_text.split():
        if len(word) > 3 and ('_' in word or word[0].isupper() or 'def ' in problem_text):
            key_terms.add(word.strip('.,()[]'))
    
    # 1. Problem Focus Analysis
    problem_focused_searches = 0
    for cmd, ok, resp in triples[:10]:
        if ok and any(term in cmd.lower() for term in key_terms):
            problem_focused_searches += 1
    
    analysis['problem_focus'] = min(10, problem_focused_searches * 2)
    
    # 2. Exploration Quality
    # Good: starts broad then narrows down
    # Bad: random searches or gets stuck repeating
    search_pattern_score = 0
    
    # Check for logical progression
    search_commands = [(i, cmd) for i, (cmd, ok, _) in enumerate(triples) if ok and any(x in cmd for x in ['rg', 'grep', 'find', 'ls'])]
    
    if len(search_commands) >= 2:
        # Check if searches become more specific over time
        early_searches = [cmd for i, cmd in search_commands[:3]]
        later_searches = [cmd for i, cmd in search_commands[3:6]]
        
        # Count flags and specificity
        early_specificity = sum(cmd.count('-') + cmd.count('/') for cmd in early_searches)
        later_specificity = sum(cmd.count('-') + cmd.count('/') for cmd in later_searches)
        
        if len(later_searches) > 0 and later_specificity > early_specificity:
            search_pattern_score += 5
        
        # Penalize exact repetition
        if len(set(cmd.strip() for cmd in early_searches + later_searches)) < len(early_searches + later_searches) * 0.7:
            search_pattern_score -= 3
    
    analysis['exploration_quality'] = max(0, min(10, search_pattern_score + 3))
    
    # 3. Progress Made
    # Look for evidence of finding relevant files/functions
    progress_indicators = 0
    
    for cmd, ok, resp in triples:
        if ok and resp:
            # Check if response contains promising results
            if any(term in resp.lower() for term in key_terms):
                progress_indicators += 1
            if 'def ' in resp or 'class ' in resp:
                progress_indicators += 1
            if len(resp.strip()) > 100:  # Substantial output
                progress_indicators += 1
    
    analysis['progress_made'] = min(10, progress_indicators)
    
    # 4. Strategic Thinking
    strategy_score = 0
    
    # Look for evidence of strategic approach
    # - Using different search tools appropriately
    # - Following up on promising results
    # - Using sed/head/tail to examine specific sections
    
    used_tools = set()
    for cmd, ok, _ in triples:
        if ok:
            first_word = cmd.split()[0] if cmd.split() else ""
            used_tools.add(first_word)
    
    # Bonus for tool diversity
    if len(used_tools) >= 3:
        strategy_score += 3
    if 'rg' in used_tools and any(x in used_tools for x in ['sed', 'head', 'tail']):
        strategy_score += 3  # Good search-then-examine pattern
    if 'find' in used_tools and 'rg' in used_tools:
        strategy_score += 2  # Using complementary search tools
    
    analysis['strategic_thinking'] = min(10, strategy_score)
    
    # 5. Efficiency
    # Penalize excessive failed attempts, reward quick success
    total_commands = len(triples)
    successful_commands = sum(oks)
    
    if total_commands > 0:
        success_rate = successful_commands / total_commands
        efficiency_score = success_rate * 10
        
        # Penalize excessive commands (>20 suggests thrashing)
        if total_commands > 20:
            efficiency_score -= (total_commands - 20) * 0.2
        
        analysis['efficiency'] = max(0, min(10, efficiency_score))
    
    # Calculate total strategy score
    analysis['total_score'] = (
        analysis['problem_focus'] + 
        analysis['exploration_quality'] + 
        analysis['progress_made'] + 
        analysis['strategic_thinking'] + 
        analysis['efficiency']
    ) / 5
    
    return analysis

def calculate_deep_reward(completion: str) -> float:
    """Calculate the deep reward (replicated from terminal.py)."""
    triples = parse_calls(completion)
    cmds = [c for c, _, _ in triples]
    oks = [ok for _, ok, _ in triples]
    resps = [r for _, _, r in triples]
    
    # Regex patterns
    re_search = re.compile(r"\b(rg|grep|find)\b")
    re_slice = re.compile(r"\b(sed|head|tail)\b")
    re_precise_slice = re.compile(r"\b(sed\s+-n\s+'?\d+\s*,\s*\d+\s*p'?|head\s+-n\s+\d+|tail\s+-n\s+\d+)\b")
    re_search_flags = re.compile(r"(?:^|\s)-(?:n)\b|\b--type=\w+\b")
    re_path_token = re.compile(r"[\w./-]+\.(py|ts|js|java|go|rs|md|txt|json|yaml|yml|toml|ini)\b|\b(?:src|lib|tests?|docs)/[\w./-]+\b")
    
    # Chain bonus
    bonus_chain = 0.0
    for idx, (cmd, ok, resp) in enumerate(triples[:10]):
        if ok and re_search.search(cmd):
            raw_paths = re_path_token.findall(resp)
            norm_paths = set(p if isinstance(p, str) else ''.join(p) for p in raw_paths)
            if norm_paths:
                later = triples[idx + 1:]
                if any(ok2 and re_slice.search(cmd2) and any(p in cmd2 for p in norm_paths) for cmd2, ok2, _ in later):
                    bonus_chain = 0.25
                    break
    
    # Success ratio
    first6 = oks[:6]
    if len(first6) == 6:
        bonus_success = 0.20 * (sum(1 for x in first6 if x) / len(first6))
    else:
        bonus_success = 0.0
    
    # Precise reads
    bonus_precise = 0.0
    if bonus_chain == 0.0:
        if any(ok and re_precise_slice.search(cmd) for cmd, ok in zip(cmds, oks)):
            bonus_precise = 0.20
    
    # Scoped search
    bonus_scoped = 0.0
    for (cmd, ok, resp) in triples[:10]:
        if ok and re_search.search(cmd):
            if re_search_flags.search(cmd) or "/" in cmd or re_path_token.search(resp):
                bonus_scoped = 0.15
                break
    
    if bonus_chain > 0.0:
        bonus_scoped = 0.0
    
    # Penalties
    penalty = 0.0
    seen = set()
    for cmd in cmds[:10]:
        if cmd in seen:
            penalty += 0.07
            break
        seen.add(cmd)
    
    for i in range(1, min(10, len(triples))):
        if not oks[i] and not oks[i-1] and cmds[i].strip() == cmds[i-1].strip():
            penalty += 0.07
            break
    
    if bonus_precise == 0.0 and any("truncated" in r.lower() for r in resps[:10]):
        penalty += 0.10
    
    total = max(0.0, min(1.0, bonus_chain + bonus_success + bonus_precise + bonus_scoped - min(penalty, 0.20)))
    return total

def main():
    """Main analysis function."""
    print("="*80)
    print("Strategy Quality vs Deep Reward Correlation Analysis")
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
    tables = extractor.extract_all_training_tables(run, max_tables=20)  # Analyze more tables
    
    if not tables:
        print("❌ No tables found")
        return
    
    print(f"✅ Extracted {len(tables)} tables")
    
    # Analyze strategy vs reward correlation
    samples = []
    
    print("\n" + "="*80)
    print("Analyzing Strategy Quality vs Deep Reward Correlation")
    print("="*80)
    
    for table_idx, df in enumerate(tables):
        # Look for deep reward column
        deep_reward_cols = [col for col in df.columns if 'terminal_exploration_depth' in col.lower()]
        
        if not deep_reward_cols:
            continue
        
        reward_col = deep_reward_cols[0]
        
        # Sample evenly across rollouts
        for rollout_idx in range(0, min(len(df), 20), 2):  # Every 2nd rollout for diversity
            try:
                row = df.iloc[rollout_idx]
                completion = row.get('Completion', '')
                prompt = row.get('Prompt', '')
                
                # Get actual reward
                reward_str = row.get(reward_col, '0')
                try:
                    actual_reward = float(reward_str) if reward_str else 0.0
                except:
                    actual_reward = 0.0
                
                # Calculate expected reward to verify consistency
                expected_reward = calculate_deep_reward(completion)
                
                # Analyze strategy quality
                strategy_analysis = analyze_strategy_quality(completion, prompt)
                
                samples.append({
                    'table': table_idx,
                    'rollout': rollout_idx,
                    'actual_reward': actual_reward,
                    'expected_reward': expected_reward,
                    'problem': extract_problem_context(prompt),
                    'strategy_score': strategy_analysis['total_score'],
                    'problem_focus': strategy_analysis['problem_focus'],
                    'exploration_quality': strategy_analysis['exploration_quality'],
                    'progress_made': strategy_analysis['progress_made'],
                    'strategic_thinking': strategy_analysis['strategic_thinking'],
                    'efficiency': strategy_analysis['efficiency'],
                    'completion_preview': completion[:800]  # More context
                })
                
            except Exception as e:
                continue
    
    if not samples:
        print("❌ No valid samples found")
        return
    
    print(f"Analyzed {len(samples)} samples")
    
    # Sort by reward to examine correlation
    samples.sort(key=lambda x: x['actual_reward'])
    
    # Group samples by reward level for detailed analysis
    low_reward = [s for s in samples if s['actual_reward'] <= 0.1]
    med_reward = [s for s in samples if 0.1 < s['actual_reward'] <= 0.3]
    high_reward = [s for s in samples if s['actual_reward'] > 0.3]
    
    print(f"\nReward Distribution:")
    print(f"  Low (≤0.1): {len(low_reward)} samples")
    print(f"  Medium (0.1-0.3): {len(med_reward)} samples") 
    print(f"  High (>0.3): {len(high_reward)} samples")
    
    # Analyze correlation
    rewards = [s['actual_reward'] for s in samples]
    strategy_scores = [s['strategy_score'] for s in samples]
    
    correlation = np.corrcoef(rewards, strategy_scores)[0, 1]
    print(f"\nCorrelation between Deep Reward and Strategy Quality: {correlation:.3f}")
    
    # Detailed examples from each category
    print("\n" + "="*80)
    print("DETAILED STRATEGY ANALYSIS BY REWARD LEVEL")
    print("="*80)
    
    categories = [
        ("LOW REWARD (≤0.1)", low_reward[:3]),
        ("MEDIUM REWARD (0.1-0.3)", med_reward[:3]), 
        ("HIGH REWARD (>0.3)", high_reward[:3])
    ]
    
    for category_name, category_samples in categories:
        if not category_samples:
            continue
            
        print(f"\n{'='*60}")
        print(f"{category_name}")
        print(f"{'='*60}")
        
        for i, sample in enumerate(category_samples, 1):
            print(f"\n--- Example {i} (Table {sample['table']}, Rollout {sample['rollout']}) ---")
            print(f"Actual Reward: {sample['actual_reward']:.3f}")
            print(f"Strategy Score: {sample['strategy_score']:.1f}/10")
            print(f"  Problem Focus: {sample['problem_focus']:.1f}/10")
            print(f"  Exploration Quality: {sample['exploration_quality']:.1f}/10") 
            print(f"  Progress Made: {sample['progress_made']:.1f}/10")
            print(f"  Strategic Thinking: {sample['strategic_thinking']:.1f}/10")
            print(f"  Efficiency: {sample['efficiency']:.1f}/10")
            
            print(f"\nProblem: {sample['problem']}")
            
            # Show first few commands to illustrate strategy
            triples = parse_calls(sample['completion_preview'])
            if triples:
                print(f"\nFirst commands:")
                for j, (cmd, ok, resp) in enumerate(triples[:5]):
                    status = "✓" if ok else "✗"
                    print(f"  {j+1}. {status} {cmd}")
                    if resp and len(resp) > 50:
                        print(f"     → {resp[:100]}{'...' if len(resp) > 100 else ''}")
            
            # Highlight concerning patterns
            if sample['actual_reward'] > 0.3 and sample['strategy_score'] < 4:
                print(f"⚠️  HIGH REWARD BUT POOR STRATEGY - Possible reward gaming!")
            elif sample['actual_reward'] < 0.1 and sample['strategy_score'] > 6:
                print(f"⚠️  LOW REWARD BUT GOOD STRATEGY - Reward may be too harsh!")
    
    # Final correlation analysis
    print("\n" + "="*80)
    print("CORRELATION ANALYSIS SUMMARY")
    print("="*80)
    
    # Calculate average strategy scores by reward tier
    if low_reward:
        avg_low_strategy = sum(s['strategy_score'] for s in low_reward) / len(low_reward)
    else:
        avg_low_strategy = 0
        
    if med_reward:
        avg_med_strategy = sum(s['strategy_score'] for s in med_reward) / len(med_reward)
    else:
        avg_med_strategy = 0
        
    if high_reward:
        avg_high_strategy = sum(s['strategy_score'] for s in high_reward) / len(high_reward)
    else:
        avg_high_strategy = 0
    
    print(f"Average Strategy Scores by Reward Level:")
    print(f"  Low Reward (≤0.1): {avg_low_strategy:.2f}/10")
    print(f"  Med Reward (0.1-0.3): {avg_med_strategy:.2f}/10")
    print(f"  High Reward (>0.3): {avg_high_strategy:.2f}/10")
    
    print(f"\nOverall Correlation: {correlation:.3f}")
    
    if correlation > 0.6:
        print("✅ STRONG POSITIVE CORRELATION - Deep reward aligns well with strategy quality")
    elif correlation > 0.3:
        print("⚠️  MODERATE CORRELATION - Some alignment but room for improvement")
    else:
        print("❌ WEAK CORRELATION - Deep reward may not capture strategy quality well")
    
    # Check for specific problems
    gaming_cases = [s for s in samples if s['actual_reward'] > 0.3 and s['strategy_score'] < 4]
    harsh_cases = [s for s in samples if s['actual_reward'] < 0.1 and s['strategy_score'] > 6]
    
    if gaming_cases:
        print(f"⚠️  Found {len(gaming_cases)} potential reward gaming cases")
    if harsh_cases:
        print(f"⚠️  Found {len(harsh_cases)} cases where good strategy got low reward")
    
    print("\n✅ Strategy correlation analysis complete!")

if __name__ == "__main__":
    main()