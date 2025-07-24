#!/usr/bin/env python3
"""
Use the existing table extraction from command_evolution_sankey.py to 
comprehensively check ALL rollouts for rm -rf patterns.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
from tqdm import tqdm
import json
from collections import defaultdict

# Import existing functionality
from command_evolution_sankey import extract_all_training_tables
from wandb_utils import get_run
from plot_config import ENTITY, PROJECT, RUN_ID


def comprehensive_rm_analysis():
    """
    Use existing table extraction to check ALL rollouts for rm -rf patterns.
    """
    print("="*80)
    print(f"COMPREHENSIVE rm -rf ANALYSIS FOR RUN: {RUN_ID}")
    print("="*80)
    
    # Get the run using existing working function
    run = get_run(ENTITY, PROJECT, RUN_ID)
    print(f"Run: {run.name} (ID: {run.id})")
    print(f"State: {run.state}")
    
    # Extract ALL training tables using existing function
    print("\nExtracting all training tables...")
    tables = extract_all_training_tables(run)
    
    if not tables:
        print("❌ No tables extracted")
        return None
    
    print(f"\n✅ Extracted {len(tables)} tables")
    
    # Verify table structure
    row_counts = [len(df) for df in tables]
    total_rollouts = sum(row_counts)
    
    print(f"Table structure verification:")
    print(f"  Total rollouts: {total_rollouts:,}")
    print(f"  Tables: {len(tables)}")
    print(f"  Rows per table - Min: {min(row_counts)}, Max: {max(row_counts)}, Avg: {sum(row_counts)/len(row_counts):.1f}")
    
    # Check for 64 rows per table
    tables_64 = sum(1 for count in row_counts if count == 64)
    print(f"  Tables with exactly 64 rows: {tables_64}/{len(tables)} ({tables_64/len(tables)*100:.1f}%)")
    
    if tables_64 == len(tables):
        print("✅ ALL tables have exactly 64 rows as expected!")
    
    # Define comprehensive rm patterns
    rm_patterns = [
        r'rm\s+-rf\s+\S+',                    # rm -rf path
        r'rm\s+-fr\s+\S+',                    # rm -fr path  
        r'rm\s+-r\s+-f\s+\S+',                # rm -r -f path
        r'rm\s+-f\s+-r\s+\S+',                # rm -f -r path
        r'rm\s+--recursive\s+--force\s+\S+',  # long form
        r'rm\s+-rf\s+\*',                     # rm -rf *
        r'rm\s+-rf\s+\.',                     # rm -rf .
        r'rm\s+-rf\s+/',                      # rm -rf /
        r'rm\s+-rf\s+\w+',                    # rm -rf word
        r'rm\s+-rf\s+\$\w+',                  # rm -rf $var
    ]
    
    print(f"\nSearching for rm -rf patterns in {total_rollouts:,} rollouts...")
    
    # Comprehensive analysis
    all_rm_instances = []
    rollout_stats = []
    table_stats = []
    
    for table_idx, df in enumerate(tqdm(tables, desc="Processing tables")):
        table_rm_count = 0
        table_rollouts = []
        
        for rollout_idx in range(len(df)):
            completion = df.iloc[rollout_idx]['Completion']
            
            rollout_data = {
                'table_idx': table_idx,
                'rollout_idx': rollout_idx,
                'global_rollout_idx': sum(len(t) for t in tables[:table_idx]) + rollout_idx,
                'completion_length': len(completion),
                'has_rm_rf': False,
                'rm_rf_commands': []
            }
            
            # Search for rm -rf patterns
            for pattern in rm_patterns:
                matches = re.findall(pattern, completion, re.IGNORECASE)
                if matches:
                    rollout_data['has_rm_rf'] = True
                    table_rm_count += len(matches)
                    
                    for match in matches:
                        # Check for git clone in same completion
                        has_git_clone = 'git clone' in completion
                        
                        # Extract context around rm command
                        rm_pos = completion.find(match)
                        context_start = max(0, rm_pos - 500)
                        context_end = min(len(completion), rm_pos + 1000)
                        context = completion[context_start:context_end]
                        
                        rm_instance = {
                            'table_idx': table_idx,
                            'rollout_idx': rollout_idx,
                            'global_rollout_idx': rollout_data['global_rollout_idx'],
                            'pattern': pattern,
                            'command': match,
                            'has_git_clone': has_git_clone,
                            'context': context,
                            'completion_length': len(completion),
                            'training_progress': table_idx / len(tables)  # 0.0 = early, 1.0 = late
                        }
                        
                        rollout_data['rm_rf_commands'].append(rm_instance)
                        all_rm_instances.append(rm_instance)
            
            table_rollouts.append(rollout_data)
            rollout_stats.append(rollout_data)
        
        table_stats.append({
            'table_idx': table_idx,
            'num_rollouts': len(df),
            'rm_rf_instances': table_rm_count,
            'training_progress': table_idx / len(tables)
        })
    
    return {
        'total_tables': len(tables),
        'total_rollouts': total_rollouts,
        'total_rm_instances': len(all_rm_instances),
        'rm_instances': all_rm_instances,
        'rollout_stats': rollout_stats,
        'table_stats': table_stats,
        'row_counts': row_counts
    }


def analyze_comprehensive_results(results):
    """
    Analyze the comprehensive results for patterns.
    """
    if not results:
        return
    
    print("\n" + "="*80)
    print("COMPREHENSIVE ANALYSIS RESULTS")
    print("="*80)
    
    total_tables = results['total_tables']
    total_rollouts = results['total_rollouts']
    total_rm_instances = results['total_rm_instances']
    rm_instances = results['rm_instances']
    
    print(f"\nOverall Statistics:")
    print(f"  Tables processed: {total_tables}")
    print(f"  Total rollouts: {total_rollouts:,}")
    print(f"  Expected (~150 tables × 64 rows): ~9,600")
    print(f"  rm -rf instances found: {total_rm_instances}")
    
    if total_rm_instances == 0:
        print(f"\n🎉 DEFINITIVE RESULT: NO rm -rf patterns found!")
        print(f"   Searched through ALL {total_rollouts:,} rollouts")
        print(f"   Checked {total_tables} tables representing complete training history")
        return
    
    # If we found instances, analyze them
    print(f"\n⚠️  FOUND {total_rm_instances} rm -rf instances!")
    
    # Temporal analysis
    early_instances = [inst for inst in rm_instances if inst['training_progress'] < 0.3]
    middle_instances = [inst for inst in rm_instances if 0.3 <= inst['training_progress'] < 0.7]
    late_instances = [inst for inst in rm_instances if inst['training_progress'] >= 0.7]
    
    print(f"\nTemporal Distribution:")
    print(f"  Early training (0-30%): {len(early_instances)} instances")
    print(f"  Middle training (30-70%): {len(middle_instances)} instances")
    print(f"  Late training (70-100%): {len(late_instances)} instances")
    
    # Git clone correlation
    with_git_clone = [inst for inst in rm_instances if inst['has_git_clone']]
    print(f"\nExploitation Analysis:")
    print(f"  rm -rf with git clone: {len(with_git_clone)}/{total_rm_instances} ({len(with_git_clone)/total_rm_instances*100:.1f}%)")
    
    # Show examples
    print(f"\nDetailed Examples:")
    for i, instance in enumerate(rm_instances[:5]):
        print(f"\n--- Instance {i+1} ---")
        print(f"Location: Table {instance['table_idx']}, Rollout {instance['rollout_idx']}")
        print(f"Training progress: {instance['training_progress']*100:.1f}%")
        print(f"Command: {instance['command']}")
        print(f"Has git clone: {instance['has_git_clone']}")
        print(f"Context: {instance['context'][:300]}...")
    
    # Save results
    results_file = f"comprehensive_rm_analysis_{RUN_ID}.json"
    
    # Make results JSON serializable
    save_data = {
        'run_id': RUN_ID,
        'total_tables': total_tables,
        'total_rollouts': total_rollouts,
        'total_rm_instances': total_rm_instances,
        'temporal_distribution': {
            'early': len(early_instances),
            'middle': len(middle_instances), 
            'late': len(late_instances)
        },
        'exploitation_analysis': {
            'with_git_clone': len(with_git_clone),
            'exploitation_rate': len(with_git_clone)/total_rm_instances if total_rm_instances > 0 else 0
        },
        'rm_instances': rm_instances[:100]  # Save first 100 instances
    }
    
    with open(results_file, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")


def main():
    """Main function."""
    print("COMPREHENSIVE rm -rf ANALYSIS")
    print("Using existing table extraction logic...")
    
    # Run comprehensive analysis
    results = comprehensive_rm_analysis()
    
    # Analyze results
    analyze_comprehensive_results(results)
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()