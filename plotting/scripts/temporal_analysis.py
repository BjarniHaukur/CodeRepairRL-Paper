#!/usr/bin/env python3
"""
Analyze the temporal distribution of rm -rf patterns from cached files.
"""

import json
import os
import re
from collections import defaultdict


def extract_table_number_from_filename(filename):
    """Extract table number from filename like table_1642_7088da670ab9d7bd4a96.html"""
    match = re.search(r'table_(\d+)_', filename)
    if match:
        return int(match.group(1))
    return None


def analyze_temporal_distribution():
    """Analyze temporal distribution of rm -rf patterns."""
    
    # Load results
    with open('cached_rm_analysis_results.json', 'r') as f:
        data = json.load(f)
    
    print("="*80)
    print("TEMPORAL ANALYSIS OF rm -rf PATTERNS")
    print("="*80)
    
    print(f"\nOverall Statistics:")
    print(f"  Total files: {data['total_files']}")
    print(f"  Total rollouts: {data['total_rollouts']}")
    print(f"  Total rm -rf instances: {data['total_rm_instances']}")
    print(f"  Files with rm -rf: {data['files_with_rm_rf']}")
    
    # Group instances by file and extract table numbers
    instances_by_file = defaultdict(list)
    table_numbers = []
    
    for instance in data['rm_instances']:
        filename = instance['file_name']
        table_num = extract_table_number_from_filename(filename)
        if table_num is not None:
            instances_by_file[filename].append(instance)
            table_numbers.append(table_num)
    
    if not table_numbers:
        print("❌ Could not extract table numbers from filenames")
        return
    
    # Sort table numbers to understand temporal progression
    table_numbers.sort()
    min_table = min(table_numbers)
    max_table = max(table_numbers)
    
    print(f"\nTemporal Analysis:")
    print(f"  Table number range: {min_table} to {max_table}")
    print(f"  Total span: {max_table - min_table} steps")
    
    # Calculate training progress for each instance
    instances_with_progress = []
    for instance in data['rm_instances']:
        filename = instance['file_name']
        table_num = extract_table_number_from_filename(filename)
        if table_num is not None:
            # Calculate progress as position in training (0.0 = early, 1.0 = late)
            progress = (table_num - min_table) / (max_table - min_table) if max_table > min_table else 0.0
            instances_with_progress.append({
                **instance,
                'table_number': table_num,
                'training_progress': progress
            })
    
    # Sort by training progress
    instances_with_progress.sort(key=lambda x: x['training_progress'])
    
    # Analyze distribution across training phases
    early_instances = [inst for inst in instances_with_progress if inst['training_progress'] < 0.33]
    middle_instances = [inst for inst in instances_with_progress if 0.33 <= inst['training_progress'] < 0.67]
    late_instances = [inst for inst in instances_with_progress if inst['training_progress'] >= 0.67]
    
    print(f"\nTraining Phase Distribution:")
    print(f"  Early training (0-33%): {len(early_instances)} instances")
    print(f"  Middle training (33-67%): {len(middle_instances)} instances")
    print(f"  Late training (67-100%): {len(late_instances)} instances")
    
    # Check if pattern increases over time
    if len(late_instances) > len(early_instances):
        print(f"  ⚠️  Pattern INCREASES during training!")
        print(f"     Late training has {len(late_instances) - len(early_instances)} more instances than early")
    elif len(early_instances) > len(late_instances):
        print(f"  ✅ Pattern DECREASES during training")
        print(f"     Early training has {len(early_instances) - len(late_instances)} more instances than late")
    else:
        print(f"  ➡️  Pattern remains STABLE during training")
    
    # Exploitation analysis
    with_git_clone = [inst for inst in instances_with_progress if inst['has_git_clone']]
    early_exploitation = [inst for inst in early_instances if inst['has_git_clone']]
    middle_exploitation = [inst for inst in middle_instances if inst['has_git_clone']]
    late_exploitation = [inst for inst in late_instances if inst['has_git_clone']]
    
    print(f"\nExploitation Pattern Analysis:")
    print(f"  Total rm -rf + git clone: {len(with_git_clone)}/{len(instances_with_progress)} ({len(with_git_clone)/len(instances_with_progress)*100:.1f}%)")
    early_pct = (len(early_exploitation)/len(early_instances)*100) if len(early_instances) > 0 else 0
    middle_pct = (len(middle_exploitation)/len(middle_instances)*100) if len(middle_instances) > 0 else 0
    late_pct = (len(late_exploitation)/len(late_instances)*100) if len(late_instances) > 0 else 0
    
    print(f"  Early exploitation: {len(early_exploitation)}/{len(early_instances)} ({early_pct:.1f}%)")
    print(f"  Middle exploitation: {len(middle_exploitation)}/{len(middle_instances)} ({middle_pct:.1f}%)")
    print(f"  Late exploitation: {len(late_exploitation)}/{len(late_instances)} ({late_pct:.1f}%)")
    
    # Show specific examples of exploitation
    if with_git_clone:
        print(f"\nExploitation Examples (rm -rf + git clone):")
        for i, instance in enumerate(with_git_clone[:3]):
            print(f"\n--- Exploitation Example {i+1} ---")
            print(f"Table: {instance['table_number']} (training progress: {instance['training_progress']*100:.1f}%)")
            print(f"Command: {instance['command']}")
            print(f"Context snippet: {instance['context'][:200]}...")
    
    # Show progression timeline
    print(f"\nTemporal Timeline:")
    print(f"  First rm -rf at table {instances_with_progress[0]['table_number']} ({instances_with_progress[0]['training_progress']*100:.1f}%)")
    print(f"  Last rm -rf at table {instances_with_progress[-1]['table_number']} ({instances_with_progress[-1]['training_progress']*100:.1f}%)")
    
    # Files with most instances
    print(f"\nFiles with Most rm -rf Instances:")
    file_counts = defaultdict(int)
    for instance in instances_with_progress:
        file_counts[instance['file_name']] += 1
    
    sorted_files = sorted(file_counts.items(), key=lambda x: x[1], reverse=True)
    for filename, count in sorted_files[:5]:
        table_num = extract_table_number_from_filename(filename)
        progress = (table_num - min_table) / (max_table - min_table) if max_table > min_table else 0.0
        print(f"  {filename}: {count} instances (table {table_num}, {progress*100:.1f}% progress)")
    
    print(f"\n{'='*80}")
    print("TEMPORAL ANALYSIS COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    analyze_temporal_distribution()