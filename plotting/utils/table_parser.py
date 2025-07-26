"""
Consolidated HTML table parsing utilities for WandB rollout data.

This module contains all the common logic for extracting and parsing HTML tables
from WandB runs, eliminating code duplication across plotting scripts.
"""

import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple, Any
import wandb


class TableExtractor:
    """
    Handles extraction and parsing of HTML tables from WandB runs.
    
    This class consolidates all table processing logic that was previously
    duplicated across multiple plotting scripts.
    """
    
    def __init__(self, max_workers: int = 10):
        """
        Initialize the table extractor.
        
        Args:
            max_workers: Number of parallel workers for table processing
        """
        self.max_workers = max_workers
    
    def extract_all_training_tables(
        self, 
        run: wandb.apis.public.Run, 
        max_tables: Optional[int] = None
    ) -> List[pd.DataFrame]:
        """
        Extract all tables from training history using parallel processing.
        
        Args:
            run: WandB run object
            max_tables: Limit number of tables for testing (None = all)
            
        Returns:
            List of DataFrames containing parsed table data
        """
        print("=" * 60)
        print("Extracting All Training Tables (Parallel)")
        print("=" * 60)
        
        # Get history with table data
        history = self._get_table_history(run)
        
        if max_tables:
            history = history.head(max_tables)
            print(f"Limited to {len(history)} tables for testing")
        
        # Prepare data for parallel processing
        table_data = [(i, row) for i, (_, row) in enumerate(history.iterrows())]
        
        tables = []
        failed_count = 0
        
        # Use ThreadPoolExecutor for parallel downloads
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(self._process_single_table, run, i, row): i 
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
                        print(f"Error processing table {index}: {e}")
                        failed_count += 1
                    pbar.update(1)
        
        # Sort tables by index to maintain order
        tables.sort(key=lambda x: x[0])
        tables = [df for _, df in tables]
        
        print(f"\nSuccessfully extracted {len(tables)} tables ({failed_count} failed)")
        total_rollouts = sum(len(df) for df in tables)
        print(f"Total rollouts: {total_rollouts}")
        
        return tables
    
    def _get_table_history(self, run: wandb.apis.public.Run) -> pd.DataFrame:
        """Get history entries that contain table data."""
        try:
            # Import here to avoid circular dependency
            from wandb_utils import get_history
            history = get_history(run, keys=['train/global_step', 'table'])
            table_history = history[~history['table'].isna()]
            print(f"Found {len(table_history)} table entries in training history")
            return table_history
        except ImportError:
            # Fallback implementation
            history = run.history(keys=['train/global_step', 'table'])
            table_history = history[~history['table'].isna()]
            print(f"Found {len(table_history)} table entries in training history")
            return table_history
    
    def _process_single_table(
        self, 
        run: wandb.apis.public.Run, 
        table_index: int, 
        row_data: Any
    ) -> Optional[pd.DataFrame]:
        """
        Process a single table - used for parallel processing.
        
        Args:
            run: WandB run object
            table_index: Index of the table
            row_data: Row data from history
            
        Returns:
            DataFrame with parsed table data, or None if failed
        """
        try:
            global_step = int(row_data['train/global_step'])
            table_info = row_data['table']
            table_path = table_info['path']
            
            # Download table with caching enabled, but to cache/ directory
            table_file = run.file(table_path)
            html_content = table_file.download(replace=False, exist_ok=True)
            
            with open(html_content.name, 'r', encoding='utf-8') as f:
                html_data = f.read()
            
            # Parse HTML table
            df = self._parse_html_table(html_data)
            
            if df is not None:
                df['global_step'] = global_step
                df['table_index'] = table_index
                return df
                
        except Exception as e:
            # Silently fail for individual tables to avoid spam
            return None
        
        return None
    
    def _parse_html_table(self, html_data: str) -> Optional[pd.DataFrame]:
        """
        Parse HTML table data into a DataFrame.
        
        Args:
            html_data: Raw HTML content
            
        Returns:
            DataFrame with parsed table data, or None if parsing failed
        """
        soup = BeautifulSoup(html_data, 'html.parser')
        table = soup.find('table')
        
        if not table:
            return None
        
        # Extract headers
        headers = self._extract_headers(table)
        if not headers:
            return None
        
        # Extract rows
        rows = self._extract_rows(table)
        if not rows:
            return None
        
        # Create DataFrame
        return self._create_dataframe(headers, rows)
    
    def _extract_headers(self, table) -> List[str]:
        """Extract table headers."""
        headers = []
        header_row = table.find('thead')
        
        if header_row:
            headers = [th.get_text(strip=True) for th in header_row.find_all('th')]
        else:
            # Fallback: use first row as headers
            first_row = table.find('tr')
            if first_row:
                headers = [td.get_text(strip=True) for td in first_row.find_all(['td', 'th'])]
        
        return headers
    
    def _extract_rows(self, table) -> List[List[str]]:
        """Extract table rows."""
        rows = []
        tbody = table.find('tbody')
        
        if tbody:
            for tr in tbody.find_all('tr'):
                row = [td.get_text(separator='\n', strip=True) for td in tr.find_all('td')]
                if row:  # Only add non-empty rows
                    rows.append(row)
        else:
            # Fallback: all rows except first (if used as header)
            all_rows = table.find_all('tr')
            start_idx = 1 if not table.find('thead') else 0
            for tr in all_rows[start_idx:]:
                row = [td.get_text(separator='\n', strip=True) for td in tr.find_all('td')]
                if row:
                    rows.append(row)
        
        return rows
    
    def _create_dataframe(self, headers: List[str], rows: List[List[str]]) -> pd.DataFrame:
        """Create DataFrame from headers and rows."""
        # Ensure all rows have same length as headers
        max_cols = len(headers)
        padded_rows = []
        
        for row in rows:
            padded_row = row[:max_cols]  # Truncate if too long
            padded_row.extend([''] * (max_cols - len(padded_row)))  # Pad if too short
            padded_rows.append(padded_row)
        
        return pd.DataFrame(padded_rows, columns=headers)
    
    def extract_rollout_data(
        self, 
        tables: List[pd.DataFrame], 
        problem_key_func: Optional[callable] = None
    ) -> Dict[str, List[Dict]]:
        """
        Extract rollout data organized by problem/prompt.
        
        Args:
            tables: List of DataFrames from extract_all_training_tables
            problem_key_func: Function to extract problem key from prompt (optional)
            
        Returns:
            Dict mapping problem keys to lists of rollout data
        """
        if problem_key_func is None:
            problem_key_func = self._default_problem_key
        
        rollout_data = {}
        
        for table_idx, df in enumerate(tables):
            for rollout_idx in range(len(df)):
                try:
                    row = df.iloc[rollout_idx]
                    
                    # Extract key fields
                    prompt = row.get('Prompt', '')
                    completion = row.get('Completion', '')
                    reward = self._parse_reward(row.get('Reward', 0))
                    
                    # Get problem key
                    problem_key = problem_key_func(prompt)
                    
                    # Store rollout data
                    rollout_info = {
                        'prompt': prompt,
                        'completion': completion,
                        'reward': reward,
                        'table_index': table_idx,
                        'rollout_index': rollout_idx,
                        'global_step': row.get('global_step', 0)
                    }
                    
                    if problem_key not in rollout_data:
                        rollout_data[problem_key] = []
                    
                    rollout_data[problem_key].append(rollout_info)
                    
                except Exception as e:
                    # Skip problematic rollouts
                    continue
        
        print(f"Extracted rollout data for {len(rollout_data)} unique problems")
        return rollout_data
    
    def _default_problem_key(self, prompt: str) -> str:
        """
        Default function to extract problem key from prompt.
        
        Args:
            prompt: Full prompt text
            
        Returns:
            Unique key for the problem
        """
        # Split on "<|im_start|>user" and take the last part (the actual user prompt)
        parts = prompt.split("<|im_start|>user")
        if len(parts) > 1:
            return parts[-1].strip()
        return prompt.strip()
    
    def _parse_reward(self, reward_value: Any) -> float:
        """Parse reward value to float."""
        try:
            if isinstance(reward_value, str):
                return float(reward_value)
            return float(reward_value)
        except (ValueError, TypeError):
            return 0.0


# Convenience functions for backward compatibility
def extract_all_training_tables(
    run: wandb.apis.public.Run, 
    max_tables: Optional[int] = None
) -> List[pd.DataFrame]:
    """Convenience function for extracting tables."""
    extractor = TableExtractor()
    return extractor.extract_all_training_tables(run, max_tables)


def process_single_table(
    run: wandb.apis.public.Run, 
    table_index: int, 
    row_data: Any
) -> Optional[pd.DataFrame]:
    """Convenience function for processing single table."""
    extractor = TableExtractor()
    return extractor._process_single_table(run, table_index, row_data)