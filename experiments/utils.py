# get data from csv into df
import pandas as pd
import json

def parse_csv(path: str) -> pd.DataFrame:
    """
    Read a CSV file and return a DataFrame.
    
    Args:
        path (str): The path to the CSV file.
        
    Returns:
        pd.DataFrame: The DataFrame containing the data from the CSV file.
    """
    df = pd.read_csv(path)
    return df

def format_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    parse the num_scheduled_tokens column from string to json to parse info on decodes and prefills
    """
    df['num_scheduled_tokens'] = df['num_scheduled_tokens'].apply(lambda x: x.replace("'", '"'))
    df['num_scheduled_tokens'] = df['num_scheduled_tokens'].apply(json.loads)

    def process_tokens(token_dict):
        """
        Process token dictionary to extract decode/prefill statistics
        """
        num_decodes = 0
        num_prefills = 0
        sum_decode_tokens = 0
        sum_prefill_tokens = 0
        max_prefill_tokens = 0
        
        for key, value in token_dict.items():
            if value == 1:
                # Decode
                num_decodes += 1
                sum_decode_tokens += value
            else:
                # Prefill (anything > 1)
                num_prefills += 1
                sum_prefill_tokens += value
                max_prefill_tokens = max(max_prefill_tokens, value)
        
        return pd.Series({
            'num_decodes': num_decodes,
            'num_prefills': num_prefills,
            'sum_decode_tokens': sum_decode_tokens,
            'sum_prefill_tokens': sum_prefill_tokens,
            'max_prefill_tokens': max_prefill_tokens
        })
    
    def parse_cached_tokens(scheduled_new_reqs: str):
        reqs = scheduled_new_reqs.split('lora_request')
        total_cached = 0
        max_cached_tokens = 0
        for i in range(len(reqs) - 1):
            cached = int(reqs[i].split(',')[-2].split('=')[-1])
            total_cached += cached
            max_cached_tokens = max(max_cached_tokens, cached)

        return pd.Series({
            'sum_cached_prefill_tokens': total_cached,
            'max_cached_prefill_tokens': max_cached_tokens
        })

    df[['sum_cached_prefill_tokens', 'max_cached_prefill_tokens']] = df['scheduled_new_reqs'].apply(parse_cached_tokens)

    df[['num_decodes', 'num_prefills', 'sum_decode_tokens', 'sum_uncached_prefill_tokens', 'max_uncached_prefill_tokens']] = df['num_scheduled_tokens'].apply(process_tokens)

    df['sum_prefill_tokens'] = df['sum_uncached_prefill_tokens'] + df['sum_cached_prefill_tokens']
    df['max_prefill_tokens'] = df['max_uncached_prefill_tokens'] + df['max_cached_prefill_tokens']

    df.drop(columns=['num_scheduled_tokens', 'num_decodes', 'distribution', 'dataset', 'model','scheduled_new_reqs', 'block_size', 'gpu_memory_utilization', 'num_gpu_blocks', 'enable_prefix_caching', 'max_num_sequences', 'max_model_len', 'temperature'], inplace=True) # Num_decodes is the same sum_decode_tokens, so we can drop it

    # add num_total_requests column = num_decodes + num_prefills
    df['num_total_requests'] = df['num_prefills'] + df['sum_decode_tokens']
    return df


def concatenate_dataframes(path: str) -> pd.DataFrame:
    """
    Read traverse directory and concatenate all CSV files into a single DataFrame.
    """
    import os
    import glob

    # Get all CSV files in the directory
    csv_files = glob.glob(os.path.join(path, '*.csv'))
    
    # Read and concatenate all CSV files into a single DataFrame
    df_list = [parse_csv(file) for file in csv_files]
    combined_df = pd.concat(df_list, ignore_index=True)
    
    return combined_df

import os
import re
import numpy as np
from itertools import product
from typing import Dict, List, Tuple, Optional

def find_matching_experiments(sim_dir: str, vllm_dir: str, sweep_config: Dict) -> List[Tuple[str, str, Dict]]:
    """
    Find matching experiment files between simulation and vLLM directories.
    
    Args:
        sim_dir: Path to simulation results directory
        vllm_dir: Path to vLLM results directory
        
    Returns:
        List of tuples containing (sim_file_path, vllm_file_path) for matching experiments
    """
    # Get all files in both directories
    sim_files = os.listdir(sim_dir)
    vllm_files = os.listdir(vllm_dir)
    
    # Convert to lowercase for matching
    sim_files_lower = {f.lower(): f for f in sim_files}
    vllm_files_lower = {f.lower(): f for f in vllm_files}
    
    matching_pairs = []
    
    # Define search space
    num_prompts_list = sweep_config['num_prompts']
    request_rate_list = sweep_config['request_rate']
    temperature_list = sweep_config['temperature']
    max_num_batched_tokens = sweep_config['max_num_batched_tokens']
    long_prefill_token_threshold = sweep_config['long_prefill_token_threshold']
    datasets_list = sweep_config['datasets']
    
    # Search for matching files using itertools.product
    for num_prompts, request_rate, temperature, max_tokens, threshold, dataset in product(
        num_prompts_list, request_rate_list, temperature_list, 
        max_num_batched_tokens, long_prefill_token_threshold, datasets_list
    ):
        # Create expected filename pattern
        pattern = f"exp_{num_prompts}p_{request_rate}r_{temperature}t_{max_tokens}mbt_{threshold}lpt_{dataset['name']}"
        pattern_lower = pattern.lower()
        
        # Find files that start with this pattern
        sim_matches = [f for f in sim_files_lower.keys() if f.startswith(pattern_lower)]
        vllm_matches = [f for f in vllm_files_lower.keys() if f.startswith(pattern_lower)]
        
        # If we have matches in both directories, pair them
        for sim_match in sim_matches:
            for vllm_match in vllm_matches:
                 # Exact filename match
                # print(f"Matching: {sim_match} with {vllm_match}")
                sim_path = os.path.join(sim_dir, sim_files_lower[sim_match])
                vllm_path = os.path.join(vllm_dir, vllm_files_lower[vllm_match])
                expriment_config = {"num_prompts": num_prompts, "request_rate": request_rate,
                                   "temperature": temperature, "max_num_batched_tokens": max_tokens,
                                   "long_prefill_token_threshold": threshold, "dataset": dataset['name']}
                matching_pairs.append((sim_path, vllm_path, expriment_config))
    
    return matching_pairs

def parse_sim_results(file_path: str) -> Dict:
    """
    Parse simulation results file.
    
    Args:
        file_path: Path to simulation results file
        
    Returns:
        Dictionary containing parsed metrics
    """
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Extract vLLM estimated duration
    duration_match = re.search(r'vLLM estimated Duration\(s\):\s*([0-9.]+)', content)
    vllm_duration = float(duration_match.group(1)) if duration_match else None
    
    # Extract TTFTs
    ttft_match = re.search(r'TTFTs\s*:\s*\[(.*?)\]', content, re.DOTALL)
    ttfts = []
    if ttft_match:
        ttft_str = ttft_match.group(1)
        ttfts = [float(x.strip(' ,')) for x in ttft_str.split(',') if x.strip(' ,')]
    
    # Extract TPOTs
    tpot_match = re.search(r'TPOTs\s*:\s*\[(.*?)\]', content, re.DOTALL)
    tpots = []
    if tpot_match:
        tpot_str = tpot_match.group(1)
        tpots = [float(x.strip(' ,')) for x in tpot_str.split(',') if x.strip(' ,')]
    
    # Extract E2Es
    e2e_match = re.search(r'E2Es\s*:\s*\[(.*?)\]', content, re.DOTALL)
    e2es = []
    if e2e_match:
        e2e_str = e2e_match.group(1)
        e2es = [float(x.strip(' ,')) for x in e2e_str.split(',') if x.strip(' ,')]
    
    return {
        'ttfts': ttfts,
        'tpots': tpots,
        'e2es': e2es,
        'duration': vllm_duration
    }

def parse_vllm_results(file_path: str) -> Dict:
    """
    Parse vLLM results file.
    
    Args:
        file_path: Path to vLLM results file
        
    Returns:
        Dictionary containing parsed metrics
    """
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Extract benchmark duration
    duration_match = re.search(r'Benchmark duration \(s\):\s*([0-9.]+)', content)
    benchmark_duration = float(duration_match.group(1)) if duration_match else None
    
    # Extract TTFTs
    ttft_match = re.search(r'TTFTs:\s*\[(.*?)\]', content, re.DOTALL)
    ttfts = []
    if ttft_match:
        ttft_str = ttft_match.group(1)
        ttfts = [float(x.strip(' ,')) for x in ttft_str.split(',') if x.strip(' ,')]
    
    # Extract TPOTs
    tpot_match = re.search(r'TPOTs:\s*\[(.*?)\]', content, re.DOTALL)
    tpots = []
    if tpot_match:
        tpot_str = tpot_match.group(1)
        tpots = [float(x.strip(' ,')) for x in tpot_str.split(',') if x.strip(' ,')]
    
    # Extract e2els (end-to-end latencies)
    e2e_match = re.search(r'e2els:\s*\[(.*?)\]', content, re.DOTALL)
    e2es = []
    if e2e_match:
        e2e_str = e2e_match.group(1)
        e2es = [float(x.strip(' ,')) for x in e2e_str.split(',') if x.strip(' ,')]
    
    return {
        'ttfts': ttfts,
        'tpots': tpots,
        'e2es': e2es,
        'duration': benchmark_duration
    }

def calculate_errors(sim_results: Dict, vllm_results: Dict) -> Dict:
    """
    Calculate MSE and MAPE for latency metrics and percent error for duration.
    
    Args:
        sim_results: Parsed simulation results
        vllm_results: Parsed vLLM results
        
    Returns:
        Dictionary containing error metrics
    """
    errors = {}
    
    def calculate_mape(actual, predicted):
        """Calculate Mean Absolute Percentage Error"""
        # Avoid division by zero by filtering out very small actual values
        mask = np.abs(actual) > 1e-10
        if np.sum(mask) == 0:
            return float('inf')
        return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
        # return np.mean(np.abs((actual - predicted) / actual)) * 100
    
    # Calculate MSE and MAPE for TTFT
    if sim_results['ttfts'] and vllm_results['ttfts']:
        sim_ttft = np.array(sim_results['ttfts'])
        vllm_ttft = np.array(vllm_results['ttfts'])
        min_len = min(len(sim_ttft), len(vllm_ttft))
        if min_len > 0:
            sim_subset = sim_ttft[:min_len]
            vllm_subset = vllm_ttft[:min_len]
            
            mse_ttft = np.mean((sim_subset - vllm_subset) ** 2)
            mape_ttft = calculate_mape(vllm_subset, sim_subset)
            
            errors['ttft_mse'] = mse_ttft
            # errors['ttft_mape'] = mape_ttft
            errors['ttft_accuracy'] = 100 - mape_ttft  # MAPE-based accuracy
    
    # Calculate MSE and MAPE for TPOT
    if sim_results['tpots'] and vllm_results['tpots']:
        sim_tpot = np.array(sim_results['tpots'])
        vllm_tpot = np.array(vllm_results['tpots'])
        min_len = min(len(sim_tpot), len(vllm_tpot))
        if min_len > 0:
            sim_subset = sim_tpot[:min_len]
            vllm_subset = vllm_tpot[:min_len]
            
            mse_tpot = np.mean((sim_subset - vllm_subset) ** 2)
            mape_tpot = calculate_mape(vllm_subset, sim_subset)
            
            errors['tpot_mse'] = mse_tpot
            # errors['tpot_mape'] = mape_tpot
            errors['tpot_accuracy'] = 100 - mape_tpot # MAPE-based accuracy
    
    # Calculate MSE and MAPE for E2E
    if sim_results['e2es'] and vllm_results['e2es']:
        sim_e2e = np.array(sim_results['e2es'])
        vllm_e2e = np.array(vllm_results['e2es'])
        min_len = min(len(sim_e2e), len(vllm_e2e))
        if min_len > 0:
            sim_subset = sim_e2e[:min_len]
            vllm_subset = vllm_e2e[:min_len]
            
            mse_e2e = np.mean((sim_subset - vllm_subset) ** 2)
            mape_e2e = calculate_mape(vllm_subset, sim_subset)
            
            errors['e2e_mse'] = mse_e2e
            # errors['e2e_mape'] = mape_e2e
            errors['e2e_accuracy'] = max(0, 100 - mape_e2e)  # MAPE-based accuracy
    
    # Calculate percent error for duration
    if sim_results['duration'] is not None and vllm_results['duration'] is not None:
        percent_error = abs(sim_results['duration'] - vllm_results['duration']) / vllm_results['duration'] * 100
        duration_accuracy = 100 - percent_error
        
        # errors['duration_percent_error'] = percent_error
        errors['duration_accuracy'] = duration_accuracy
    
    return errors

import matplotlib.pyplot as plt

def plot_results(plotting_data, x_axis, y_axis, sweep_configs):
    for metric, data in plotting_data.items():
        plt.figure(figsize=(10, 6))
        for x_value, y_values in data.items():
            plt.plot(sweep_configs[x_axis], y_values, marker='o', label=f'{y_axis}={x_value}')
        
        plt.title(f'{metric} vs {x_axis} and {y_axis}')
        plt.xlabel(x_axis)
        plt.ylabel(metric)
        plt.xticks(sweep_configs[x_axis])
        plt.legend()
        plt.grid()
        plt.show()