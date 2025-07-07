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