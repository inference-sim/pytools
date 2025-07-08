# vLLM Benchmark Runner

Simple tool to run vLLM performance benchmarks with different configurations.

## Setup

1. Install vLLM:
```bash
pip install vllm
```

2. Copy files to vLLM benchmarks directory:
```bash
# Navigate to your vllm installation
cd vllm/benchmarks/

# Copy the benchmark runner and config
cp /path/to/benchmark_runner.py .
cp /path/to/config.yaml .
```

## Usage

Specify output folder:
```bash
python benchmark_runner.py config.yaml --output ./my_results
```

Config example

```yaml
# Generated test configurations
test:

  baseline:
    name: "exp1"
    description: "Basic vLLM performance test"
    model: "Qwen/Qwen2.5-0.5B"
    runs: 1

    # vLLM server parameters
    vllm:
      gpu_memory_utilization: 0.9
      enable_prefix_caching: true
      disable_log_requests: false
      block_size: 16 
      max_model_len: 2048
      max_num_batched_tokens: 2048
      max_num_seqs: 256
      long_prefill_token_threshold: 1000000
      seed: 42  # Set a seed for reproducibility

    # Benchmark parameters
    benchmark:
      backend: "vllm"
      dataset_name: "sharegpt"
      dataset_path: "ShareGPT_V3_unfiltered_cleaned_split.json"
      num_prompts: 100
      request_rate: 16
      sharegpt_output_len: 0 # Set to 0 for no output length limit
      temperature: 0.0
      seed: 42  # Set a seed for reproducibility
```
## Generating configs

A config file that sweeps over paramters can be generated via:

```bash
python config_generator.py
```

To change the parameters sweeped over edit `main()`

```python
# Define parameter sweep ranges
    num_prompts_list = [400]
    request_rate_list = [4]
    temperature_list = [0.0]
    max_num_batched_tokens = [256, 512, 1024, 2048, 4096]
    long_prefill_token_threshold = [16, 32, 64, 128, 256, 512, 1024]
    datasets_list = [
        {'name': 'sharegpt', 'path': 'ShareGPT_V3_unfiltered_cleaned_split.json'},
    ]
```


## Tests Available

Results are saved as timestamped text files in the output folder.
