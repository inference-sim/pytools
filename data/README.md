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
python benchmark_runner.py config.yaml --runs 2 --output ./my_results
```

Config example

```yaml
test:
  baseline:
    name: "basic_benchmark"
    description: "Basic vLLM performance test"
    model: "Qwen/Qwen2.5-0.5B"
    runs: 1

    # vLLM server parameters
    vllm:
      gpu_memory_utilization: 0.4
      enable_prefix_caching: true
      disable_log_requests: true
      block_size: 16
      max_model_len: 2048
      max_num_batched_tokens: 2048
      max_num_seqs: 256
      long_prefill_token_threshold: 1000000

    # Benchmark parameters
    benchmark:
      backend: "vllm"
      dataset_name: "sharegpt"
      dataset_path: "ShareGPT_V3_unfiltered_cleaned_split.json"
      num_prompts: 5000
      request_rate: 50
      temperature: 0.0
```

## Tests Available

Results are saved as timestamped text files in the output folder.