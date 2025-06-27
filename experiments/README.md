**How are we collecting data?**

Currently logging data from `core.py` for getting data on per-step basis. Very neccesary for prediciting model times, etc.

```python
if self.num_requests_added > 0:
            self.executions_stats = pd.concat(
                [self.executions_stats,
                pd.DataFrame({
                    "num_scheduled_tokens": [scheduler_output.num_scheduled_tokens],
                    "num_total_scheduled_tokens": [scheduler_output.total_num_scheduled_tokens],
                    "scheduled_new_reqs": [scheduler_output.scheduled_new_reqs],
                    "execute_time": [execute_time],
                    "scheduler_time": [scheduler_time],
                        "update_time": [update_time],
                        "model": [self.vllm_config.model_config.model],
                        "block_size": [self.vllm_config.cache_config.block_size],
                        "gpu_memory_utilization": [self.vllm_config.cache_config.gpu_memory_utilization],
                        "num_gpu_blocks": [self.vllm_config.cache_config.num_gpu_blocks],
                        "enable_prefix_caching": [self.vllm_config.cache_config.enable_prefix_caching],
                        "max_num_sequences": [self.vllm_config.scheduler_config.max_num_seqs],
                        "max_model_len": [self.vllm_config.scheduler_config.max_model_len],
                        "temperature": [self.temperature],
                        "arrival_rate": [self.request_rate],
                        "distribution": [self.distribution],
                        "dataset": [self.dataset_name]

                })],
                ignore_index=True)
```

```python
def shutdown(self):
        print("Shutting down EngineCore...")
        # print stats on shutdown
        # create graphs for prefill and decode running times

        # create output name based on arrival_rate, temperature, model, and dataset
        output_name = f"execution_stats_{self.num_requests_added}_{self.request_rate}_{self.temperature}_{self.vllm_config.model_config.model.replace('/','-')}_{self.dataset_name}.csv"
        print(f"Saving execution stats to {output_name}")

        self.executions_stats.to_csv(output_name, index=False)
```

