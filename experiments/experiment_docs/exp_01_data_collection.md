# Experiment 001: Data Collection

**Date**: 06/18/25
**Author**: Satyam
**Status**: Completed

## Purpose/Goal

Create a regression model that predicts execution times for the `step()` function in the simulator using real vLLM profiling data. We need to collect timing data from the different phases of vLLM's step function (schedule, execute, update) to understand performance characteristics and build predictive models.

**Related Issues**: 
- [Link to GitHub issue if relevant]

## How to Reproduce

**Code Changes**:
Modified `vllm/vllm/v1/engine/core.py` to add timing instrumentation:

```python
# Check for any requests remaining in the scheduler - unfinished,
# or finished and not yet removed from the batch.

timestart_s = time.perf_counter()
### Scheduling Phase
if not self.scheduler.has_requests():
    return {}, False
scheduler_output = self.scheduler.schedule()

timeend_s = time.perf_counter()

### Execution Phase
model_output = self.execute_model(scheduler_output)

timeend_e = time.perf_counter()

### Post-processing Phase
engine_core_outputs = self.scheduler.update_from_output(
    scheduler_output, model_output)  # type: ignore

timeend_u = time.perf_counter()

self.scheduler_time = timeend_s - timestart_s
self.execute_time = timeend_e - timeend_s
self.update_time = timeend_u - timeend_e
```

Added shutdown method to save data:

```python
def shutdown(self):
    print("Shutting down EngineCore...")
    # create output name based on arrival_rate, temperature, model, and dataset
    output_folder = './execution_step_data/train/'
    
    output_file = Path(output_folder) / f"execution_stats_{self.num_requests_added}_{self.request_rate}_{self.temperature}_{self.vllm_config.model_config.model.replace('/','-')}_{self.dataset_name}.csv"
    
    print(f"Saving execution stats to {output_file}")
    self.executions_stats.to_csv(output_file, index=False)
```

**Configuration**:
- Config file: `config.yaml`
- Various arrival rates, temperatures, models, and datasets tested
- [Benchmark runner documentation](https://github.com/inference-sim/pytools/blob/modelingv1/data/README.md)

**Data Collection**:
- Timing data collected for each step() call
- Data stored in pandas DataFrame during execution
- CSV files saved to `./execution_step_data/train/`
- File naming: `execution_stats_{num_requests}_{request_rate}_{temperature}_{model}_{dataset}.csv`

**Commands to Run**:
```bash
python benchmark_runner.py config.yaml --output ./results/
```
- [Benchmark runner documentation](https://github.com/inference-sim/pytools/blob/modelingv1/data/README.md)

## Analysis

**Analysis Performed**:
- Initial time analysis to understand relative costs of each phase
- Statistical analysis of timing distributions
- Analysis done in: `experiments/step_time_analysis.ipynb`

**Key Analysis Code**:
```python
import pandas as pd
from utils import parse_csv, format_data, concatenate_dataframes # found in utils.py

path1 = './profiling/train/'

# # parse and concat all dataframes
df = concatenate_dataframes(path1)
df = format_data(df)

df['total_time'] = df['execute_time'] + df['scheduler_time'] + df['update_time']
df['execute_time_pct'] = (df['execute_time'] / df['total_time'])
df['scheduler_time_pct'] = (df['scheduler_time'] / df['total_time'])
df['update_time_pct'] = (df['update_time'] / df['total_time'])
# average the percentages
avg_execute_time_pct = df['execute_time_pct'].mean()
avg_scheduler_time_pct = df['scheduler_time_pct'].mean()
avg_update_time_pct = df['update_time_pct'].mean()
# print the results
print("\nAverage Percentages of function times over total step time:")
print(f"Execute Time Percentage: {avg_execute_time_pct:.2%}")
print(f"Scheduler Time Percentage: {avg_scheduler_time_pct:.2%}")
print(f"Update Time Percentage: {avg_update_time_pct:.2%}")
```

## Key Takeaways

**Major Findings**:
>Average Percentages of function times over total step time:

>Execute Time Percentage: 95.50%

>Scheduler Time Percentage: 2.70%

>Update Time Percentage: 1.80%

- Execute phase takes 95% of total step() time
- Scheduling overhead is minimal compared to execution
- Update phase is relatively fast post-processing step
- Execute phase is the primary bottleneck for performance optimization

**Implications**:
- Future optimization efforts should focus on the execute phase
- Simulator models should prioritize accurate execution time prediction
- Scheduling and update phases can be modeled with simpler approaches
- This data provides foundation for regression model development

**Future Work**:
- Build regression models using this timing data
- Validate simulator predictions against real vLLM performance
- Investigate execute phase bottlenecks in more detail