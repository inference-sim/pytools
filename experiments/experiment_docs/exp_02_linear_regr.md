# Experiment 02: [Title]

**Date**: 

**Author**: Satyam

**Status**: [Completed]

## Purpose/Goal
Try to predict execute model via some input features collected from benchmarking

## How to Reproduce
**Code Changes**: 

Add all profiling changes made in `experiment_docs/exp_01_data_collection.md`

In 
```python
# add to init
        self.executions_stats = pd.DataFrame(columns=["num_scheduled_tokens", "num_total_scheduled_tokens", "scheduled_new_reqs", 
                                                      "loop_time", "loop_step_time", "loop_queue_time", "execute_time", "scheduler_time", "update_time", 
                                                      "model", "block_size", "gpu_memory_utilization", "num_gpu_blocks", "enable_prefix_caching", "max_num_sequences", "max_model_len", 
                                                      "temperature", "arrival_rate", "distribution", "dataset", "time_stamp"])
```


**Configuration**: Config files used, commit references
**Data Collection**: What data was collected and where it's stored
**Commands**: Step-by-step commands to reproduce

## Analysis
**Analysis Performed**: What analysis was done
**Code/Files**: Reference to notebooks or analysis files used

## Key Takeaways
**Findings**: Main discoveries and quantitative results
**Implications**: What these results mean for the project
**Future Work**: Suggested follow-up experiments