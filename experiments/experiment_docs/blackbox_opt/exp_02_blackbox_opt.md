# Experiment XXX: [Blackbox Optimizer Test 2]

**Date**:
**Author**: [Satyam]
**Status**: [Completed]

## Purpose/Goal
Improve optimizer accuracy

## How to Reproduce

Do all the changes in `experiments/experiment_docs/blackbox_opt/exp_01_blackbox_opt.md`


**Code Changes**: 

Exception handling in output parsing

**Configuration**: 

As the optimizer was heading the right way but still struggling to get to a low error I tried bounding the values for the coefficients closer to the baseline params:

```python
pbounds = {
    'sum_decode_tokens': (0, 0.0001),
    'sum_prefill_tokens': (0, 0.00001),
    'max_prefill_tokens': (-0.00001, 0),
    'num_prefills': (0, 0.01),
    'sum_decode_tokenss2': (0, 0.0000001),
    'sum_decode_tokensmsumprefill_tokens': (-0.000001, 0),
    'sum_decode_tokensmmaxprefill_tokens': (0, 0.000001),
    'sum_decode_tokensmnumprefills': (0, 0.0001),
    'sum_prefill_tokenss2': (-0.0000001, 0), 
    'sum_prefill_tokensmmaxprefill_tokens': (0, 0.000001),
    'sum_prefill_tokensmnumprefills': (0, 0.0001),
    'max_prefill_tokenss2': (-0.000001, 0),
    'max_prefill_tokensmnumprefills': (-0.0001, 0),
    'num_prefillss2': (-0.001, 0),
    'intercept': (0, 0.01),
    'schedule_time': (300, 600),
    'update_time': (60, 100),
    'queue_overhead': (900, 1100),
    'vllm_overhead': (5000, 6000),}

    optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds=pbounds,
    random_state=1,
    )

    optimizer.maximize(
        init_points=10,
        n_iter=100,
    )
```


**Commands**: 

Same commands as in `experiments/experiment_docs/blackbox_opt/exp_01_blackbox_opt.md`, just using different pbounds.


## Analysis

For this experiment the best optimizer values:
> Best objective value: -2.37352285278071


## Key Takeaways
**Findings**:

The results are than baseline! Bounding values closer guide the algorithim towards better values.

**Future Work**: Suggested follow-up experiments:

We need to be careful of overfitting so we need to see what happens when the sim is tested across many different trial runs

Need for a feature: We need some kind of exception handling when 