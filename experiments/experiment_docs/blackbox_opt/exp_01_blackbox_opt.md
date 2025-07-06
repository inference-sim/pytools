# Experiment 01: [Blackbox Optimizer Test 1]

**Date**: 07/03/2026

**Author**: Satyam

**Status**: [In Progress]

## Purpose/Goal

Use blackbox optimization to train model coefficients

## How to Reproduce

Pull the inference sim repo and copy file blackbox_opt.ipynb. 



**Code Changes**: 
In vllm:
`benchmark_serving.py`
```python
# After appending ttfts and e2els for each request

    print("TTFTs: [")
    for ttft in ttfts:
        print(f"{ttft}", end=", ")
    print("]")

    print("e2els: [")
    for e2el in e2els:
        print(f"{e2el}", end=", ")
    print("]")
```

We use this later for data collection

In inference-sim:
Add parameter passing in `request_rate_sweep.py` so the optimizer can directly influence the simulator behavior. These arguments are then passed to the sim via the `args_tempelate`
```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Go binary with request rate sweep")
    parser.add_argument(
        "--regression_coeffs",
        type=str,
        default="3.38283913e-05,9.82346868e-06,-3.11237143e-06,1.50291993e-03,4.24173346e-08,-1.06897441e-07,1.92844617e-07,2.60430816e-05,-7.72212201e-09,2.67059068e-08,7.20303280e-06,-1.06904337e-08,-1.05254706e-05,-9.19828725e-04,0.005708624032334771",
    )
    parser.add_argument(
        "--schedule_time",
        type=str,
        default="544",
    )
    parser.add_argument(
        "--update_time",
        type=str,
        default="80",
    )
    parser.add_argument(
        "--queue_overhead_time",
        type=str,
        default="1000",
    )
    parser.add_argument(
        "--vllm_overhead_time",
        type=str,
        default="6000",
    )
    args = parser.parse_args()
```

```python
def run_go_binary(thread_id, arguments):
    result = subprocess.run(
        [GO_BINARY_PATH] + arguments,
        capture_output=True,
        text=True,
        check=True,
        encoding='utf-8'
    )
    print(result.stdout, flush=True) # New line to push results to stdout
```



**Configuration**: 
Config file:
```yaml
  baseline:
    name: "exp1_30rr"
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
      request_rate: 30
      sharegpt_output_len: 0 # Set to 0 for no output length limit
      temperature: 0.0
      seed: 42  # Set a seed for reproducibility
```


Bounds for params used in the optimizer. All bounded to be positive and in a smaller search space towards around what the trained coeffienct were:
```python
# Bounded region of parameter space
pbounds = {
    'sum_decode_tokens': (0, 0.0001),
    'sum_prefill_tokens': (0, 0.001),
    'max_prefill_tokens': (0, 0.0001),
    'num_prefills': (0, 0.0001),
    'sum_decode_tokenss2': (0, 0.0001),
    'sum_decode_tokensmsumprefill_tokens': (0, 0.0001),
    'sum_decode_tokensmmaxprefill_tokens': (0, 0.0001),
    'sum_decode_tokensmnumprefills': (0, 0.0001),
    'sum_prefill_tokenss2': (0, 0.0001), 
    'sum_prefill_tokensmmaxprefill_tokens': (0, 0.0001),
    'sum_prefill_tokensmnumprefills': (0, 0.0001),
    'max_prefill_tokenss2': (0, 0.0001),
    'max_prefill_tokensmnumprefills': (0, 0.0001),
    'num_prefillss2': (0, 0.0001),
    'intercept': (0, 0.01),
    'schedule_time': (300, 600),
    'update_time': (60, 100),
    'queue_overhead': (900, 1100),
    'vllm_overhead': (5000, 6000),}
```

**Data Collection**: 
Use the config set up we collected on set of vllm ttft and e2e data for a run and saved in the `blackbox_opt.ipynb`
```python
# For now use constant values for vllm_ttfts and vllm_e2es

vllm_ttfts = [
0.012435252778232098, 0.009395882952958345, 0.012567796278744936, 0.015322412829846144, 0.011697649955749512, 0.013088695239275694, 0.014777038246393204, 0.012686633039265871, 0.02117804205045104, 0.021836630068719387, 0.016655396670103073, 0.015868980903178453, 0.02295303111895919, 0.023163228295743465, 0.031361973844468594, 0.024341952987015247, 0.026004493236541748, 0.021002538967877626, 0.0219248509965837, 0.018022753298282623, 0.014443457592278719, 0.02480849390849471, 0.019598938059061766, 0.016469583846628666, 0.01523833628743887, 0.017173814121633768, 0.016928269062191248, 0.02183521818369627, 0.0206878911703825, 0.05854615522548556, 0.05681315064430237, 0.04987847385928035, 0.043255149852484465, 0.0396235678344965, 0.019120340701192617, 0.02889071498066187, 0.022600724827498198, 0.02234139060601592, 0.018349184654653072, 0.0186814502812922, 0.015227964147925377, 0.018818443175405264, 0.029649354983121157, 0.027635287959128618, 0.01853758003562689, 0.01639375602826476, 0.021319673862308264, 0.016914449632167816, 0.018753784243017435, 0.015995902940630913, 0.018240667413920164, 0.02017358411103487, 0.021751046180725098, 0.02768043288961053, 0.01951850112527609, 0.015044958796352148, 0.02111377427354455, 0.01786909718066454, 0.02102688606828451, 0.019092899281531572, 0.024889993015676737, 0.018621352035552263, 0.016848224215209484, 0.018639008980244398, 0.0273125059902668, 0.03337852796539664, 0.016743354965001345, 0.02345914114266634, 0.021683991886675358, 0.019324283115565777, 0.01994457235559821, 0.02036265330389142, 0.02135285222902894, 0.02100067026913166, 0.027160054072737694, 0.019208349753171206, 0.0175353791564703, 0.032339252065867186, 0.030020272824913263, 0.02411013375967741, 0.019037755206227303, 0.018443888053297997, 0.018910886719822884, 0.030474115163087845, 0.027349235955625772, 0.029933215584605932, 0.02698174398392439, 0.020943788345903158, 0.021292726043611765, 0.02480420796200633, 0.021610640920698643, 0.016870138701051474, 0.017710881773382425, 0.021240055095404387, 0.04371336288750172, 0.02232355112209916, 0.017495656851679087, 0.027022548019886017, 0.025941645726561546, 0.024313100147992373, ]

vllm_e2es = [
0.01243477500975132, 1.728445002809167, 6.293655326124281, 3.36385326879099, 3.0681904926896095, 3.7817438333295286, 0.8536133351735771, 1.6802788530476391, 1.9696030509658158, 0.021836436819285154, 0.26861430890858173, 3.4804028598591685, 0.022952823899686337, 1.8848898652940989, 2.9168300488963723, 0.02434179186820984, 0.026004310231655836, 2.7906856359913945, 0.021924633998423815, 0.018022576346993446, 3.086558997631073, 0.02480829507112503, 0.019598791375756264, 0.016469409223645926, 3.4232060741633177, 0.43898611329495907, 0.016928064171224833, 1.0565687902271748, 3.1023759059607983, 0.2928661871701479, 1.5161170749925077, 0.049878283869475126, 3.5931831360794604, 0.0396233880892396, 2.372470040805638, 0.12193845817819238, 0.02260051667690277, 1.3731577936559916, 0.44265588093549013, 2.623970297165215, 0.015227763913571835, 6.346380535047501, 0.029649184085428715, 0.868159556761384, 2.1411039540544152, 4.745520658791065, 0.02131947223097086, 4.096554935909808, 0.01875356724485755, 0.31218736805021763, 3.2590340822935104, 6.188668855000287, 0.02175083989277482, 0.027680260129272938, 0.39204079005867243, 0.5059276237152517, 1.5098992669954896, 3.8954369290731847, 0.7591470442712307, 1.0208364240825176, 0.02488980581983924, 0.01862116204574704, 3.903964619152248, 4.660428964998573, 0.027312325779348612, 0.03337835008278489, 3.5112321451306343, 0.023458980955183506, 3.4664018931798637, 1.2706626802682877, 0.019944400060921907, 1.9233045224100351, 2.1120217321440578, 1.2805547988973558, 0.027159880846738815, 1.7143783019855618, 3.126032245811075, 0.03233905183151364, 0.03002011589705944, 5.607961312867701, 3.09012442920357, 5.058007639367133, 2.2598047726787627, 4.198010358959436, 3.9562214836478233, 0.029933073557913303, 1.6581682157702744, 0.020943609066307545, 3.6020733090117574, 5.869788001757115, 0.02161044580861926, 0.31265699677169323, 1.0484589696861804, 0.21515931421890855, 2.9344231858849525, 0.16157546220347285, 1.1131132780574262, 0.02702236780896783, 0.02594144968315959, 0.02431293996050954, ]

```

We use this to then compare our optimizer results to our baseline.




**Commands**: Step-by-step commands to reproduce

1. Run vllm and from the output files parse the vllm ttfts and e2es (from above or any desired run), load as np.arrays.

```bash
~$ python benchmark_runner.py config.yaml --output ./results/blackbox # saved to results/blackbox/exp1_30rr_30_100_0.0_...
```

In `blackbox_opt.ipynb`:

```python
vllm_ttfts = np.array(vllm_ttfts)
vllm_e2es = np.array(vllm_e2es)
```

2. Ensure based on the run used for vLLM generate the right arrival data and edit files according to this [pull request](https://github.com/inference-sim/inference-sim/pull/44) for the simulator.

3. The optimizer function calls helper functions to run the sim, parse outputs, and compute the error. Use them to run the optimizer function
```python
def black_box_function(sum_decode_tokens: float, sum_prefill_tokens: float, max_prefill_tokens: float, num_prefills: float, sum_decode_tokenss2: float, sum_decode_tokensmsumprefill_tokens: float, sum_decode_tokensmmaxprefill_tokens: float, sum_decode_tokensmnumprefills: float, sum_prefill_tokenss2: float, sum_prefill_tokensmmaxprefill_tokens: float, sum_prefill_tokensmnumprefills: float, max_prefill_tokenss2: float, max_prefill_tokensmnumprefills: float, num_prefillss2: float, intercept: float, schedule_time: int, update_time: int, queue_overhead: int, vllm_overhead: int):
    
    # run a python subprocess to execute the vllm command

    coefficients = [sum_decode_tokens, sum_prefill_tokens, max_prefill_tokens, num_prefills, sum_decode_tokenss2, sum_decode_tokensmsumprefill_tokens, sum_decode_tokensmmaxprefill_tokens, sum_decode_tokensmnumprefills, sum_prefill_tokenss2, sum_prefill_tokensmmaxprefill_tokens, sum_prefill_tokensmnumprefills, max_prefill_tokenss2, max_prefill_tokensmnumprefills, num_prefillss2, intercept]
    coefficients_str = ','.join(map(str, coefficients))

    result = run_command(["python","request_rate_sweep.py", 
            "--regression_coeffs", f'{coefficients_str}',
            "--schedule_time", f"{str(int(schedule_time))}",
            "--update_time", f"{str(int(update_time))}",
            "--queue_overhead_time", f"{str(int(queue_overhead))}",
            "--vllm_overhead_time", f"{str(int(vllm_overhead))}",])
    
    sim_values = parse_output(result)
    error = get_error(sim_values, vllm_ttfts, vllm_e2es)
    
    return -error # negative as we want to maximize (go towards 0)
```


Verify that it works by using our vllm data trained (baseline) coefficients.

```python
# Test on current best parameters
black_box_function(
    sum_decode_tokens=3.38283913e-05,
    sum_prefill_tokens=9.82346868e-06,
    max_prefill_tokens=-3.11237143e-06,
    num_prefills=1.50291993e-03,
    sum_decode_tokenss2=4.24173346e-08,
    sum_decode_tokensmsumprefill_tokens=-1.06897441e-07,
    sum_decode_tokensmmaxprefill_tokens=1.92844617e-07,
    sum_decode_tokensmnumprefills=2.60430816e-05,
    sum_prefill_tokenss2=-7.72212201e-09,
    sum_prefill_tokensmmaxprefill_tokens=2.67059068e-08,
    sum_prefill_tokensmnumprefills=7.20303280e-06,
    max_prefill_tokenss2=-1.06904337e-08,
    max_prefill_tokensmnumprefills=-1.05254706e-05,
    num_prefillss2=-9.19828725e-04,
    intercept=0.005708624032334771,
    schedule_time=544,
    update_time=80,
    queue_overhead=1000,
    vllm_overhead=6000
)
```

4. Train optimizer:

```python
optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds=pbounds,
    random_state=1,
)

optimizer.maximize(
    n_iter=100,
)
```

Once trained you can view the best params and error:

```python
# Get the best parameters
best_params = optimizer.max['params']
best_value = optimizer.max['target']

print("Best parameters found:")
for param, value in best_params.items():
    print(f"  {param}: {value}")

print(f"\nBest objective value: {best_value}")

# You can also see all trials
print("\nAll trials:")
for i, res in enumerate(optimizer.res):
    print(f"Trial {i+1}:")
    print(f"  Parameters: {res['params']}")
    print(f"  Target: {res['target']}")
```

## Analysis

Data from the baseline:

> Error: -3.4914122879967424

From optimizer training:

> Best objective value: -204321.9118010524

We are still quite off from our initial error but the parameters are going in the desired direction.


```
Best parameters found:
  sum_decode_tokens: 4.912687373218388e-05
  sum_prefill_tokens: 0.0005074387994500158
  max_prefill_tokens: 9.001857382002145e-05
  num_prefills: 8.264378462907746e-05
  sum_decode_tokenss2: 4.429253622347732e-05
  sum_decode_tokensmsumprefill_tokens: 7.063317372409196e-05
  sum_decode_tokensmmaxprefill_tokens: 7.578569012536311e-05
  sum_decode_tokensmnumprefills: 1.808420657174411e-05
  sum_prefill_tokenss2: 1.4635303393752965e-06
  sum_prefill_tokensmmaxprefill_tokens: 1.0631700972456038e-05
  sum_prefill_tokensmnumprefills: 5.9617950228572905e-05
  max_prefill_tokenss2: 5.370528189381161e-05
  max_prefill_tokensmnumprefills: 1.0204010036542932e-05
  num_prefillss2: 2.8561187157070857e-05
  intercept: 0.0009894275931819007
  schedule_time: 426.07523968439835
  update_time: 86.83738022300543
  queue_overhead: 988.4423346912623
  vllm_overhead: 5590.818515077307
```



## Key Takeaways
**Findings**: 

Framework for training the optimizer is ready, we can iterate now to improve training.

Overall I think this shows promise the blackbox optimizer strategy.

**Future Work**: 

Suggested follow-up experiments:

* Using bounds centered around the baseline coefficients we have
* Tuning the error function
* Increase initial training rounds
* Remove the bounds to allow values to be less than zero

