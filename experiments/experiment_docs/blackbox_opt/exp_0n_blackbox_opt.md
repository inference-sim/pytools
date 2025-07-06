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
|   iter    |  target   | sum_de... | sum_pr... | max_pr... | num_pr... | sum_de... | sum_de... | sum_de... | sum_de... | sum_pr... | sum_pr... | sum_pr... | max_pr... | max_pr... | num_pr... | intercept | schedu... | update... | queue_... | vllm_o... |
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
| 1         | -5493158. | -.170e-05 | 0.0007203 | -.143e-08 | -.023e-05 | -.467e-05 | -.233e-06 | -.862e-05 | -.455e-05 | -.967e-05 | -.388e-05 | -.191e-05 | -.852e-05 | -.044e-05 | -.781e-05 | 0.0002738 | 501.14025 | 76.692192 | 1011.7379 | 5140.3869 |
| 2         | -219337.5 | -.981e-05 | 0.0008007 | -.682e-05 | -.134e-05 | -.923e-05 | -.763e-05 | -.946e-05 | -.504e-06 | -.905e-06 | -.698e-05 | -.781e-05 | -.834e-06 | -.211e-05 | -.578e-05 | 0.0053316 | 507.56313 | 72.620625 | 1037.3001 | 5834.6256 |
| 3         | -1.36e+07 | -.828e-06 | 0.0007501 | -.888e-05 | -.481e-05 | -.804e-05 | -.892e-05 | -.032e-05 | -.478e-05 | -.085e-05 | -.936e-05 | -.877e-05 | -.300e-05 | -.936e-06 | -.788e-05 | 0.0021162 | 379.66399 | 79.662926 | 910.67250 | 5574.1176 |
| 4         | -9754283. | -.467e-05 | 0.0005893 | -.997e-05 | -.023e-05 | -.140e-05 | -.944e-05 | -.141e-05 | -.995e-06 | -.358e-05 | -.637e-05 | -.148e-05 | -.445e-05 | -.865e-05 | -.034e-05 | 0.0013747 | 341.78290 | 92.295651 | 979.53536 | 5165.3541 |
| 5         | -6051931. | -.275e-05 | 0.0003477 | -.508e-05 | -.259e-05 | -.833e-05 | -.236e-05 | -.509e-05 | -.488e-05 | -.699e-05 | -.958e-05 | -.280e-05 | -.648e-05 | -.634e-05 | -.216e-05 | 0.0011474 | 584.84677 | 77.996485 | 1015.6779 | 5408.1368 |
| 6         | -8995931. | -.864e-05 | 0.0009887 | -.223e-05 | -.372e-05 | -.290e-05 | -.519e-05 | -.259e-05 | -.708e-06 | -.039e-05 | -.682e-05 | -.138e-05 | -.300e-05 | -.586e-05 | -.979e-05 | 0.0099541 | 508.35713 | 71.995924 | 1037.4961 | 5834.6011 |
| 7         | -5090924. | -.108e-05 | 0.0006563 | -.659e-05 | -.472e-05 | -.759e-05 | -.062e-05 | -.640e-05 | -.217e-05 | -.952e-05 | -.700e-05 | -.493e-05 | -.982e-06 | -.389e-05 | -.888e-05 | 0.0028977 | 591.85117 | 99.810976 | 927.53082 | 5723.9532 |
| 8         | -5425130. | -.653e-05 | 0.0001125 | -.048e-06 | -.858e-05 | -.534e-05 | -.490e-05 | -.731e-05 | -.023e-05 | -.458e-05 | -.762e-05 | -.128e-05 | -.906e-05 | -.237e-05 | -.259e-06 | 0.0034783 | 479.32002 | 81.157908 | 940.56098 | 5667.8382 |
| 9         | -7820151. | -.139e-05 | 0.0004112 | -.574e-05 | -.067e-05 | -.016e-05 | -.546e-05 | -.748e-07 | -.447e-05 | -.144e-05 | -.589e-05 | -.783e-05 | -.682e-05 | -.232e-05 | -.566e-05 | 0.0012306 | 442.83747 | 71.174819 | 1067.3665 | 5678.8711 |
| 10        | -2.14e+07 | -.663e-06 | 0.0007539 | -.205e-05 | -.938e-05 | -.254e-05 | -.617e-05 | -.381e-05 | -.883e-05 | -.678e-05 | -.604e-05 | -.188e-05 | -.923e-05 | -.084e-05 | -.923e-06 | 0.0095691 | 541.89091 | 90.021011 | 1050.1775 | 5840.9966 |
| 11        | -4986377. | -.092e-05 | 0.0002529 | -.995e-05 | -.952e-05 | -.568e-05 | -.230e-05 | -.569e-05 | -.962e-05 | -.017e-05 | -.498e-05 | -.962e-05 | -.370e-05 | -.596e-05 | -.708e-05 | 0.0063536 | 510.88728 | 65.874288 | 1078.1118 | 5806.7586 |
| 12        | -1.12e+07 | -.514e-05 | 0.0004739 | -.554e-05 | -.148e-05 | -.027e-05 | -.612e-05 | -.499e-05 | -.276e-05 | -.447e-05 | -.402e-05 | -.206e-05 | -.123e-07 | -.831e-06 | -.096e-05 | 0.0071216 | 414.24621 | 69.819148 | 906.37698 | 5216.3460 |
| 13        | -7252841. | -.595e-06 | 0.0009364 | -.293e-05 | -.407e-06 | -.986e-05 | -.771e-05 | -.854e-05 | -.472e-05 | -.604e-05 | -.089e-05 | -.881e-05 | -.679e-06 | -.042e-05 | -.720e-05 | 0.0086605 | 403.27621 | 79.004046 | 915.04131 | 5355.4611 |
| 14        | -8800726. | -.030e-05 | 0.0003873 | -.445e-05 | -.702e-05 | -.465e-05 | -.914e-05 | -.227e-05 | -.896e-05 | -.861e-05 | -.156e-05 | -.663e-05 | -.374e-05 | -.281e-05 | -.510e-05 | 0.0073707 | 448.54683 | 83.183348 | 930.79416 | 5446.0365 |
| 15        | -3494071. | -.889e-05 | 0.0006945 | -.183e-05 | -.987e-05 | -.136e-05 | -.270e-05 | -.576e-05 | -.695e-05 | -.707e-05 | -.639e-05 | -.242e-05 | -.891e-05 | -.289e-05 | -.621e-05 | 0.0025953 | 326.45832 | 89.952834 | 1098.6858 | 5039.6201 |
| 16        | -8962970. | -.611e-05 | -.330e-06 | -.918e-05 | -.964e-05 | -.602e-05 | -.616e-05 | -.677e-05 | -.319e-05 | -.688e-05 | -.071e-05 | -.695e-05 | -.230e-05 | -.203e-05 | -.157e-05 | 0.0020924 | 351.93576 | 89.855627 | 915.56393 | 5191.2719 |
| 17        | -9448131. | -.947e-05 | 0.0001945 | -.563e-05 | -.399e-05 | -.086e-05 | -.917e-07 | -.108e-05 | -.144e-05 | -.624e-05 | -.440e-05 | -.014e-05 | -.185e-05 | -.834e-05 | -.599e-05 | 0.0070905 | 404.34185 | 95.748904 | 987.57135 | 5732.8789 |
| 18        | -4693196. | -.852e-05 | 0.0006322 | -.459e-05 | -.923e-05 | -.836e-05 | -.907e-05 | -.601e-05 | -.894e-05 | -.714e-05 | -.205e-05 | -.423e-05 | -.879e-06 | -.179e-05 | -.516e-05 | 0.0014253 | 545.35704 | 68.531142 | 1079.0469 | 5189.8153 |
| 19        | -976983.1 | -.144e-05 | 0.0005385 | -.684e-05 | -.974e-05 | -.333e-05 | -.646e-05 | -.961e-06 | -.549e-06 | -.936e-06 | -.293e-05 | -.128e-06 | -.621e-05 | -.928e-05 | -.895e-05 | 0.0068406 | 393.19970 | 62.537735 | 1037.8895 | 5728.4942 |
| 20        | -1401780. | -.753e-05 | 0.0006007 | -.680e-05 | -.737e-05 | -.826e-05 | -.882e-05 | -.837e-05 | -.526e-05 | -.860e-06 | -.366e-05 | -.705e-05 | -.193e-05 | -.537e-05 | -.869e-05 | 0.0050587 | 426.61204 | 66.973922 | 989.99888 | 5409.2413 |
| 21        | -7213647. | -.834e-05 | 0.0008395 | -.250e-05 | -.166e-06 | -.677e-05 | -.980e-05 | -.314e-05 | -.901e-05 | -.150e-05 | -.300e-05 | -.198e-05 | -.750e-05 | -.351e-05 | -.395e-05 | 0.0076326 | 461.68798 | 75.888135 | 975.91480 | 5888.7991 |
| 22        | -8912055. | -.840e-05 | 0.0007815 | -.558e-05 | -.085e-05 | -.371e-05 | -.028e-05 | -.893e-05 | -.988e-05 | -.262e-05 | -.296e-05 | -.024e-06 | -.988e-05 | -.070e-05 | -.739e-05 | 0.0039078 | 412.14767 | 75.252422 | 1064.3219 | 5034.2346 |
| 23        | -1.56e+07 | -.007e-05 | 0.0006727 | -.724e-05 | -.309e-05 | -.663e-05 | -.558e-05 | -.480e-05 | -.499e-05 | -.392e-05 | -.499e-05 | -.670e-05 | -.111e-06 | -.078e-05 | -.189e-05 | 0.0051579 | 376.22612 | 79.968650 | 1032.4218 | 5619.3170 |
...
| 103       | -1.91e+07 | -.133e-05 | 0.0008021 | -.681e-05 | -.757e-06 | -.558e-05 | -.635e-05 | -.056e-05 | -.016e-05 | -.869e-05 | -.721e-05 | -.187e-05 | -.286e-06 | -.186e-05 | -.731e-05 | 0.0034524 | 392.58240 | 67.002727 | 1035.4993 | 5781.6819 |
| 104       | -4991723. | -.652e-05 | 0.0008617 | -.400e-05 | -.999e-05 | -.982e-05 | -.125e-05 | -.088e-05 | -.853e-05 | -.076e-05 | -.102e-05 | -.579e-05 | -.732e-07 | -.408e-06 | -.529e-05 | 0.0085987 | 368.66357 | 96.544941 | 991.99697 | 5477.4934 |
| 105       | -1.60e+07 | -.956e-05 | 0.0009293 | -.891e-05 | -.626e-05 | -.595e-06 | -.332e-05 | -.049e-05 | -.244e-05 | -.365e-05 | -.426e-05 | -.247e-05 | -.951e-05 | -.189e-05 | -.863e-05 | 0.0065725 | 414.39545 | 98.388312 | 1049.7902 | 5628.6872 |
=============================================================================================================================================================================================================================================================
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...
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

