# AccelLayer

## About
AccelLayer is an analytical tool based on the [CHARM][1] analytical model to analyze the latency of Transformer-based layers, targeting on VCK190 platform.

[1]:https://github.com/arc-research-lab/CHARM

## Usage

```bash
git clone https://github.com/arc-research-lab/AccelLayer.git
python3 main.py
```

Type the **sequence length**, **batch size**, **attention head dimension**, **number of attention heads**, and **MLP ratio in FC layers** in the terminal.

```teiminal
Enter sequence length (seq): 192
Enter batch size (batch): 1
Enter head dimension (head_dim): 64
Enter number of heads (heads): 12
Enter MLP ratio (mlp_ratio): 4
```
Then, the script will start to run and print the final results in the terminal.

```teiminal
============================
============================
The latency for each layer is below:
QKV gen MM:              1.049453ms.
QKV transpose layer:     0.221184ms.
Q*K batch MM:            0.625908ms.
Softmax layer:           0.221184ms.
K*V batch MM:            0.815868ms.
Transpose layer:         0.073728ms.
Projection MM:           0.360845ms.
Add layer:               0.147456ms.
LayerNorm layer:         0.073728ms.
FC MM:                   1.393757ms.
GeLU layer:              0.294912ms.
FC MM:                   1.215665ms.
Add layer:               0.147456ms.
LayerNorm layer:         0.073728ms.
============================
Total Transformer Block: 6.714872000000001ms.
============================
============================
```
## Analysis Details

Assume that linear kernels are running at AIE with 1GHz, and non-linear kernels are running at PL with 230MHz, and the datatype is FP32.
### Linear kernels

When users type the Transformer-related parameter (sequence length, batch size, attention head dimension, number of attention heads, and MLP ratio in FC layers), the linear kernel workloads are determined.

For example, if the user chooses the same parameter as above, the linear kernel workload consisting of 6 matrix multiplication (MM) or batched matrix multiplication (BMM) is[^1]:
```terminal
[[ 192  768 2304    1]
 [ 192   64  192   12]
 [ 192  192   64   12]
 [ 192  768  768    1]
 [ 192  768 3072    1]
 [ 192 3072  768    1]]
```
[^1]:The first three elements of each row in the array are the row size of LHS, the column size of LHS, and the column size of RHS. The fourth element of each row in the array is the batch size of BMM.

This tool will input the linear kernel workload to the CHARM one monolithic analytical model and generate the estimated latency based on the most efficient hardware configurations.
### Non-linear kernels

When the linear kernel workloads are determined, the non-linear kernel workloads are also determined[^2]. 

```terminal
Transpose layer0: [192, 2304, 1]
Softmax layer: [192, 192, 12]
Transpose layer1: [192, 768, 1]
Add layer0: [192, 768, 2]
Layernorm layer0: [192, 768, 1]
GeLU layer: [192, 3072, 1]
Add layer1: [192, 768, 2]
Layernorm layer1: [192, 768, 1]
```
[^2]:The first two elements of each row in the array are the row size of the input matrix and the column size of the input matrix. The third element of each row in the array is the batch size of the input matrix.

Assuming that we process the non-linear kernels on PL with the estimated off-chip bandwidth of 8GB/s, we can estimate the latency for each non-linear layer.