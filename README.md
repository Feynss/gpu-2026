# Content
- [How To](#how-to)
- [Configuration](#configuration)
- [Time Measurement](#time-measurement)
- [Tasks](#tasks)
- [Results](#results)

# How To
1. Create [github](https://github.com/) account (if not exists);
2. Make sure SSH clone & commit is working ([Connecting to GitHub with SSH](https://docs.github.com/en/authentication/connecting-to-github-with-ssh));
3. Fork this repo (just click **Fork** button on the top of the page, detailed instructions [here](https://docs.github.com/en/get-started/exploring-projects-on-github/contributing-to-a-project))
4. Clone your forked repo into your local machine, use your user instead of `username`:
```sh
git clone git@github.com:username/gpu-2026.git
cd gpu-2026
```
5. Go to your group folder, e.g.:
```sh
cd 3823B1FI1
```
6. Go to needed task folder, e.g.:
```sh
cd 1_gelu_omp
```
7. Create new folder with your surname and name (**make sure it's the same for all tasks**), e.g.:
```sh
mkdir petrov_ivan
```
8. Copy your task source/header files (including main program) into this folder (use `copy` instead of `cp` on Windows), e.g.:
```sh
cd petrov_ivan
cp /home/usr/lab/*.cpp .
cp /home/usr/lab/*.h .
```
8. Push your sources to github repo, e.g.:
```sh
cd ..
git add .
git commit -m "1_gelu_omp task"
git push
```
9. Go to your repo in browser, click **Contribute** button on the top of page, then **Open pull request**. Provide meaningfull request title and description, then **Create pull request** (see details [here](https://docs.github.com/en/get-started/exploring-projects-on-github/contributing-to-a-project)).
10. Go to Pull Requests [page](https://github.com/avgorshk/gpu-2025/pulls) in course repo, find your pull request and check if there are no any merge conflicts occur. If merge conflicts happen - resolve it following the instruction provided by github.

# Time Measurement
The following scheme is used to measure task execution time:
```cpp
int main() {
    // ...

    // Warming-up
    Task(input, size);

    // Performance Measuring
    std::vector<double> time_list;
    for (int i = 0; i < 4; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        Task(input, size);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        time_list.push_back(duration.count());
    }
    double time = *std::min_element(time_list.begin(), time_list.end());

    // ...
}
```

# Configuration
- CPU: Intel Core i5 12600K (4 cores, 4 threads)
- RAM: 16 GB
- GPU: NVIDIA RTX 4060 (8 GB)
- OS:  Ubuntu 22.04.3 LTS
- Host Compiler: GCC 11.4.0 (C++17)
- CUDA: 13.3

# Tasks
## Task #1: OpenMP GELU Implementation
The **Gaussian Error Linear Unit (GELU)** is an activation function frequently used in Deep Neural Networks (DNNs) and can be thought of as a smoother ReLU.

To approximate GELU function, use the following formula:

GELU(x) =  $0.5x(1 + tanh(\sqrt{2 / \pi}(x + 0.044715 * x^3)))$

Implement the function with the following interface in C++:
```cpp
std::vector<float> GeluOMP(const std::vector<float>& input);
```
Size of result vector should be the same as for `input`. Use OpenMP technology to make your function parallel & fast.

Two files are expected to be uploaded:
- gelu_omp.h
```cpp
#ifndef __GELU_OMP_H
#define __GELU_OMP_H

#include <vector>

std::vector<float> GeluOMP(const std::vector<float>& input);

#endif // __GELU_OMP_H
```
- gelu_omp.cpp
```cpp
#include "gelu_omp.h"

std::vector<float> GeluOMP(const std::vector<float>& input) {
    // Place your implementation here
}
```
**Performance Hints:**
 - better formula to compute GELU, e.g. replace *tanh()* with *exp()*;
 - loop unrolling;
 - loop vectorization;
 - vector allocation and computations in different threads *(Windows only)*.

## Task #2: CUDA GELU Implementation
Implement the function with the following interface in CUDA C++ using the formula described above:
```cpp
std::vector<float> GeluCUDA(const std::vector<float>& input);
```
Size of result vector should be the same as for `input`. Use CUDA technology to make your function work on NVIDIA GPU. Try to make it fast.

Two files are expected to be uploaded:
- gelu_cuda.h
```cpp
#ifndef __GELU_CUDA_H
#define __GELU_CUDA_H

#include <vector>

std::vector<float> GeluCUDA(const std::vector<float>& input);

#endif // __GELU_CUDA_H
```
- gelu_cuda.cu
```cpp
#include "gelu_cuda.h"

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    // Place your implementation here
}
```
**Performance Hints:**
 - overlap host memory allocation and CUDA computations;
 - allocate and free device memory once;
 - use better formula to compute GELU, e.g. replace *tanh()* with *exp()*.

## Task #3: Naive Matrix Multiplication using CUDA
General matrix multiplication (GEMM) is a very basic and broadly used linear algebra operation applied in high performance computing (HPC), statistics, deep learning and other domains. There are a lot of GEMM algorithms with different mathematical complexity form $O(n^3)$ for naive and block approaches to $O(n^{2.371552})$ for the method descibed by Williams et al. in 2024 [[1](https://epubs.siam.org/doi/10.1137/1.9781611977912.134)]. But despite a variety of algorithms with low complexity, block matrix multiplication remains the most used implementation in practice since it fits to modern HW better.

To start learning matrix multiplication smoother, let us start with naive approach here. To compute matrix multiplication result C for matricies A and B, where C = A * B and the size for all matricies are $n*n$, one should use the following formula for each element of C (will consider only square matricies for simplicity):

$c_{ij}=\sum_{k=1}^na_{ik}b_{kj}$

In this task one should implement naive approach for matrix multiplication in CUDA trying to make it fast enough *(pay attention to global memory accesses in your code)*.

Each matrix must be stored in a linear array by rows, so that `a.size()==n*n`. Function takes two matricies and their size as inputs, and returns result matrix also stored by rows.

For simplicity, let's consider matrix size is always power of 2.

Two files are expected to be uploaded:
- naive_gemm_cuda.h:
```cpp
#ifndef __NAIVE_GEMM_CUDA_H
#define __NAIVE_GEMM_CUDA_H

#include <vector>

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n);

#endif // __NAIVE_GEMM_CUDA_H
```
- naive_gemm_cuda.cu:
```cpp
#include "naive_gemm_cuda.h"

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    // Place your implementation here
}
```
**Performance Hints:**
 - warp-friendly memory accesses;
 - multiple elements per warp processing;
 - loop unrolling and memory load vectorization;
 - block size selection;
 - overlap host memory allocation and CUDA computations.

## Task #4: Block Matrix Multiplication using CUDA
In real applications block-based approach for matrix multiplication can get multiple times faster execution comparing with naive version due to cache friendly approach. To prove this in practice, implement such a version in C++ using OpenMP.

In block version algorithm could be divided into three stages:
1. Split matricies into blocks (block size normally affects performance significantly so choose it consciously);
2. Multiply two blocks to get partial result;
3. Replay step 2 for all row/column blocks accumulating values into a single result block.

From math perspective, block matrix multiplication could be described by the following formula, where $C_{IJ}$, $A_{IK}$ and $B_{KJ}$ are sub-matricies with the size $block\_size*block\_size$:

$C_{IJ}=\sum_{k=1}^{block_count}A_{IK}B_{KJ}$

Each matrix must be stored in a linear array by rows, so that `a.size()==n*n`. Function takes two matricies and their size as inputs, and returns result matrix also stored by rows.

In CUDA C++ block-based approach looks similar. But to get better performance one should use CUDA shared memory to store each particular block while computations. With this consideration, algorithm will be the following:
1. A single CUDA block should compute a single block of result matrix C, a single CUDA thread - a single matrix C element;
2. For each A block in a row and B block in a column:
    1. Load A block into shared memory;
    2. Load B block into shared memory;
    3. Synchronize over all threads in block;
    4. Compute BlockA * BlockB and accumulate into C block in shared memory;
    5. Synchronize over all threads in block;
3. Dump block C from shared to global memory.

For simplicity, let's consider matrix size is always power of 2.

Two files are expected to be uploaded:
- block_gemm_cuda.h:
```cpp
#ifndef __BLOCK_GEMM_CUDA_H
#define __BLOCK_GEMM_CUDA_H

#include <vector>

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n);

#endif // __BLOCK_GEMM_CUDA_H
```
- block_gemm_cuda.cu:
```cpp
#include "block_gemm_cuda.h"

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    // Place your implementation here
}
```
**Performance Hints:**
 - shared memory usage to store matrix block;
 - warp-friendly memory accesses;
 - multiple elements per warp processing;
 - loop unrolling and memory load vectorization;
 - block size selection;
 - overlap host memory allocation and CUDA computations.

## Task #5: Matrix Multiplication using cuBLAS
The most performant way to multiply two matrices on particular hardware is to use vendor-provided library for this purpose. In CUDA it's [cuBLAS](https://docs.nvidia.com/cuda/cublas/index.html). Try to use cuBLAS API to implement general matrix multiplication in most performant way.

Each matrix must be stored in a linear array by rows, so that `a.size()==n*n`. Function takes two matricies and their size as inputs, and returns result matrix also stored by rows.

For simplicity, let's consider matrix size is always power of 2.

Note, that in cuBLAS API matrix is expected to be stored by columns, so additional transpose may be required.

Two files are expected to be uploaded:
- gemm_cublas.h:
```cpp
#ifndef __GEMM_CUBLAS_H
#define __GEMM_CUBLAS_H

#include <vector>

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n);

#endif // __GEMM_CUBLAS_H
```
- gemm_cublas.cu:
```cpp
#include "gemm_cublas.h"

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    // Place your implementation here
}
```
**Performance Hints:**
 - overlap host memory allocation and CUDA computations;
 - avoid redundant device memory allocation.

## Task #6: CUDA Softmax Implementation
The **softmax** function is a fundamental operation in machine learning, often used to convert a vector of raw scores into a probability distribution. For an input vector $x$ of length $N$, the softmax is defined element-wise as:

Softmax(x) = $e^{x_i}/(\sum_{j=1}^ne^{x_j})$ for $i=1,..,N$

When the input is a matrix, softmax is applied independently to each row.

To make the computation numerically stable in floating-point arithmetic, the following equivalent formula is used in practice:

Softmax(x) = $e^{(x_i-row\_max)}/(\sum_{j=1}^ne^{(x_j-row\_max)})$ for $i=1,..,N$

Here $row\_max$ is $max(x_i)$ for $i=1,..,N$, normally computed independently for each row in matrix.

Implement the function with the following interface in C++ using CUDA:
```cpp
std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count);
```
Note the following:
- the parameter input holds the matrix elements in row‑major order (all elements of row 0, then row 1, etc.);
- the number of rows is given by `row_count`;
- the number of columns (size of each row) can be derived as `row_size = input.size() / row_count` (it is guaranteed that `input.size()` is divisible by row_count);
- the function must compute softmax for each row independently and return a vector of the same size containing the row‑wise softmax results.

Use CUDA to parallelize the computation. The implementation should be efficient – consider using shared memory for per‑row reductions and exponentiations.

For simplicity, let's consider matrix sizes are always power of 2.

Two files are expected to be uploaded:
- softmax_cuda.h:
```cpp
#ifndef SOFTMAX_CUDA_H
#define SOFTMAX_CUDA_H

#include <vector>

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count);

#endif // SOFTMAX_CUDA_H
```
- softmax_cuda.cu:
```cpp
#include "softmax_cuda.h"

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count) {
    // Place your implementation here
}
```
**Performance Hints:**
 - overlap host memory allocation and CUDA computations;
 - use registers and/or shared memory to cache input values.

## Task #7: Layer Norm Implementation in PyCUDA
Layer Normalization (**LayerNorm**) is a widely used technique in deep learning that normalizes activations across the feature dimension for each sample independently. For an input vector x of length N (the features of one sample), LayerNorm is defined as:

$$x'_i=(x_i-\mu)/\sqrt{\sigma^2+\epsilon}$$
$$y_i=\gamma_ix'_i+\beta_i$$

where:
- $\mu=1/N*\sum_{j=1}^Nx_j$ is the mean of the features;
- $\sigma^2=1/N*\sum_{j=1}^N(x_j-\mu)^2$ is the variance;
- $\epsilon$ is a small constant for numerical stability (e.g. $10^-5$);
- $\gamma$ and $\beta$ are learnable parameters (vectors of length N) that scale and shift the normalized output.

When the input is a matrix (batch of samples), LayerNorm is applied independently to each row.

To complete the task, one have to implement the following function in PyCUDA, the only file is expected to be upload:
- layernorm_pycuda.py
```py
import numpy as np

def layernorm_pycuda(input, gamma, beta, row_size, eps=1e-5):
    """
    Apply Layer Normalization to each row of the input matrix.

    Parameters
    ----------
    input : list or numpy.ndarray of float
        Flattened matrix in row‑major order. Its length must be divisible by row_size.
    gamma : list or numpy.ndarray of float
        Scale parameter, length = row_size.
    beta : list or numpy.ndarray of float
        Shift parameter, length = row_size.
    row_size : int
        Number of features per row (i.e., number of columns).
    eps : float, optional
        Small constant for numerical stability.

    Returns
    -------
    numpy.ndarray
        Flattened matrix of the same shape as input, containing the row‑wise
        normalized results.
    """
    # TODO: Implement using PyCUDA
    pass
```

For simplicity, let's consider `row_size` is power of 2. Target data type is float32.
One may use numba or C strings to write CUDA kernels.

## Task #8: OpenCL GELU Implementation
Implement GELU function with the following interface in OpenCL using the formula described in task #1:
```cpp
std::vector<float> GeluOCL(const std::vector<float>& input, int platform);
```
Size of result vector should be the same as for `input`. Use OpenCL technology to make your function work on NVIDIA GPU. Try to make it fast.

Use `CL_DEVICE_GPU` flag to choose GPU device. Use `platform` platform and `0` device. Store your OpenCL kernel in a string constant.

Two files are expected to be uploaded:
- gelu_ocl.h
```cpp
#ifndef __GELU_OCL_H
#define __GELU_OCL_H

#include <vector>

std::vector<float> GeluOCL(const std::vector<float>& input, int platform);

#endif // __GELU_OCL_H
```
- gelu_ocl.cpp
```cpp
#include "gelu_ocl.h"

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    // Place your implementation here
}
```
**Performance Hints:**
 - perform OpenCL boilerplate code once;
 - use better formula to compute GELU, e.g. replace *tanh()* with *exp()*;
 - overlap host memory allocation and GPU computations.

# Results
## 1_gelu_omp (134217728 elements)
|Group|Name|Result|Rank|
|-----|----|------|----|
|**FAST**|**FAST**|**0.1741**|**-**|
|3823B1PE1|rusakova_aleksandra|0.2258|3|
|3823B1PE1|tsibareva_ekaterina|0.2378|4|
|3823B1PE3|dergachev_arseniy|0.2386|1|
|3823B1PE2|kolotukhin_alexander|0.2394|3|
|3823B1PE3|batkov_filipp|0.2535|2|
|3823B1PE2|sinev_artem|0.2567|1|
|3823B1PE2|klimenko_vladislav|0.2611|2|
|3823B1PE1|redkina_alina|0.2634|2|
|3823B1PE2|zorin_danila_artemovich|0.4702|4|
|3823B1PE1|morozov_nikita|0.4711|1|
|**REF**|**REF**|**0.5440**|**-**|

## 2_gelu_cuda (134217728 elements)
|Group|Name|Result|Rank|
|-----|----|------|----|
|**FAST**|**FAST**|**0.2067**|**-**|
|3823B1PE3|dergachev_arseniy|0.2192|1|
|3823B1PE1|redkina_alina|0.2390|1|
|3823B1PE2|sinev_artem|0.2406|1|
|3823B1PE1|rusakova_aleksandra|0.2437|3|
|3823B1PE2|zorin_danila_artemovich|0.2690|2|
|**REF**|**REF**|**0.2717**|**-**|
|3823B1PE2|klimenko_vladislav|0.2741|3|
|3823B1PE3|batkov_filipp|0.2750|2|
|3823B1PE1|tsibareva_ekaterina|0.2788|2|

## 3_naive_gemm_cuda (4096 elements)
|Group|Name|Result|Rank|
|-----|----|------|----|
|3823B1PE1|redkina_alina|0.0712|1|
|**FAST**|**FAST**|**0.0732**|**-**|
|3823B1PE1|rusakova_aleksandra|0.0927|2|
|3823B1PE2|klimenko_vladislav|0.1138|1|
|**REF**|**REF**|**0.5864**|**-**|

## 4_block_gemm_cuda (4096 elements)
|Group|Name|Result|Rank|
|-----|----|------|----|
|3823B1PE1|redkina_alina|0.0631|1|
|**FAST**|**FAST**|**0.0701**|**-**|
|3823B1PE1|rusakova_aleksandra|0.1805|2|
|**REF**|**REF**|**0.3076**|**-**|

## 5_gemm_cublas (4096 elements)
|Group|Name|Result|Rank|
|-----|----|------|----|
|3823B1PE1|redkina_alina|0.0359|1|
|**FAST**|**FAST**|**0.0417**|**-**|
|**REF**|**REF**|**0.0591**|**-**|

## 6_softmax_cuda (8192x16384 elements)
|Group|Name|Result|Rank|
|-----|----|------|----|
|**FAST**|**FAST**|**0.2142**|**-**|
|**REF**|**REF**|**0.2707**|**-**|

## 7_layernorm_pycuda (8192x16384 elements)
|Group|Name|Result|Rank|
|-----|----|------|----|
|**REF**|**REF**|**0.1740**|**-**|

## 8_gelu_ocl (134217728 elements)
|Group|Name|Result|Rank|
|-----|----|------|----|
|**FAST**|**FAST**|**0.2036**|**-**|
|**REF**|**REF**|**0.3289**|**-**|

# Tasks Done
## 3823B1PE1
|Group|Name|Passed|Score|
|-----|----|------|-----|
|3823B1PE1|morozov_nikita|1/8|61|
|3823B1PE1|redkina_alina|5/8|317|
|3823B1PE1|rusakova_aleksandra|4/8|247|
|3823B1PE1|tsibareva_ekaterina|2/8|121|

Passed: 0

## 3823B1PE2
|Group|Name|Passed|Score|
|-----|----|------|-----|
|3823B1PE2|klimenko_vladislav|3/8|185|
|3823B1PE2|kolotukhin_alexander|1/8|62|
|3823B1PE2|sinev_artem|2/8|127|
|3823B1PE2|zorin_danila_artemovich|2/8|120|

Passed: 0

## 3823B1PE3
|Group|Name|Passed|Score|
|-----|----|------|-----|
|3823B1PE3|batkov_filipp|2/8|124|
|3823B1PE3|dergachev_arseniy|2/8|128|

Passed: 0

**Total Passed: 0**

---
*Maximum Score: 512 (64 per task)*
