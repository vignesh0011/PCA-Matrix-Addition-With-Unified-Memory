# PCA-Matrix-Addition-With-Unified-Memory

## Aim:
To perform Matrix addition with unified memory and check its performance with nvprof.

## Procedure:
Step 1 :
Include the required files and library.

Step 2 :
Introduce a function named "initialData","sumMatrixOnHost","checkResult" to return the initialize the data , perform matrix summation on the host and then check the result.

Step 3 :
Create a grid 2D block 2D global function to perform matrix on the gpu.

Step 4 :
Declare the main function. In the main function set up the device & data size of matrix , perform memory allocation on host memory & initialize the data at host side then add matrix at host side for result checks followed by invoking kernel at host side. Then warm-up kernel,check the kernel error, and check device for results.Finally free the device global memory and reset device.

Step 5 :
Execute the program and run the terminal . Check the performance using nvprof.

# Program:
``` c
#include <stdio.h>
#include <cuda.h>

__global__ void cudaAdd(int* a, int* b, int* c, const int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    srand(time(0));
    int a[100], b[100], c[100];

    for (int i = 0; i < 100; i++) {
        a[i] = rand() * 1000;
        b[i] = rand() * 1000;
    }

    int* d_a, * d_b, * d_c;
    cudaMalloc(&d_a, sizeof(int) * 100);
    cudaMalloc(&d_b, sizeof(int) * 100);
    cudaMalloc(&d_c, sizeof(int) * 100);

    cudaMemcpy(d_a, a, sizeof(int) * 100, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(int) * 100, cudaMemcpyHostToDevice);

    cudaMemset(d_c, 0, sizeof(int) * 100);

    int iLen = 256;
    dim3 block(iLen);
    dim3 grid((100 + block.x - 1) / block.x);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);

    cudaAdd << <grid, block >> > (d_a, d_b, d_c, 100);

    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float elapsed;
    cudaEventElapsedTime(&elapsed, start, end);

    cudaMemcpy(c, d_c, sizeof(int) * 100, cudaMemcpyDeviceToHost);

    printf("The kernel ran for %.2f milliseconds.\n", elapsed);
    for (int i = 0; i < 100; i++) {
        printf("%d ", c[i]);
    }
    printf("\n");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    return 0;
}

```

without memset:-
``` c
#include <stdio.h>
#include <cuda.h>

__global__ void cudaAdd(int* a, int* b, int* c, const int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    srand(time(0));
    int a[100], b[100], c[100];

    for (int i = 0; i < 100; i++) {
        a[i] = rand() * 1000;
        b[i] = rand() * 1000;
    }

    int* d_a, * d_b, * d_c;
    cudaMalloc(&d_a, sizeof(int) * 100);
    cudaMalloc(&d_b, sizeof(int) * 100);
    cudaMalloc(&d_c, sizeof(int) * 100);

    cudaMemcpy(d_a, a, sizeof(int) * 100, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(int) * 100, cudaMemcpyHostToDevice);

    cudaMemset(d_c, 0, sizeof(int) * 100);

    int iLen = 256;
    dim3 block(iLen);
    dim3 grid((100 + block.x - 1) / block.x);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);

    cudaAdd << <grid, block >> > (d_a, d_b, d_c, 100);

    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float elapsed;
    cudaEventElapsedTime(&elapsed, start, end);

    cudaMemcpy(c, d_c, sizeof(int) * 100, cudaMemcpyDeviceToHost);

    printf("The kernel ran for %.2f milliseconds.\n", elapsed);
    for (int i = 0; i < 100; i++) {
        printf("%d ", c[i]);
    }
    printf("\n");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    return 0;
}
```
## Output:
### With Memset:
![image](https://github.com/curiouzs/PCA-Matrix-Addition-With-Unified-Memory/assets/75234646/d7367215-4a49-4d18-b3a6-64062a6465cc)

### Without memset:
![image](https://github.com/curiouzs/PCA-Matrix-Addition-With-Unified-Memory/assets/75234646/db3ad135-aa5a-449f-99be-c03c10d03a21)

## Result:
Thus Matrix addition with unified memory is done and its performance with nvprof is checked.
