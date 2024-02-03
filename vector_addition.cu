#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#define N 1024

__global__ void vector_add(float *a, float *b, float *result, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
                result[idx] = a[idx] + b[idx];
        }
}
int main() {

        size_t bytes = N * sizeof(float);

        //need to allocate the host memory
        float *h_a, *h_b, *h_result;
        h_a = (float*)malloc(N * sizeof(float));
        h_b = (float*)malloc(N * sizeof(float));
        h_result = (float*)malloc(N * sizeof(float));

        for(int i = 0; i < N; ++i) {
                h_a[i] = 1.0f;
                h_b[i] = 2.0f;
        }

        //make sure device memory is not bad
        float *d_a, *d_b, *d_result;
        cudaMalloc((void**)&d_a, bytes);
        cudaMalloc((void**)&d_b, bytes);
        cudaMalloc((void**)&d_result, bytes);

        //get data from host to device
        cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

        //get the grid
        dim3 blockSize(256);
        dim3 gridSize((N + blockSize.x - 1) / blockSize.x);

        //launch the kernel
        vector_add<<<gridSize, blockSize>>>(d_a, d_b, d_result, N);

        //copy the result from device to host
        cudaMemcpy(h_result, d_result, bytes, cudaMemcpyDeviceToHost);

        //Dislpay result or further processing
        for(int i = 0; i < N; ++i) {
                std::cout << h_result[i] << " ";
        }

        std::cout << std::endl;

        //free the device and host memory
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_result);
        free(h_a);
        free(h_b);
        free(h_result);

        return 0;
}
