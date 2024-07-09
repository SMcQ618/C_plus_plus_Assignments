#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define N 1033 // number of points
#define SMALL_PERCENTAGE 0.30
#define MEDIUM_PERCENTAGE 0.32
#define LARGE_PERCENTAGE 0.38

const float large_size_range[2] = {16.0f, 20.0f};
const float medium_size_range[2] = {12.0f, 16.75f};

// Generate random sizes within the range
float generateRandomSize(const float *size_range) {
    return size_range[0] + (rand() / (float)RAND_MAX) * (size_range[1] - size_range[0]);
}

// Kernel to compute linear regression
__global__ void linearRegress(float *x, float *y, float *slope, float *intercept) {
    // Calculate the mean of the x and y values
    float x_mean = 0.0f;
    float y_mean = 0.0f;

    for(int i = 0; i < N; ++i) {
        x_mean += x[i];
        y_mean += y[i];
    }
    x_mean /= N;
    y_mean /= N;

    // Compute the slope and intercept of regression
    float numerator = 0.0f;
    float denominator = 0.0f;
    for(int i = 0; i < N; ++i) {
        numerator += (x[i] - x_mean) * (y[i] - y_mean);
        denominator += (x[i] - x_mean) * (x[i] - x_mean);
    }

    *slope = numerator / denominator;
    *intercept = y_mean - (*slope) * x_mean;
}

// Error checking function
void checkCUDAError(cudaError_t result, const char *file, const int line) {
    if(result != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d: %s\n", file, line, cudaGetErrorString(result));
        exit(EXIT_FAILURE);
    }
}

int main() {
    // Allocate memory on host and device
    float *h_size = (float*)malloc(N * sizeof(float));
    float *h_price = (float*)malloc(N * sizeof(float));
    float *h_slope = (float*)malloc(sizeof(float));
    float *h_intercept = (float*)malloc(sizeof(float));
    // Generate random points on the host
    srand(time(NULL));
    for(int i = 0; i < N; ++i) {
        h_size[i] = generateRandomSize(large_size_range);
        h_price[i] = generateRandomSize(medium_size_range);
    }

    // Allocate memory on the device
    float *d_x, *d_y, *d_slope, *d_intercept;
    checkCUDAError(cudaMalloc((void**)&d_x, N * sizeof(float)), __FILE__, __LINE__);
    checkCUDAError(cudaMalloc((void**)&d_y, N * sizeof(float)), __FILE__, __LINE__);
    checkCUDAError(cudaMalloc((void**)&d_slope, sizeof(float)), __FILE__, __LINE__);
    checkCUDAError(cudaMalloc((void**)&d_intercept, sizeof(float)), __FILE__, __LINE__);

    // Copy data from host to device
    checkCUDAError(cudaMemcpy(d_x, h_size, N * sizeof(float), cudaMemcpyHostToDevice), __FILE__, __LINE__);
    checkCUDAError(cudaMemcpy(d_y, h_price, N * sizeof(float), cudaMemcpyHostToDevice), __FILE__, __LINE__);

    // Compute the linear regression on the GPU
    int numThreadsPerBlock = 128;
    int numBlocks = (N + numThreadsPerBlock - 1) / numThreadsPerBlock;
    linearRegress<<<numBlocks, numThreadsPerBlock>>>(d_x, d_y, d_slope, d_intercept);
    checkCUDAError(cudaGetLastError(), __FILE__, __LINE__);

    // Transfer the results from the device to the host
    checkCUDAError(cudaMemcpy(h_slope, d_slope, sizeof(float), cudaMemcpyDeviceToHost), __FILE__, __LINE__);
    checkCUDAError(cudaMemcpy(h_intercept, d_intercept, sizeof(float), cudaMemcpyDeviceToHost), __FILE__, __LINE__);

    // Output the results
    printf("Slope of the line: %f\n", *h_slope);
    printf("Intercept of the line: %f\n", *h_intercept);

    // Free memory
    free(h_size);
    free(h_price);
    free(h_slope);
    free(h_intercept);
    checkCUDAError(cudaFree(d_x), __FILE__, __LINE__);
    checkCUDAError(cudaFree(d_y), __FILE__, __LINE__);
    checkCUDAError(cudaFree(d_slope), __FILE__, __LINE__);
    checkCUDAError(cudaFree(d_intercept), __FILE__, __LINE__);

    return 0;
}
