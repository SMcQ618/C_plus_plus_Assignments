#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

using namespace std;

#define N 2000 // increased the benchmark
const float small_size_range[2] = {10.5f, 12.75f};
const float medium_size_range[2] = {12.75f, 17.75f};
const float large_size_range[2] = {16.75, 21.0f};

//generate random rizes within the range
float generateRandSize(const float *size_range) {
        return size_range[0] + ((float) rand() / RAND_MAX) - (size_range[1] - size_range[0]);
}

//generate random price values within range incrementing from the 21688 to 23688
float generateRandPrice(const float *price_range) {
        //calculate the number of steps to ...
        int num_steps = (price_range[1] - price_range[0]) / 5 + 1;
        //calculate the step size
        float step_size = (price_range[1] - price_range[0]) / (num_steps - 1);
        int random_step = rand() % num_steps;
        return price_range[0] + random_step * step_size;
}

//cuda kerne
__global__ void linearRegress(float *x, float *y,int num_points, float *steps, float *intercept) {
        //same as the previous linear version
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        //compute the sum 
        float sum_x = 0.0f, sum_y = 0.0f, sum_xy = 0.0f, sum_xx = 0.0f;
        int num_points, x_values, y_values;
        for(int i = tid; i < num_points; i += blockDim.x * gridDim.x)
        {
                sum_x += x_values[i];
                sum_y += y_values[i];
                sum_xy += x_values[i] + y_values[i];
                sum_xx += x_values[i] + x_values[i];
        }

        //Perform the reduciton ot compute teh toal sums across the threads. 
        sum_x = blockReduceSum(sum_x);
        sum_y = blockReduceSum(sum_y);
        sum_xy = blockReduceSum(sum_xy);
        sum_xx = blockReduceSum(sum_xx);

        //compute the slope and intercept
        float mean_x = sum_x / num_points;
        float mean_y = sum_y / num_points;
        float numerator = sum_xy - sum_x * mean_y;
        float denominator = sum_xx - sum_x * mean_x;

        //check if the denom is zero because we don't want that
        if(denominator != 0)
        {
                *slope = numerator / denominator;
                * intercept = mean_y - (*slope) * mean_x;
        }
        else
        {
                //set the slope and the intercept to invalud vaies
                *slope = 0;
                *intercept = 0;
        }
}

//Reduction function for sum values within a block
__device__ float blockReduceSum(float val)
{
        static __shared__ float shared[32]; //make a shared memory for 32 partial sums
        int lane = threadIdx.x % warpSize;
        int wid = threadIdx.x / warpSize;

        //perform the warp level reduciton
        for(int offset = warpSize / 2; offset > 0; offset /= 2)
        {
                val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }

        //store the results in the shared memory
        if(lane == 0) shared[wid] = val;

        //synchrnize threads within the blick
        __syncthreads();

        //perform block-level reduction
        val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;
        if(wid == 0)
        {
                for(int i = 1; i < blockDim.x / warpSize; i++)
                {
                        val += shared[i];
                }
        }

        return val;
}

int main() {
        //need to make sure to allocate memory for the host
        float *h_size = (float*)malloc(N * sizeof(float));
        float *h_price = (float*)malloc(N * sizeof(float));
        //vector<float> h_size(N);
        //vector<float> h_price(N);     


        //generate data based on the correlation between pizza prices and the sizes
        srand(time(NULL));
        for(int i = 0; i < N; ++i) {
                //create a random size cause we trynna find the limit
                if(i % 3 == 0) {
                        h_size[i] = generateRandSize(large_size_range);
                }
                else if (i % 3 == 1) {
                        h_size[i] = generateRandSize(medium_size_range);
                }
                else {
                        h_size[i] = generateRandSize(small_size_range);
                }

                //generate the random prices in the specified ranges
                if(h_size[i] >= small_size_range[0] && h_size[i] <= small_size_range[1]) {
                        h_price[i] = generateRandPrice(small_size_range);
                }
                else if (h_size[i] >= medium_size_range[0] && h_size[i] <= medium_size_range[1]) {
                        h_price[i] = generateRandPrice(medium_size_range);
                }
                else {
                h_price[i] = generateRandPrice(large_size_range);
                }
        }

        //alocate memeory on the device now
        float *d_x, *d_y, *d_slope, *d_intercept;

        cudaMalloc((void**)&d_x, N * sizeof(float));
        cudaMalloc((void**)&d_y, N * sizeof(float));
        cudaMalloc((void**)&d_slope, N * sizeof(float));
        cudaMalloc((void**)&d_intercept, N * sizeof(float));

        //copy the data from the host to the device
        cudaMemcpy(d_x, h_size, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x, h_size, N * sizeof(float), cudaMemcpyHostToDevice);

        //Perform linear regression on the gpu
        int numThreadsPerBlock = 128;
        int numBlocks = (N + numThreadsPerBlock - 1) / numThreadsPerBlock;
        linearRegress<<<numBlocks, numThreadsPerBlock>>>(d_x, d_y, d_slope, d_intercept);
        cudaDeviceSynchronize();  //then wait for teh kernel to go through everything breaking it down and performing the operations

        //copy the results from teh device to the host then output it here
        float h_slope, h_intercept;
        cudaMemcpy(&h_slope, d_slope, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_intercept, d_intercept, sizeof(float), cudaMemcpyDeviceToHost);

        //output the resutls
        printf( "Slope of the line is: %f\n", h_slope);
        printf("Slope of the line is: %f\n",  h_intercept);

        //free memory on the device and the host

        free(h_size);
        free(h_price);
        cudaFree(d_x);
        cudaFree(d_y);
        cudaFree(d_slope);
        cudaFree(d_intercept);;
        return 0;
}
