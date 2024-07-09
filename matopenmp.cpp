#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <chrono>
#include <omp.h>
#include <time.h>
#include <thread>

using namespace std;
#define N 2048
void initialize_matrix(int** matrix) {
        #pragma omp parallel for
        for(int i = 0; i < N; i++) {
                for(int j = 0; j < N; j ++) {
                        matrix[i][j] = rand() % 10;
                }
        }
}

void multiply_matrix(int** A, int** B, int** C) {
        #pragma omp parallel for
        for(int i = 0; i < N; i++) {
                for(int j = 0; j < N; j++) {
                        C[i][j] = 0;
                        for(int k = 0; k < N; k++) {
                                C[i][j] += A[i][k] * B[k][j];
                        }
                }
        }
}

int main() {
        //this sets up the nubmer of threads
        omp_set_num_threads(4);

        int** A = new int*[N];
        for(int i = 0; i < N; ++i)
                A[i] = new int[N];

        int** B = new int*[N];
        for(int i = 0; i < N; ++i)
                B[i] = new int[N];

        int** C = new int*[N];
        for(int i = 0; i < N; ++i)
                C[i] = new int[N];

        //make the numbers random
        srand(42);
        //initalize the matricies for both A and B
        initialize_matrix(A);
        initialize_matrix(B);

        auto start_time = chrono::high_resolution_clock::now();

        #pragma omp parrallel
        {
                num_threads = omp_get_num_threads();
                #pragma omp single
                {
                        cout << "Number of threads: " << omp_get_max_threads() << endl;
                }       
                
                multiply_matrix(A,B,C);
        }       
        auto end_time = chrono::high_resolution_clock::now();
        
        auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
        
        cout << "Time taken for the matrix: " << fixed << setprecision(4) << duration << " milliseconds" << endl;
        //get the number of threads used
        int num_threads = omp_get_max_threads();
        cout << "Number of threads used: " << num_threads << endl;
        
        //cout << "Time taken for the matrix: " << fixed << setprecision(4) << duration.count() / 1000.0 << " milliseconds" << endl;

        for(int i = 0; i < N; i++) {
                delete[] A[i];
                delete[] B[i];
                delete[] C[i];
        }       
        delete[] A;
        delete[] B;
        delete[] C;
        
        return 0;
}
