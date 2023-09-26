#include <stdio.h>
#include <pthread.h>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <ctime>

#define N 256

using namespace std;
using namespace chrono;

int** matrix1; // Matrix 1
int** matrix2; // Matrix 2
int** result;  // result

// Function to perform matrix multiplication
void matrixMultiplication() {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            result[i][j] = 0;
            for (int k = 0; k < N; ++k) {
                result[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }
}

// Thread function
void *thread_function(void *arg) {
    // Measure the execution time of matrix multiplication
    auto startTime = high_resolution_clock::now();

    matrixMultiplication();

    auto endTime = high_resolution_clock::now();
    double time_taken = duration_cast<duration<double>>(endTime - startTime).count();

    // Output the result and execution time
    cout << "Matrix Multiplication Result:\n";
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            cout << result[i][j] << " ";
        }
        cout << endl;
    }

    cout << "Execution Time: " << setprecision(3) << time_taken << " seconds" << endl;

    return NULL;
}

int main() {
    // Allocate memory for matrices and result
    matrix1 = new int*[N];
    matrix2 = new int*[N];
    result = new int*[N];

    for (int i = 0; i < N; ++i) {
        matrix1[i] = new int[N];
        matrix2[i] = new int[N];
        result[i] = new int[N];
    }

    // Initialize matrix1 with all 1's
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            matrix1[i][j] = 1;
        }
    }

    // Initialize matrix2 with all 2's
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            matrix2[i][j] = 2;
        }
    }

    pthread_t threads[15]; // Array to hold thread IDs
    int i;

    // Create 10 threads
    for (i = 0; i < 15; i++) {
        if (pthread_create(&threads[i], NULL, thread_function, NULL) != 0) {
            perror("pthread_create");
            return 1;
        }
    }

    // Wait for each thread to finish
    for (i = 0; i < 15; i++) {
        if (pthread_join(threads[i], NULL) != 0) {
            perror("pthread_join");
            return 1;
        }
    }

    // Free the dynamically allocated memory
    for (int i = 0; i < N; ++i) {
        delete[] matrix1[i];
        delete[] matrix2[i];
        delete[] result[i];
    }
    delete[] matrix1;
    delete[] matrix2;
    delete[] result;

    return 0;
}
