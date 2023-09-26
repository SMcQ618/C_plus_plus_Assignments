#include <iostream>
#include <chrono>
#include <iomanip>
#include <ctime>
//need iomanip for setting the precision
#define N 256
using namespace std;
using namespace chrono;

//Function to perform matrix multiplication
void matrixMultiplication(const int** matrix1, const int** matrix2, int** result)
{
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            result[i][j] = 0;
            for (int k = 0; k < N; ++k)
            {
                result[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }
}

//prints a single element of the matrix
void printMatrixElement(const int** matrix, int row, int col)
{
    cout << "Matrix[" << row << "][" << col << "]: " << matrix[row][col] << endl;
}

int main()
{
    //have to allocate the memory because I kept running out
    int** matrix1 = new int*[N];
    int** matrix2 = new int*[N];
    int** result = new int*[N];

    for (int i = 0; i < N; ++i)
    {
        matrix1[i] = new int[N];
        matrix2[i] = new int[N];
        result[i] = new int[N];
    }

    //Initialize matrix1 with all 1's
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            matrix1[i][j] = 1;
        }
    }

    //Initialize matrix2 with all 2's
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            matrix2[i][j] = 2;
        }
    }

    // Measure the execution time of matrix multiplication
    time_t start, end;
    time(&start);
    auto startTime = chrono::high_resolution_clock::now();

    matrixMultiplication((const int**)matrix1, (const int**)matrix2, result);

    /*THis is the good part that works
     * auto endTime = chrono::high_resolution_clock::now();
    //double duration = chrono::duration<double>(endTime- startTime).count();
    //the microsecond was declared as a double, and not initialized
    // using duration<double> gets the time in seconds which is what i was looking for*/

    auto endTime = high_resolution_clock::now();
    double time_taken = (endTime- startTime).count() / 1.0e9;

    //double duration = (endTime - startTime).count() / 1.0e9;
    //try to divide the microseconds by a factor os 1000000 to get seconds

    //underneath is the original time wehre we used microseconds
    // auto duration = chrono::duration_cast<chrono::microseconds>(endTime - startTime).count();

    // Output the result and execution time
    cout << "Matrix Multiplication Result:\n";
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            cout << result[i][j] << " ";
        }
        cout << endl;
    }


    //cout << "Execution Time: " << duration << " seconds" << endl;
    cout << "Execution Time: " << setprecision(3) << time_taken;
    cout<< " seconds" << endl;
    //need the time output to be more accurate
    // Print a single element of the resulting matrix
    int row = 1; // Choose the row index (0 to N-1)
    int col = 1; // Choose the column index (0 to N-1)
    printMatrixElement((const int**)result, row, col);

    // Free the dynamically allocated memory
    for (int i = 0; i < N; ++i)
    {
        delete[] matrix1[i];
        delete[] matrix2[i];
        delete[] result[i];
    }
    delete[] matrix1;
    delete[] matrix2;
    delete[] result;

    return 0;
}

