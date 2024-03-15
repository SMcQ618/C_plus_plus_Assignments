#include <iostream>
#include <chrono>
#include <iomanip>
//need iomanip for setting the precision
#define N 512
#define TileSize 32
/* 32 is used as cache line size of 64bytes, using this sizze ensures the dta from teh local memory i fetched into teh cache. optimizing the memory access. and the size helps maximize teh case utiliziton and minimize thrashing*/

using namespace std;
using namespace chrono;

void matrixMultiplicationT(const int** matrix1, const int** matrix2, int** result)
{
        for (int i = 0; i < N; i += TileSize)
        {
                for (int j = 0; j < N; j+= TileSize)
                {
                        for (int k = 0; k < N; k+= TileSize)
                        {
                                for (int l = i; l < i + TileSize; ++l)
                                {
                                        for (int m = j; m < j + TileSize; ++m)
                                        {
                                                int sum = 0;
                                                for (int n = k; n < k + TileSize; ++n)
                                                {
                                                        sum += matrix1[l][n] * matrix2[n][m];
                                                }
                                                result[l][m] += sum;
                                        }
                                }
                        }
                }
        }
}

void printMatrixEle(const int** matrix, int row, int col)
{
        cout << "Matrix[" << row << "][" << col << "]: " << matrix[row][col] << endl;
}

int main()
{
        int** matrix1 = new int*[N];
        int** matrix2 = new int*[N];
        int** result = new int*[N];
        //reason for using new is for the dynamic memory allocation
        for (int i = 0; i < N; ++i)
        {
                matrix1[i] = new int[N];
                matrix2[i] = new int[N];
                result[i] = new int[N];
        }

        for (int i = 0; i < N; ++i)
        {
                for(int j = 0; j < N; j++)
                {
                        matrix1[i][j] = 1;
                        matrix2[i][j] = 2;
                        result[i][j] = 0;
                }
        }

        time_t start, end;
        time(&start);
        auto startTime = chrono::high_resolution_clock::now();

        matrixMultiplicationT((const int**)matrix1, (const int**)matrix2, result);

        auto endTime = high_resolution_clock::now();
        double time_taken = (endTime - startTime).count() / 1.09;
        //was getting an error with how the time is coming out but i think execution time and  a stopwatch time differently
        cout<< "Matrix Multiplication Result:\n";
        for (int i = 0; i < N; ++i)
        {
                for (int j = 0; j < N; ++j)
                {
                        cout << result[i][j] << " ";
                }
                cout << endl;
        }

        cout << "Execution Time: " << setprecision(3) <<time_taken << " seconds" << endl;

        int row = 1;
        int col = 1;
        printMatrixEle((const int**)result, row, col);

        for(int i = 0; i < N; ++i)
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
