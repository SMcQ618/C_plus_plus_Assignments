#include <iostream>
#include <cstdlib>
#include <ctime>


//will need two loops where 1 creates the random number and another that will sort them
void RandmInts(int arr[], int size)
{
        srand((time(0)));
        for (int i = 0; i < size; ++ i)
        {
          arr[i] = rand() % 1000 + 1;
        //for what ever i is, then it will equal that number
        }
}
//quicksort seems to be the most efficient

int partition(int arr[], int low, int high)
{
        int pivot = arr[high];
        int i = low - 1;
        for(int j = low; j <= high - 1; j++)
        {
           if (arr[j] < pivot)
           {
                i++;
                std::swap(arr[i], arr[j]);
           }
        }
        std::swap(arr[i + 1], arr[high]);
        return i + 1;
}

void SortInts(int arr[], int low, int high)
{
        if (low < high)
        {
                int pivot = partition(arr, low, high);
                SortInts(arr, low, pivot -1);
                SortInts(arr, pivot + 1, high);
        }
}
//this should switch the values, think this was the pass by reference from Cornerstone

int main()
{
        const int length = 1000;
        int numbers[length]; //the numbers have the size of 1000
        RandmInts(numbers, length); /* call the funciton*/

        //call the sorting funciton
        SortInts(numbers, 0, length - 1);

        for(int i = 0; i < length; ++i)
        {
                std::cout << numbers[i] << " \n ";
        }

        return 0;
}
