#include<iostream>
#include <ctime>
#include <cstdlib>
using namespace std;

/*int main ()
{
    int i, j,temp,pass=0;
    int a[] = {10,2,0,14,43,25,18,1,5,68,8,43,89,45};
    cout <<"Input list ...\n";
    for(i = 0; i<10; i++) {
        cout <<a[i]<<"\t";
    }
    cout<<endl;
    for(i = 0; i<10; i++) {
        for(j = i+1; j<10; j++)
        {
            if(a[j] < a[i]) {
                temp = a[i];
                a[i] = a[j];
                a[j] = temp;
            }
        }
        pass++;
    }
    cout <<"Sorted Element List ...\n";
    for(i = 0; i<10; i++) {
        cout <<a[i]<<"\t";
    }
    cout<<"\nNumber of passes taken to sort the list:"<<pass<<endl;
    return 0;
}
 */


// Function to generate random numbers
void randmNum(int arr[], int size) {
    srand(time(0));
    for (int i = 0; i < size; i++) {
        arr[i] = rand();
    }
}

// Function to swap two elements
void swap(int* a, int* b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

// Partition function for quicksort
int partition(int arr[], int low, int high) {
    int pivot = arr[high]; // Choose the last element as the pivot
    int i = low - 1;
    /*for (int j = low; j <= high - 1; j++) {
        if (arr[j] <= pivot) {
            i++;
            swap(&arr[i], &arr[j]);
        }
    }
    swap(&arr[i + 1], &arr[high]);
    return (i + 1);*/
}

// Quicksort function
void quicksort(int arr[], int low, int high) {
    if (low < high) {
        int pivot = partition(arr, low, high);

        quicksort(arr, low, pivot - 1);
        quicksort(arr, pivot + 1, high);
    }
}

// Function to print an array
void printArray(int arr[], int size) {
    for (int i = 0; i < size; i++) {
        std::cout << arr[i] << " ";
    }
    std::cout << "\n";
}

int main() {
    const int size = 1000;
    int numbers[size];

    // Generate random numbers
    generateRandomNumbers(numbers, size);

    std::cout << "Before sorting:\n";
    printArray(numbers, size);

    // Perform quicksort
    quicksort(numbers, 0, size - 1);

    std::cout << "After sorting:\n";
    printArray(numbers, size);

    return 0;
}
