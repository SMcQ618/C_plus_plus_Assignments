#include <iostream>
#include <cstdlib>
#include <ctime>

// Function to generate an array of random numbers
void generateRandomNumbers(int arr[], int size) {
    srand(time(0));
    for (int i = 0; i < size; ++i) {
        arr[i] = rand() % 1000 + 1;
    }
}

// Function to partition the array and return the pivot index
int partition(int arr[], int low, int high) {
    int pivot = arr[high];
    int i = low - 1;

    for (int j = low; j <= high - 1; j++) {
        if (arr[j] < pivot) {
            i++;
            std::swap(arr[i], arr[j]);
        }
    }

    std::swap(arr[i + 1], arr[high]);
    return i + 1;
}

// Quicksort algorithm to sort the array
void quicksort(int arr[], int low, int high) {
    if (low < high) {
        int pivot = partition(arr, low, high);
        quicksort(arr, low, pivot - 1);
        quicksort(arr, pivot + 1, high);
    }
}

int main() {
    // Generate different seed for the random number generator each time the program runs


    // Generate 1000 random integers between 1 and 1000
    const int arraySize = 1000;
    int numbers[arraySize];

    generateRandomNumbers(numbers, arraySize);

    // Sort the numbers using Quicksort
    quicksort(numbers, 0, arraySize - 1);

    // Print the sorted numbers
    for (int i = 0; i < arraySize; ++i) {
        std::cout << numbers[i] << " ";
    }

    return 0;
}
