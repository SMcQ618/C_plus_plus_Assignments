#include <stdio.h>
#include <pthread.h>

void *thread_function(void *arg) {
    pthread_t tid = pthread_self(); // Get the thread's ID
    unsigned long tid_as_ulong = (unsigned long)tid;
    //unsigned long tid_last_five_digits = tid_as_ulong % 100000;
    printf("Thread ID: %d\n", tid_as_ulong);//tid_last_five_digits); // Print the last 5 digits of the thread's ID
    return NULL;
}

int main() {
    pthread_t threads[10]; // Array to hold thread IDs
    int i;

    // Create five threads
    for (i = 0; i < 10; i++) {
        if (pthread_create(&threads[i], NULL, thread_function, NULL) != 0) {
            perror("pthread_create");
            return 1;
        }
    }

    // Wait for each thread to finish
    for (i = 0; i < 10; i++) {
        if (pthread_join(threads[i], NULL) != 0) {
            perror("pthread_join");
            return 1;
        }
    }

    return 0;
}
