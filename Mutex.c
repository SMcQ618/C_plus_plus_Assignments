#include <stdio.h>
#include <pthread.h>

#define NUM_THREADS 10
//create a constant out of the number of threads I want

int sharedVariable = 0;
//create a variable and set it equal to 0
pthread_mutex_t mutex;

//create a fucntion to execut each thread
void* threadFunc(void* threadID) {
        pthread_t id = *((int*)threadID);

        //lock the mutex before accessing teh shared vairbale
        pthread_mutex_lock(&mutex);

        sharedVariable += id + 1;
        //this will increment the shared variable by the thread i
        //then unlock teh mutex after updating it
        pthread_mutex_unlock(&mutex);

        pthread_exit(NULL);
}

int main() {
        pthread_t threads[NUM_THREADS];
        int threadIDs[NUM_THREADS];

        //initialize the mutex
        pthread_mutex_init(&mutex, NULL);

        int i;

        for(int i = 0; i < NUM_THREADS; ++i) {
                threadIDs[i] = i + 1;
        int status = pthread_create(&threads[i], NULL, threadFunc, (void*)&threadIDs[i]);
                if (status) {
                        fprintf(stderr, "Error creating the threads: %d\n", status);
                        return -1;
                }
        }

        for(int i = 0; i < NUM_THREADS; ++i) {
                pthread_join(threads[i], NULL);
        }
        //pass the correct arguement type
        printf("Shared Varibale value: %d\n", sharedVariable);
        pthread_mutex_destroy(&mutex);
        //destory the mutex to fix memeory

        return 0;
}
