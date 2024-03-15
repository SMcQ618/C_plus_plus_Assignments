#include <stdio.h>
#include <pthread.h>

void *thread_function(void *arg){
        pthread_t tid = pthread_self();
        // this will get the threads ID

        // unsigned long tid_as_ulong = (unsigned long)tid;
        // unsigned long tid_last_five_digits = tid_as_ulong % 100000;
        //this will print 5 digits of the thread's ID
        printf("Thread ID: %05llu\n", tid);
        return  NULL;
}

int main() {
        pthread_t threads[5];
        int i; //had to make  an integer as i was getting erros because i kept reefining it
        //the create 5 threads
        for (i = 0; i < 5; i++)
        {
                if (pthread_create(&threads[i], NULL, thread_function, NULL) != 0)
                {
                    perror("pthread_create");
                    return 1;
                }
        }
        //wait for each thread to finish otherwise create an error
        for (i = 0; i < 5; i++)
        {
                if (pthread_join(threads[i], NULL) != 0)
                {
                    perror("pthread_join");
                    return 1;
                }
        }

        return 0;
}
