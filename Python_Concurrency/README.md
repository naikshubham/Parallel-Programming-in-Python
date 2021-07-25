## Python Concurrency

### Concurrency concepts
- Concurrency is the execution of multiple instructions sequences at the same time. This is possible when the instructions being executed are independent of each other.
- This independency is important both in the order of execution and in the use of shared resources.

##### order of execution
- The order of execution of these instruction sequences should have no effect on the eventual outcome.
- If task 1 finishes after tasks 2 and 3, or if task 2 is initiated first, but finishes last, the eventual outcome should still be the same.

##### Shared resources
- The different instruction sequences share as few resources between each other as possible. The more shared resources that exist between concurrently executing instructions, the more coordination is neccessary between those instructions in order to make sure that the shared resource stays in a consistent state.
- This coordination is typically what makes concurrent programming complicated.However we can avoid many of these complications by choosing the right concurrency patterns and mechanisms depending on what we are trying to achieve.

### Concurrency Types
- The two main forms of concurrency that we will focus on  are parallel programming and asynchronous programming

#### Parallel Programming
- Parallel computing invovles taking a computational task and splitting it into smaller subtasks that are then assigned to multiple threads or processes to be executed on multiple processor cores simultaneously.
- With single threaded code, if we have multiple processor cores on our system only one core will be charged with executing our task, while the other cores sit idle or execute instructions for other programs.
- With parallel programming all our processor cores can be engaged, theoretically cutting our processing time by a factor of the number of cores that we have available.

##### Best suited for CPU intensive tasks
- Because it's designed to take advantage of multi-core systems, parallel programming is best suited for tasks that are CPU intensive. That is tasks in which most of the time is spent solving the problem rather than reading to or writing from a device.
- Another term for such tasks are CPU bound tasks, they will perform better if we could get better performance out of the CPU.
- Example of such workloads are string operations, search algorithms, graphics processing, any number crunching algorithm.
- If we havw a task in which most of the time is spent in reading from or writing to a device which is more comming known as input or output or IO, then that task is more suited for asynchronous programming.

##### Asynchronous programming
- Asynchronous programming's concurrency model works by delegating a subtask to another actor such as another thread or device, and then continuing to do other work instead of waiting for a response.
- When the subtask is completed, the actor then notifies the main thread, which calls the function to handle the results. This is usually known as a callback function.
- In some languages, instead of executing a callback, the main thread is given an object that represents the results from a not-yet-completed operation. This object is typically called a future, promise or task and the application can either wait on its completion or check back at a later time.
- Examples of IO bound tasks are database reads and writes, web service calls, or any form of copying, downloading or uploading data either to disk or to a network.

### Concurrency in Python
- Python supports both parallel and asynchronous programming natively.

##### Threading
- The threading modeule in python allows us to create thread objects that are mapped to native operating system threads and can be used for the concurrent execution of code.
- However, it should be noted that in Cpython which is the most common implementation of the Python specification, **threads are limited to executing python code serially by a mechanism known as the Global Interpreter Lock**. Therefore with threading in Python, **concurrency is limited to what we get when the operating systems switches between threads**. 

##### Multiprocessing
- The multiprocessing package in an implementation of parallelism that uses sub-processes instead of threads. This technique avoids the global interpreter lock and allows Python to take advantage of multiple processor cores.

##### Concurrent.futures
- The `concurrent.futures` module was introduced in Python3.2 and its provides a common high level interface using thread pools or process pools.

##### asyncio
- In Python3.4 the asyncio module was introduced as a provisional package to enable asynchronous programming. Starting python3.6 api was declared to be stable.

### Example : Thumbnail Maker
- Purpose of this library will be that it will take multiple images of a certain size and produces smaller copies of the original image while keeping the dimensions intact.
- Attempting to serve large images will lead to very slow load times and high data usage.
- A typical workflow would be to download the images from some form of file store like Amazon S3 to the image or application server. Peform the resizing and then re-upload the resized image back to the file store or some other location where they can be served. 
- The download of the images is an IO-bound task, and resizing is an CPU bound task. 

### Threading : parallelism using threads
- **Process** : A process can be defined as the execution context of a running program. An alternative and more approacable definition is that a process is a running instance of a computer program.
- Every executing process has system resources and a section of memory that is assigned to it. It also has security attributes in a process state.
- A process is composed of one or more threads of execution. A thread is the smallest sequence of instructions that can be managed, that is scheduled and executed by an operating system.
- A program can be composed of single thread of execution or multiple threads of execution. When multiple CPU Cores are available, each thread's instructions can be executed at the same time in parallel of multiple cores.
- If only a single core is available, the threads share time on that core. In either scenario, the result is that the use of multiple threads allows a process to perform multiple tasks at once.
- For example, in the media player one thread can be playing a current song while another is figuring out the next song to play and downloading it, while again another thread is responding to user clicks and navigation.
- Another example is a web or application server that uses a pool of threads to respond to multiple requests simultaneously. Each request is handled by a thread from the pool. The thread executes whatever task is assigned, and when it is completed it returns to the pool to wait for the next request.

### Creating Threads in Python
- Python has `threading` package in the standard library.
- This package allows us to create thread objects that map directly to operating system threads. The simplest way to create a thread is to instantiate an object of the thread class, passing in the thread function, as well as any function arguments, and then calling the start method on the thread object we created.

```python
import threading

def do_some_work(val):
    print('doing some work in thread')
    print('echo: {}'.format(val))
    return
    
val = 'text'
t = threading.Thread(target=do_some_work, args=(val,)) # define target func and                                                            args
t.start()   # start execution
t.join()    # suspend execution until new thread completes by calling join method               on the thread object that we want to wait for its completion
```

```python
# full threading.Thread constrcutor

class threading.Thread(group=None,
                       target=None,
                       name=None,
                       args=(),
                       kwargs=(),
                       daemon=None)
```

- Group parameter is reserved for future use, so it's always set to None. The target is the function to be invoked. It can specify a name for a thread using the name parameter. If we choose not to,default name will be used, which will be the word Thread and the Counter appended to the thread.
- args is the argument tuple, while the kwargs is a dictionary of keyword arguments for the function.
- daemon parameter specifies whether the thread will be terminated, if its parent thread terminates or not.

- This way of executing tasks with threads is the most common usage pattern for threading. It considers the thread to be a worker executing the instructions in a target function.
- There is a second usage pattern of threading where we use threads known as the **worker function** performing some tasks, but as a unit of work instead.

```python
# second method using worker function

import threading

class FibonacciThread(threading.Thread):
    def __init__(self, num):
        Thread.__init__(self)
        self.num=num
        
    def run(self):
        fib = [0] * (self.num + 1)
        fib[0] = 0
        fib[1] = 1
        for i in range(2, self.num +1):
           fib[i] = fib[i-1] + fin[i-2]
           print(fib[self.num])
           
myFibTask1 = FibonacciThread(9)
myFibTask2 = FibonacciThread(12)

myFibTask1.start()
myFibTask2.start()

myFibTask1.join()
myFibTask2.join()
```

- For second method, we create a class that inherits from the thread class in the threading package and overrides the run method. The dunder init method can also be overwritten to provide additional state in the object via the constructor, but we should keep in mind that whenever we override a dunder init method in the sub-class of thread, the super class's dunder init method must always be called first before performing any other operations.
- here in our overridden run method, we are calculating the fibonacci number for the number passed into the constructor and printing out the value.
- we can create objects of our class and because they are**thread objects**, we can **call start** to schedule the threads to start, and we can **call join** to wait for the threads to complete.
- These threads we run concurrently on the machine, and then the program would exit.

```python
threads = []
for url in img_url_list:
    t = threading.Thread(target=download_image, args=(url,))
    t.start()
    threads.append(t)
    
for t in threads:
    t.join()
```

- We can also go ahead and call `thread.join()` here, but that would be a mistake, because what would happend is that in a loop, the main thread would create a new thread, start it and then wait for it to complete before continuing the loop. This is'nt what we want.
- What we want is to create the worker threads, and then wait for all of them to complete. So to accomplish that we will create a list that will hold references to the threads objects, and then every time we create a thread in the loop we add it to the list. 
- Then after all the threads are started, we are going to loop through each of the thread list and call the join on each of the threads. So that the main thread is forced to wait for all the threads to complete before continuing.

### Threads working

```python
import threading

def do_some_work(val):
   print('doing some work in thread')
   print('echo: {}'.format(val))
   return
val = "text"
t = threading.Thread(target=do_some_work, args=(val,))
t.start()
t.join()
print('Done')
```

- We can trace the lifecycle of threads that are created and executed when the code is run. When the program starts there's only one thread in existence, the mainThread. The mainThread executes the instructions for importing the threading library, defining the do_some_work method, and creating the val variable.
- The mainThread then creates the new thread. At this point in time the new thread is in a new state.

    






























