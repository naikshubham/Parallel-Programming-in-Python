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

### Multiprocessing 

```python
import multiprocessing

def do_some_work(val):
    print("doing some work in thread")
    print("echo: {}".format(val))
    return
    
if __name__ == '__main__':
    val='text'
    t = multiprocessing.Process(target=do_some_work, args=(val,))
    t.start()
    t.join()
```

##### Pickling
- Arguments passed to the process constructor must be picklable. Pickling is the process whereby a Python object hierarchy is converted into a byte stream. "unpickling" is the inverse operation. This is also know as object serialization and deserialization.

##### Picklable objects
- Picklable objects include types such as None, booleans (True, False), Numbers(int, floats, complex numbers), strings, byte arrays, collections containing only picklable objects, top-level functions and classes whose attributes are picklable.


- After instantiating our process object, we call start on it to kick it off, and then we can call join on the object to block until the process completes its work and exits.
- The end results of this code is that our main python process creates a worker Python process which executes the `do_some_work` method and prints out the two strings. The child process then exits, which allows the main process to exit.

##### Necessity of the if __name__ == '__main__' block
- When the child process gets started, it needs to import the script containing the target function. 
- If the code for creating the **new process** is in the top level of the script, it gets executed during the import, which means that a new process is created which in turn tries to import the script causing new processes to keep getting recursively created until a runtime error occurs.
- Putting the code for **creating the new process** in the `__main__` block ensures that it only gets run when the script is executed, and not when its imported.

```python
class multiprocesssing.Process(group=None,
                               target=None,
                               name=None, #name of the process
                               args=(), #argument tuple for target invocation
                               kwargs={},#dict of kwargs for target invocation
                               daemon=None)
```

##### Daemon Process
- The difference betweeb Daemon & Non-Daemon process is that when a process exits, it terminates all its daemon child processes. If a process has child processes that are not daemon processes, the process will by default not exit until all its non-daemon child processes have exited.
- In situations where we want a background process that runs without blocking the main program from exiting, setting the child process as daemon will come in handy.
- It is important to note that a daemon child process is not allowed to create its own child processes. So we should keep that in mind while flagging a child process as a daemon or not.

##### Terminating processes
- Processes are killable via an OS provided API. In python, two API calls are used to manage the aliveness of a process, `is_alive()` and the `terminate()` methods.

```python
import multiprocessing
import time

def do_work():
    print('Starting do_work function')
    time.sleep(5)
    print('Finished do_work function')
    
if __name__ == '__main__':
    p = multiprocessing.Process(target=func)
    print('[Before start] process is alive:{}'.format(p.is_alive()))
    p.start()
    print('[Running] Process is alive:{}'.format(p.is_alive()))
    p.terminate()
    p.join()
    print('[After termination] Process is alive:{}'.format(p.is_alive()))
    print('Process exit code:{}'.format(p.exitcode))
```

##### process.terminate() precautions
- Shared resoucres used by child processes may be put in an inconsistent state after being killed
- Finally clauses and exit handlers will not be run on forcibly killed processes. So if we have a critical code in finally or exit handlers its better not to use terminate method.

### Process Pools

```python
class multiprocessing.Pool([num_processes
                             [,initializer
                             [,initargs
                             [,maxtasksperchild]]]]))
```

- If set, each worker process executes the initialization function once at startup.
- The final constructor parameter is maxtasksperchild. Bydefault this is set to None, meaning that the worker processes live as long as the pool is alive.
- However if the value is set, then after executing the specified number of tasks a worker process if killed and replaced with a new worker process. This ensures that a long-running worker process periodically has to release any system resources it holds.
- The most commong usage pattern for using process pool is to define a function to be executed and an iterable of items that serve as the function argument `map(func, iterable[, chunksize])` and than the `pool.map` method to apply the function to each value in parallel.

```python
import multiprocessing

def do_work(data):
    return data**2
    
def start_process():
    print('starting', multiprocessing.current_process().name)
    
if _name_ == '__main__':
    pool_size = multiprocessing.cpu_count() * 2
    pool = multiprocessing.Pool(processes=pool_size, initializer=start_process)
    inputs=list(range(10))
    outputs=pool.map(do_work, inputs)
    print('Outputs:', outputs)
    
    pool.close() # no more tasks
    pool.join() # wait for the worker processes to exit
```

- The map method is a blocking call, and upon completion, it returns a list of the results in the original order of the iterable.
- there is a non blocking version of map called `map_async` which also allows us to specify an optional callback function, which can act on the results of each function call as they become available.
- `pool.join` blocks the main process until all the worker processes in the pool have exited. The **`join`** method can only be called after calling `pool.close()` or `pool.terminate()`


### Abstracting concurrency
- In both threading and multiprocessing we are submitting the tasks to some executor.

```python
def sayhello(name):
    print("hello from {}".format(name))
    return
    
excutor = # new threadpool or processpool instance
future_result = executor.submit(sayhello, "hello")

# concurrent
executor = # new threadpool or processpool instance 
names = ['name1', 'name2', 'name3']
results = executor.map(sayhello, names)
```

- For executing multiple tasks concurrently we can have a map method that applies a target function to a iterable or iterable of arguments using multiple threads or processes in the background.
- What this buys us is that instead of directly dealing with the underlying threads or processes, we simply submit tasks to the interface and let the implementation manage the execution of the task.
- We don't need to worry anymore about instantiating the threads or processes, starting them and joining them, all this gets taken care of for us.
- Now if we have a program design where we need a finer grain control over the threads or processes that will be running our tasks then the executor interface is not a good choice and should stick to the process and threading API.
- But if the tasks are straightforward, then the executor interface makes it easier to parallelize operations.

### The Executor API
- The executor API is provided by the `concurrent.futures` module and it exposes only three methods, **submit, map and shutdown**.

##### Executor methods

`submit(fn, *args, **kwargs) # schedule a function to run`
- The **submit** method schedules the passing function to run on one of the executor's workers and returns a future object that represents the execution state of the function.
- The call to submit is non-blocking. The submit method immediately returns the future object to the caller, so that the caller can continue executing and get the results of the computation at a later time by calling the results method on the returned future object.

`map(func, *iterables, timeout=none, chunksize=1)`
- The map method uses the executor worker pool to apply the passed-in function to every member of the iterable or iterables concurrently. Each worker concurrently operates on an item from the iterable or a tuple of items from the iterables until all the items are processed.
- Therefore the degree of the concurrency depends on the number of workers in the worker pool. The map function returns an iterator that can raise a timeout exception if a timeout is set and the value is not available by the timeout period. If the timeout is set to None, there is no limit to the waitime.

`shutdown(wait=True)` # stop accepting tasks and shutdown
- The **shutdown** method is used to signal to the executor that no more tasks will be submitted to it, and that it should free up any resources that it's currently using once the currently-running tasks, if any are done executing.

- In below example we use a ThreadPoolExecutor to attempt to download two HTML pages concurrently.

```python
from concurrent.futures import ThreadPoolExecutor
from urllib.request import urlopen

def load_url(url, timeout):
    with urlopen(url, timeout=timeout) as conn:
        return conn.read()
    
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        url1 = "http://www.cnn.com"
        url2 = "http://www.some.com"
        
        # the submitting of the tasks is non-blocking and returns a future object
        # so both tasks get submitted immediately one after the other
        f1 = executor.submit(load_url, url1, 60)
        f2 = executor.submit(load_url, url2, 60)
        
        # to get the result of the execution we call the result method on the             future object. This is where we actually get the downloaded bytes array           returned by the load_url method
        try:
            data1 = f1.result()
            print("{} page is {} bytes".format(url1, len(data1)))
            data2 = f2.result()
            print("{} page is {} bytes".format(url2, len(data2)))
        except Exception as e:
            print("Exception downloading page" + str(e))
```

- Notice that the try catch isn't around the submit its around the try catch method. This is because exceptions that happend during the execution of submitted functions dont get raised to the caller during the submit or map operation
- For CPU bound task instantiate a ProcessPoolExecutor

```python
from concurrent.futures import ProcessPollExecutor
import hashlib

texts=['this is good','text is good']

def generate_hash(text):
    return hashlib.sha384(text).hexdigest()
    
if __name__ == '__main__':
    with ProcessPoolExecutor() as executor:
        for text, hash_t in zip(texts, executor.map(generate_hash, texts)):
            print('%s hash is: %s' %(text, hash_t))
```

### Future Object
- Returned by the submit method. The map method, even though it doesn't return a future object also uses an iterable of the future objects internally in its implementation.
- **Future** object is an object that **acts as a proxy for a result that is initially unknown**, typically because a computation of its value is not yet complete.
- In Python, the future object takes the extra responsibility of encapsulating the execution state of the computation, allowing the developer to manage that state to some extent and be notified or perform some action when the computation completes by making use of callbacks.

#### Asynchronous Programming
- By being strictly non-blocking and returning a future object, the executor API drives us to think about programming asynchronously.
- In asynchronous program, we typically focus on the main thread. The actor that actually executes the task is concealed from us.
- The main thread simply submits the task to the actor, which may be another thread, process or OS function, and then continues executing until it needs the result from the execution or until the execution is completed.
- In python, the **future object enables asynchronous programming**. The **executor represents the actor** and immediately returns the future object so that the main thread is not blocked and can go on doing other things.

```python
future = executor.submit(func, args*)
----do other things----
result = future.result()
```

- When the main thread needs the result, then it calls the `future.result` method to get back the result of the function. If the function is not yet completed, then `future.result` will block until it completes or until a timeout occurs if one is specified.

##### Future methods
- `cancel` : attempts to cancel execution. Return True is succesful. If the function call is currently being executed and cannot be cancelled, then the method will return False
- `done()` : done method is used to check whether the function call has completed execution or was successfully cancelled.
- `exception(timeout=None)` : returns the exception raised by the call if any
- If the execution is not yet completed and timeout is specified, then the method will wait until the time out expires.
- `add_done_callback(fn)` : attaches function to be called on completion or cancellation.

#### Module functions
- When we have collection of callback function, we may want to wait for all of them to complete. `concurrent.futures.wait` takes in an iterable of future objects and blocks until the futures are completed.

```python
concurrent.futures.wait(fs, timeout=None, return_when=ALL_COMPLETED)
```

- The timeout parameter can be used to control how long to wait before returning.If its set to None then the wait time is unlimited. By default, the wait method waits until all futures are completed as specified by the return_when parameter.
- If we dont want all the futures to complete and return as and when the tasks are completed then there is a `as_completed` function. The `as_completed` function takes a group of future objects and returns an iterator that yields futures as they complete. Any futures that are completed before the as_completed function was called will be yielded first.

```python
concurrent.futures.as_completed(fs, timeout=None)
```

### Asynchronous Programming
- The single-threaded asynchronous model is a concurrency model in which a single thread achieves concurrency by interleaving the execution of the multiple tasks.

#### Cooperative multitasking with Event Loops and Coroutines
- The event loop in python is responsible for scheduling and executing tasks and callbacks in response to IO events, system events and application context changes.

##### Python Event Loop
- To get an instance of an event loop, we call the `asyncio.get_event_loop`. This method returns an object of abstract_event_loop.
- After we get the `event_loop` instance, we can start it by calling the `abstractEventloop.run_forever()` method, or the `AbstractEventLoop.run_until_complete(future)` method.
- If we start the event loop by calling `run_forever`, we can stop it by calling `AbstarctEventloop.stop()`. This causes the event loop to exit at the next suitable opportunity.
- Once the event loop is in stop state then we can close it by calling close. `AbstractEventLoop.close()`

















