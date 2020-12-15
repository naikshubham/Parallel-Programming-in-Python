# Parallel Programming with Dask in Python

### Querying Python interpreter's memory usage

```python
import psutil, os

def memory_footprint():
    '''Returns memory (in MB) being used by Python process'''
    mem = psutil.Process(os.getpid()).memory_info().rss
    return (mem / 1024 **2)
```

#### Allocating memory for an array
- We import numpy and we call memory footprint before doing any work. We then create a Numpy array x that requires 50MB of storage and then call memory footprint again to see the memory usage after.

```python
import numpy as np
before = memory_footprint()

N = (1024 **2) //8 # number of floats that fill 1 MB
x = np.random.randn(50 * N) # Random array filling 50MB
after = memory_footprint()
```

#### Allocating memory for a computation
- We then compute `x**2` & again compute memory use before & after. We find an additional 50 MB of RAM is acquired for this computation (even though the computed result `x**2` is not actually bound to an identifier)

```python
before = memory_footprint()
x**2 # computes, but doesn't bind result to a variable

after = memory_footprint()
print("Extra mem obtained: {} MB".format(after - before))
```

#### Querying array memory usage
- We can use a numpy array's `nbytes` attribute to find out its actual memory requirements.

```python
x.nbytes # memory footprint in bytes

x.nbytes // (1024 **2) # memory footprint in MB
```

#### Querying DataFrame memory usage
- We can also create a Pandas DataFrame `df` using `x`. The dataframe has an attribute `memory_usage()` that returns an integer series summarizing its memory footprint in bytes.

```python
df = pd.DataFrame(x)
df.memory_usage(index = False)

df.memory_usage(index=False) // (1024 **2)  # in MB
```

### Data in Chunks
- We've seen that available memory & storage restricts datasets that can be analyzed. A common strategy is to subdivide datasets into smaller parts.

#### Using pd.read_csv() with chunksize
- We'll use a 200,000 line file summarizing New york city cab rides from the first two weeks of 2013. Then, using `read_csv()` with the parameter `chunksize=50000`, the function returns an object we can iterate over.

```python
filename = 'NYC_taxi.csv'

for chunk in pd.read_csv(filename, chunksize=50000):
    print('type: %s shape %s', % (type(chunk), chunk.shape))
```

- The loop variable `chunk` takes on the values of four DataFrames in succession, each having 50k lines.

#### Examining a chunk
- The loop variable chunk has standard DataFrame attributes like shape, info.

#### Filtering a chunk
- We can construct a logical seres is_long_trip that is True whereever the trip time exceeds 1200 seconds (or 20 minutes).

```python
is_long_trip = (chunk.trip_time_in_secs > 1200)

chunk.loc[is_long_trip].shape  # filter rows where this condition holds
```

#### Chunking and filtering together
- Embed this filtering logic within a function `filter_is_long_trip` that accepts a dataframe as input and returns a dataframe whose rows correspond to trips over 20 mins.
- Next, we can make a list of DataFrames called chunks by iterating over the output of read_csv, this time using chunks of 1,000 lines.
- We can use a list comprehension to build the list

```python
def filter_is_long_trip(data):
    """Returns dataframe filtering trips longer than 20 mins"""
    is_long_trip = (data.trip_time_in_secs > 1200)
    return data.loc[is_long_trip]
    
chunks = []
for chunk in pd.read_csv(filename, chunksize=1000):
    chunks.append(filter_is_long_trip(chunk))
    
# OR

chunks = [filter_is_long_trip(chunk) for chunk in pd.read_csv(filename, chunksize=1000)]
```

#### Using pd.concat()
- Use another list comprehension called lengths to see that the dataframes in the list chunks each have around 100 to 200 rows (rather than 1000 rows in the unfiltered chunks)
- The resulting DataFrame long_trips_df has almost 22,000 rows (much fewer than the original 200,000)

```python
len(chunks)

lengths = [len(chunk) for chunk in chunks]
lengths[-5:] # each has ~100 rows

long_trips_df = pd.concat(chunks)
long_trips_df.shape
```

### Managing Data with Generators

#### Filtering in a list comprehension
- If we replace the enclosing brackets with parentheses in a list comprehension, the result is a generator expression.

#### Filtering & summing with generators
- Generator expressions resemble comprehensions, **but use lazy evaluation**. **This means that elements are generated one-at-a-time, so they are never in memory simultaneously**. This is extremely helpful when operating at the limits of available memory.
- We can quickly build another generator distances whose elements are totals of the trip_distance column from each chunk.
- No actual computation is done until we iterate over the chained generators explicitly (in this case, by applying the function sum to distances).
- **No reading or work is done until the very last step**.

```python
chunks = (filter_is_long_trip(chunk) for chunk in pd.read_csv(filename, chunksize=1000))

distances = (chunk['trip_distance'].sum() for chunk in chunks)

sum(distances)
```

#### Examining consumed generators
- The generators chunks & distances persist after the computation. However, they have been consumed at this point.That is, trying to next function on either produces a `StopIteration exception` (which tells users that the generators is exhausted).

#### Generators to read many files
- Rather than having one large file to read in chunks, we have many large individual files that cannot fit in memory simultaneously.

```python
template = 'yellow_tripdata_2015-{:02d}.csv'
filenames = (template.format(k) for k in range(1,13)) # Generator
for fname in filenames:
    print(fname) # examine contents
```

#### Examining a sample DataFrame

```python
df = pd.read_csv('yellow_tripdata.csv', parse_dates=[1,2]) # force columns 1 & 2 to be read as datetime objects
df.info() # columns deleted from output

# for this data, we have to calculate the trip duration explicilty
# we embed this calculation within a function, `count_long_trips` that filters trips longer than 20 mins, and counts the total number of trips and long trips

def count_long_trips(df):
    df['duration'] = (df.tpep_dropoff_datetime - df.tpep_pickup_datetime).dt.seconds
    is_long_trip = df.duration > 1200
    result_dict = {'n_long' : [sum(is_long_trip)],
                   'n_total': [len(df)]}
    return pd.DataFrame(result_dict)
```

#### Aggregating with Generators
- With the function count_long_trips in place, we can organize our work into a pipeline using generators.

```python
def count_long_trips(df):
    df['duration'] = (df.tpep_dropoff_datetime - df.tpep_pickup_datetime).dt.seconds
    is_long_trip = df.duration > 1200
    result_dict = {'n_long':[sum(is_long_trip)],
                   'n_total':[len(df)]}
    return pd.DataFrame(result_dict)
    
filenames = [template.format(k) for k in range(1, 13)]

# we create a generator dataframes to load the files listed in filenames one-by-one. we create another generator totals that applies count_long_trips to each DataFrame from dataframes.

filenames = [template.format(k) for k in range(1, 13)] # Listcomp
dataframes = (pd.read_csv(fname, parse_dates=[1,2]) for fname in filenames) # generator

totals = (count_long_trips(df) for df in dataframes) # Generator

annual_totals = sum(totals) # consumes generators
```

### Delaying Computation with Dask

#### Function composition

```python
from math import sqrt

def f(z):
    return sqrt(z + 4)
def g(y):
    return y - 3
def h(x):
    return x ** 2
    
x=4
y=h(x)
z=g(y)
w=f(z)
    
# above is equivalent to: f(g(h(x))) 
```

#### Deferring computation with `delayed`

```python
from dask import delayed
y = delayed(h)(x)
z = delayed(g)(y)
w = delayed(f)(z)
print(w)

w.compute() # computation occurs now
```

- Delayed is a higher-order function or a decorator function that maps an input function to another modified output function. The value of w, then, is delayed of f of delayed of g of delayed of h of 4.
- If we examine w, it is a dask Delayed object rather than a numerical value.
- The delayed decorator stalls the computation until the method compute() is invoked.

#### Visualizing a task graph
- The Dask Delayed object has another method `visualize()` that displays a task graph in some IPython shells. This linear graph shows the execution sequence and flow of data for this computation.

#### Renaming decorated functions

```python
f = delayed(f)
g = delayed(g)
h = delayed(h)
w = f(g(h(4)))

w.compute() 
```

- The result is the same, but the functions f,g,h are now decorated permanently by delayed. This means they always return Delayed objects that defer computation until the compute() method is called.

#### Using decorator @-notation

```python
def f(x):
    return sqrt(x+4)
f = delayed(f)

@delayed #equivalent to definition in above 2 cells
def f(x):
    return sqrt(x+4)
```

- Here, the @ symbol means "apply the decorator function delayed to the function described below and bind that decorated function to the name f".

#### Deferring Computation with Loops
- Let's use the delayed decorator with some new functions increment, double & add.

```python
@delayed                    
def increment(x):
    return x+1
@delayed
def double(x):
    return 2*x
@delayed
def add(x,y):
    return x+y
    
data=[1,2,3,4,5]
output=[]
for x in data:
    a = increment(x)
    b = double(x)
    c = add(a,b)
    output.append(c)
total = sum(output)
```

- The dependencies are little trickier - c depends on a & b within each iteration and its computed value is appended to the list output. The final result `total` is a `Delayed` object and `output` is a list of intermediate `Delayed` objects. 
- Dask uses a variety of heuristic schedulers for complicated execution sequences like this. The scheduler automatically assigns tasks in parallel to extra threads or processes. In particular, **Dask users do not have to decompose computations themselves**.

#### Aggregating with delayed Functions
- Repeat yellow cab ride data analysis using Dask instead of generators.

```python
template = 'yellow_tripdata_2015-{:02d}.csv'
filenames = [template.format(k) for k in range(1,13)]

@delayed
def count_long_trips(df):
    df['duration'] = (df.tpep_dropoff_datetime - df.tpep_pickup_datetime).dt.seconds
    is_long_trip = df.duration > 1200
    result_dict = {'n_long':[sum(is_long_trip)],
                   'n_total':[len(df)]}
    return pd.DataFrame(result_dict)
    
@delayed
def read_file(fname):
    return pd.read_csv(fname, parse_dates=[1,2])
```

- Define function `count_long_trip` adding the @delayed decorator.

#### Computing fraction of long trips with `delayed` functions

```python
totals = [count_long_trips(read_file(fname)) for fname in filenames]
annual_totals = sum(totals)
annual_totals = annual_totals.compute()

fraction = annual_totals['n_long']/annual_totals['n_total']
print(fraction)
```

### Chunking Arrays in Dask
- Dask arrays : A common data structure in Dask. The Dask library actually has many tools beyond the delayed decorator.

#### Working with Numpy arrays
- Dask arrays : as extensions of Numpy arrays.

```python
import numpy as np
a = np.random.rand(10000)
print(a.shape, a.dtype)
print(a.sum())
print(a.mean())
```

#### Working with Dask arrays
- The function `from_array` constructs a Dask array from a Numpy array. It requires an argument chunks; this is the number of elements in each piece of a. The chunk size here is length of a divided by 4 (2500).

```python
import dask.array as da

a_dask = da.from_array(a, chunks=len(a) // 4)
a_dask.chunks
```

#### Aggregating in chunks
- Notice each loop iteration is independent so that they can be executed in parallel if possible.

```python
n_chunks = 4
chunk_size = len(a) // n_chunks
result = 0 # accumulate sum
for k in range(n_chunks):
    offset = k * chunk_size # track offset
    a_chunk = a[offset:offset + chunk_size] # slice chunk
    result += a_chunk.sum()
print(result)    
```

#### Aggregating with Dask arrays
- To start, create a Dask array with an appropriate chunk-size. A single call to sum yields an unevaluated Dask object. We don't need to compute offsets or slice chunks explicitly; the Dask array method does that.
- Calling `.compute()` forces evaluation of the sum(in parallel if possible).
- We can also view the associate Task Graph using visualize(). The option `randir='LR'` forces a horizontal layout. The four rectangles in the middle represents the chunks of data. As the data flows from left to right, the sum method is invoked on each chunk. The Dask scheduler can assign work to multiple threads or processes concurrently if available.

```python
a_dask = da.from_array(a, chunks=len(a)//n_chunks)
result = a_dask.sum()
result

print(result.compute())

result.visualize(rankdir='LR')  # vizualize
```

#### Dask array methods/attributes
- Dask arrays share many attributes with numpy arrays. `shape , ndim, nbytes, dtype, size`.
- Most numpy **array aggregations** are also available for Dask arrays `max, min, mean, std, var, sum, prod` etc
- Dask **array transformations** like `reshape, repeat, stack, flatten, transpose, T` are also available.
- Also, many Numpy mathematical operations and universal functions also work with Dask arrays. `round, real, imag, conj, dot` etc.

#### Timing array computations

```python
import h5py, time

with h5py.File('dist.hdf5', 'r') as dset:
    dist = dset['dist'][:]
    
dist_dask8 = da.from_array(dist, chunks=dist.shape[0]//8)
t_start = time.time()
mean8 = dist_dask8.mean().compute()
t_end = time.time()
t_elapsed = (t_end - t_start) * 1000
```

### Computing with Multidimensional Arrays

```python
# a numpy array of time series data

import numpy as np
time_series = np.loadtxt('max_temps.csv', dtype=np.int64)

# reshaping time series data
table = time_series.reshape((3,7)) # reshape row-wise by default

reshaping  : getting the correct order
time_series.reshape((7, 3), order='F')

# the option order='F' forces column-wise ordering(with successive days down the columns & successive weeks along the rows correctly)
```

#### Indexing in multiple dimensions
- Indexing multi dimensional arrays requires only one set of brackets `table[0, 4]`
- `table[1, 2:5]` # values from week 1, days 2,3 & 4
- `table[0::2, ::3]` # values from weeks 0 & 2, Days 0, 3 & 6 (every second row and every 3rd col)

#### Aggregating multidimensional arrays
- `table.mean(axis=(0, 1)) # mean of rows, then columns`

#### Connecting with Dask

```python
data = np.loadtxt('', usecols=(1,2,3,4), dtype=np.int64)

data_dask = da.from_array(data, chunks=(366, 2))
result = data_dask.std(axis=0) 
result.compute()
```









































































































