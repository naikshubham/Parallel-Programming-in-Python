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

### Analyzing Weather Data
- `HDF5 format` : is a hierarchical file format for storing and managing data.

```python
import h5py # module for reading hdf5 files

data_store = h5py.File('tmax.2008.hdf5')
for key in data_store.keys(): # iterate over keys
    print(key)
    
# extract dask array from HDF5
data = data_store['tmax'] # bind to data for introspection
type(data)

# wrap data in dask array with 3-d chunks of size 1,444,922 one for each month
# no data is read yet
import dask.array as da
data_dask = da.from_array(data, chunks=(1, 444, 922))
```

#### Aggregating while ignoring NaNs
- Applying `min()` method returns an unevaluated Dask Array. We can force evaluation using the `compute()` method. Unfortunately, this yeilds `nan (or Not a number)` when there are missing values.
- Both Numpy & Dask have nan-aware aggregation utilities. For instance the function `nanmin` returns the minimum array entry, ignoring nan values. Thus, we can capture min & max values from data_dask (excluding nans) using `nanmin & nanmax`

```python
data_dask.min()  # yields unevaluated Dask array

data_dask.min().compute() # force computation

da.nanmin(data_dask).compute() # ignoring nans

lo = da.nanmin(data_dask).compute()
hi = da.nanmax(data_dask).compute()
print(lo, hi)
```

#### Producing a visualization of data_dask
- Code to produce the preceding image. Import pyplot & create a 4 by 3 grid of subplots called panels. Loop over panels.flatten, each time slicing from data_dask & plotting the slice with imshow.
- The arguments vmin=lo & vmax=hi ensure a common colormap scale.

```python
N_months = data_dask.shape[0]  # no if images
import matplotlib.pyplot as plt
fig, panels = plt.subplots(nrows=4, ncols=3)

for month, panel in zip(range(N_months), panels.flatten()):
    im = panel.imshow(data_dask[month, :, :],
                      origin='lower',
                      vmin=lo, vmax=hi)
    panel.set_title('2008--{:02d}'.format(month+1))
    panel.axis('off')

plt.suptitle('Monthly avg (max. daily temp[C])')
plt.colorbar(im, ax=panels.ravel().tolist())
plt.show()
```

#### Stacking arrays
- We'll have to stack dask arrays in our analysis

```python
import numpy as np
a = np.ones(3) 
b = 2 * a
c = 3 * a
print(a, b, c)

np.stack([a, b]) # makes 2D array of shape (2, 3) default is axis=0

np.stack([a, b], axis=1) # makes 2D array of shape (3,2)

# Stacking 1-d arrays
X = np.stack([a, b])
Y = np.stack([b, c])
Z = np.stack([c, a])

# Stacking 2-d arrays
np.stack([X, Y, Z)] # makes 3D array of shape (3, 2, 3)
```


### Using Dask DataFrames
- The Dask Dataframes is a delayed version of pandas DataFrame, just as the Dask array is a delayed version of the Numpy Array. By convention, dask.dataframe is imported with the alias dd.
- `dd.read_csv()` : accepts single filename or glob pattern ( with wildcard *) to match filename patterns & concatenate dataframes.
- `dd.read_csv()` : does not read files immediately until `dataframe.compute()` method is invoked.
- Dask dataframes can contain more data than it can fit in available memory.

#### Reading multiple csv files
- The data from all files is automatically concatenated into a single Dask dataframe. The dask dataframe has many methods in common with pandas Dataframe; `head and tail`.
- `.head() & .tail()` do not require that we invoke `.compute()`. This is delibrate because, in most cases, no parallelism is needed to examine the leading or trailing rows from a Dask DataFrame. Most other Dask DataFrame methods, however, do need the compute method for evaluation.

```python
import dask.dataframe as dd

transactions = dd.read_csv('*csv')
```

#### Building delayed pipelines
- `.loc` accessor can use the series `is_wendy` to filter the rows corresponding to wendy's transactions and the amount column from the Dask DataFrame transactions.
- The result `wendy_amounts` is a Dask series of type int64 that remains unevaluated until `.compute()` is invoked. We can then add up individual transactions for 2016 using the series `.sum()` method. wendy_diff is a Dask scalar, when `.compute()` is called, the result is a single Numpy int64.
- We can visualize a *task graph* summarizing this entire pipeline with the visualize method. The output is a *directed acyclic task graph* describing flow of data- the rectangles- through functions-the-circles- from left to right. 

```python
is_wendy = (transactions['names'] == 'Wendy')
wendy_amounts = transactions.loc[is_wendy, 'amount']

wendy_diff = wendy_amounts.sum()
wendy_diff.visualize(rankdir='LR')
```

#### Compatibility with Pandas API
- Some features of Pandas DataFrames are not available with Dask DataFrames. For instance, excel files & some compressed file formats(xls, zip, gz) are not supported as of version 0.15.
- Other task like sorting are genuinely hard to do in parallel. On the other hand, the `.loc` accessor works just as in pandas. The same is true of setting & resetting the index.
- aggregations : sum(), mean(), std(), min(), max().
- Many other parts of the Pandas DataFrame API carry over to Dask, for instance `groupbys & datetime conversions`.


#### Dask or Pandas?
- How big is dataset? How much RAM available? How many threads/cores/CPUs available?
- Are Pandas computations/ formats supported in Dask API?
- Is computation I/O-bound (disk-intensive) or CPU-bound(processor intensive)?
- *Best use case for Dask* : Computations from Pandas API available in Dask and problem size close to limits of RAM, fits on disk.

### Building Dask Bags & Globbing
- Upto, we've used Dask arrays & dataframes to work with structured, rectangular data. For messy unstructured datasets the dask bag is a convenient, heterogeneous, list-like data structure.

#### Sequence to bags
- To construct a Dask Bag, lets start with a Python list containing other containers: lists, dictionaries, and strings.
- Dask bag converts the nested containers to a Dask Bag using the function `from_sequence`. Using the Dask bag's method `count`, we can count the number of elements of the bag.
- The methods any and all evaluate to True and False exactly as the corresponding Numpy methods would.

```python
nested_containers = [[0,1,2,3], {}, [6.5, 3.14], 'Python', {'version':3}, '']

import dask.bag as db
the_bag = db.from_sequence(nested_containers)
the_bag.count()

the_bag.any(), the_bag.all()
```

#### Reading the text files
- The Dask Bag is designed to work with messy or unstructured files, most often raw ascii text files. The `read_txt` function reads a single file or collection of files line-by-line into a dask bag.
- We can use the take method to inspect the contents of the bag. Invoking `take(1)` returns a tuple. The tuple has single element, the first line of the file. Invoking `take(3)`, then pulls a tuple with 3 elements, the first 3 lines of the file.

```python
import dask.bag as db
zen = db.read_txt('zen')
taken = zen.take(1)
```

### Functional Approaches using Dask Bags
- Functions replacing loops with: `map, filter, reduce`. Map operations also works with dask bags. The computed result is a list, not a Dask bag, so we should be wary of limits of available memory.

```python
# using map

def squared(x):
    return x ** 2
    
squares = map(squared, [1,2,3,4,5,6])
squares = list(squares)

# using filter
# this is a boolean values function that is True or False according to whether its input is even

def is_even(x):
    return x % 2 == 0
evens = filter(is_even, [1,2,3,4,5,6])
list(evens)

# using dask.bag.map
import dask.bag as db
numbers = db.from_sequence([1,2,3,4,5,6])
squares = numbers.map(squared)

# no computing occurs until `.compute()` is invoked
result = squares.compute()

# using dask.bag.filter
numbers = db.from_sequence([1,2,3,4,5,6])
evens = numbers.filter(is_even)
evens.compute()

# using .str & string methods
zen = db.read_text('zen.txt')
uppercase = zen.str.upper()
uppercase.take(1)
```

#### A bigger example

```python
def load(k):
    template = 'yellow_tripdata_2015-{:02d}.csv'
    return pd.read_csv(template.format(k))
def average(df):
    return df['total_amount'].mean()
def total(df):
    return df['total_amount'].sum()

data = db.from_sequence(range(1,13)).map(load)
totals = data.map(total)
averages = data.map(average)
totals.compute()
averages.compute()
```

#### Reductions (aggregations)

```python
t_sum, t_min, t_max = totals.sum(), totals.min(), totals.max()
t_mean, t_std = totals.mean(), totals.std()
stats = [t_sum, t_min, t_max, t_mean, t_std]
[s.compute() for s in stats]

# a single call to dask.compute takes less time bcz dask can optimize disk reads & schedule parallel graph execution

dask.compute(t_sum, t_min, t_max, t_mean, t_std)
```

### Case study : Analyzing Flight Delays
- Search for correlations between flight delays and weather events at selected airports. Dask dataframes can be constructed using single file or from many files using glob.

#### Limitations of Dask DataFrames
- There's currently no native Dask support for Excel, gzip & some other useful file formats.
- Cleaning files independently when globbing is tricky. And nested subdirectories are tricky with glob.
- Example if we have `accounts/Alice.csv and acounts/Bob.csv`. Account holder name is filename but its not recorded within the file itself.

### Reading/cleaning in a function
- We need to build a large dataframe combining these files that preserves the account holders name.

```python
import pandas as pd
from dask import delayed

# create a function pipeline decorated by delayed
@delayed
def pipeline(filename, account_name):
    df = pd.read_csv(filename)
    df['account_name'] = account_name
    return df
    
# using dd.from_delayed()    
# iterate over 3 acc holders    
delayed_dfs = []

# within the loop we construct a filepath and use pipeline to accumulate a list of delayed_dfs of delayed pandas dataframes
for account in ['Bob','Alice','Dave']:
    fname = 'accounts/{}.csv'.format(account)
    delayed_dfs.append(pipeline(fname, account))
    
import dask.dataframe as dd
dask_df = dd.from_delayed(delayed_dfs)
dask_df['amount'].mean().compute()
```

#### Flight delays data
- We are interested in `WEATHER_DELAY` column that tells the no of minutes by which flight was delayed due to weather. 

```python
df = pd.read_csv('flightdelays-2016-1.csv')

# replacing values
new_series = series.replace(6, np.nan)
```

- Pandas has a function `to_numeric` that can convert a series to floating point values.The option `errors=coerce` is used to force the non-numeric characters to be translated to nans.

```python
new_series = pd.to_numeric(series, errors='coerce')
```

### Merging & Persisting DataFrames
- A convenient way to stitch together dataframes with overlapping columns. Both pandas and dask dataframes have a merge method.

```python
left_df.merge(right_df, left_on=['cat_left'], right_on=['cat_right'], how='inner')
```

#### Repeated reads & performance

```python
import dask.dataframe as dd
df = dd.read_csv('flightdelays-2016-*.csv')
%time print(df.WEATHER_DELAY.mean().compute())

%time print(df.WEATHER_DELAY.std().compute())

%time print(df.WEATHER_DELAY.count().compute())
```

- **The bottleneck in this 3 computations is in repeatedly reading the data from disk every time we execute the compute() method. If the data is too large to fit in memory, these repeated reads are unavoidable***

#### Using persistence
- When the DataFrame does fit in the memory, Dask DataFrames have a method `persist` to keep the intermediate state of the DataFrame in memory.






































































































