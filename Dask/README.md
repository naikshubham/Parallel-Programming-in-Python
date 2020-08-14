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

# we create a generator dataframes to load the files listed in filenames one-by-one. we create another generator 
```























































































