import functools
calls = 0

#@functools.lru_cache(maxsize=None)
def comparator(a, b):
    global calls
    calls += 1
    return a.count("A") >= b.count("A")

def partition(array, low, high):
 
    pivot = array[high]
 
    i = low - 1
 
    for j in range(low, high):
        if comparator(array[j], pivot):
 
            i = i + 1
 
            (array[i], array[j]) = (array[j], array[i])
 
    (array[i + 1], array[high]) = (array[high], array[i + 1])
 
    return i + 1
 
 
def quicksort(array, low=0, high=None):
    if high is None:
        high = len(array) - 1
    if low < high:
        pi = partition(array, low, high)
        quicksort(array, low, pi - 1)
        quicksort(array, pi + 1, high)

x = ['AA']*12
quicksort(x)
print(x)
print(calls)