"""
Problem: Given a set of N coordinates as (X,Y) pairs,
we want to compute how many coordinates are within R meters of an (X,Y) centroid
where the distance metric is Euclidean.
Goal: Design and write a class that is able to solve instances of this problem.
The interface should be simple, documented,
and allow a typical developer to use your API to efficiently query coordinates and centroids
to find coordinate counts in proximity to centroids.
Using this class and the sample data provided to provide solutions to the following questions.
1. How many coordinates are within 5 meters of at least one of the centroids?
2. How many coordinates are within 10 meters of at least one of the centroids?
3. What is the minimum radius R such that 80 percent of the coordinates are within R
meters of at least one of K centroids?
4. Bonus: What is the maximum radius R such that the number of coordinates within a
distance strictly less than R of any centroid is at most 1000?
Files:
1. coordinates.csv: contains 1 million X,Y pairs with a header, units in meters
2. centroids.csv: contains 1000 cluster centroids as X,Y pairs with a header, units in meters
Deliverable:
1. State your assumptions. Provide direction to run your code and
to recreate the solutions to the questions.
This includes installing all the dependencies, specifying path,
or running the executable. Assume the developer executing
and validating your code using Linux distribution.
2. Please provide simple unit tests for your software.
3. Provide solutions along with runtimes and peak memory usage for each question.
4. Document the computation and memory complexity of each API call in your class as a
function of the K centroids and N coordinates.
"""

import pandas as pd
import numpy as np
import torch, sys
from scipy.spatial import KDTree
from datetime import datetime
import tracemalloc

class nn_kdtree:
    '''
    KDTree algorithm implemented with scipy by using
    class scipy.spatial.KDTree(
        data,
        leafsize=10,
        compact_nodes=True,
        copy_data=False,
        balanced_tree=True,
        boxsize=None)
    '''
    def kdtree(self, myfile):

        tracemalloc.start() #Compute peak memory allocation of tensor
        co_cen = torch.tensor((pd.read_csv(myfile, encoding="UTF-8")).values)
        #print("Memory: %s"%sys.getsizeof(co_cen.storage()))
        current, peak = tracemalloc.get_traced_memory() #a tuple: (current_mem: int, peak_mem: int)
        print(f"{current:0.2f}, {peak:0.2f}")
        tracemalloc.stop()
        return KDTree(co_cen)

    def coord_within_radius(self, tree, x, k=1):
        '''
            KDTree.query(x, k=1, eps=0, p=2, distance_upper_bound=inf, workers=1)
            The function returns:
                d   - float or array of floats
                i   - integer or array of integers
        '''
        start = datetime.now()
        tracemalloc.start()  # Compute peak memory allocation of tensor

        nearest_d, nearest_i = tree.query(x, k=k)
        current, peak = tracemalloc.get_traced_memory()  # a tuple: (current_mem: int, peak_mem: int)
        #   print(f"{current:0.2f}, {peak:0.2f}")
        print("current_memory = %0.2f peak_memory = %0.2f " % (current, peak))
        print("Execution time question 1= %f ms \n" % ((datetime.now() - start).total_seconds() * 1000))  # in millisecond
        tracemalloc.stop()

        return nearest_i

    def min_radius_with_percentage(self, tree, x, percentile=80):
        '''
            numpy.percentile(
            a,                          input array
            q,                          percentile or sequence of percentiles to compute
            axis=None,                  optional, default is along a flattened version of the array
            out=None,                   ndarray, output array to place the result
                                        optional
            overwrite_input=False,      If True, then allow the input array a to be modified
                                        by intermediate calculations, to save memory
                                        optional
            method='linear',            specifies the method to use for estimating the percentile
                                        optional
            keepdims=False,             bool, optional
            *,
            interpolation=None)
            The function returns        percentile (scalar or ndarray)
        '''

        start = datetime.now()
        tracemalloc.start()  # Compute peak memory allocation of tensor
        d_array, i_array = tree.query(x, k=1)
        p = np.percentile(d_array, percentile, interpolation='linear')
        current, peak = tracemalloc.get_traced_memory()  # a tuple: (current_mem: int, peak_mem: int)
        print("current_memory = %0.2f peak_memory = %0.2f " % (current, peak))
        print("Execution time question 3= %f ms \n" % ((datetime.now() - start).total_seconds() * 1000))  # in millisecond
        tracemalloc.stop()
        return p

    def max_radius_of_centroid(self, tree, x, k=1000):
        start = datetime.now()
        tracemalloc.start()  # Compute peak memory allocation of tensor
        nearest_d, nearest_i = tree.query(x, k=[k])
        m = nearest_d.max()
        current, peak = tracemalloc.get_traced_memory()  # a tuple: (current_mem: int, peak_mem: int)
        print("current_memory = %0.2f peak_memory = %0.2f " % (current, peak))
        print("Execution time question 4 = %f ms \n" % ((datetime.now() - start).total_seconds() * 1000))  # in millisecond
        tracemalloc.stop()
        return m

#test_code

def within_radius_i(kd, centroids, coordinates, i):

    coord = kd.coord_within_radius(centroids, coordinates.data, i)
    print("Number of Coordinates within %s= %s" % (i, len(coord)))


def test():
    kd = nn_kdtree()
    coordinates = kd.kdtree("/Users/macbookair/PycharmProjects/EatLab/coordinates.csv")
    centroids = kd.kdtree("/Users/macbookair/PycharmProjects/EatLab/centroids.csv")


    within_radius_i(kd, centroids, coordinates, 5)
    within_radius_i(kd, centroids, coordinates, 10)

    min_r = kd.min_radius_with_percentage(centroids, coordinates.data, 80)
    max_r = kd.max_radius_of_centroid(coordinates, centroids.data, 1000)

    print("Min radius = %s m" %min_r)
    print("Max radius = %s m" %max_r)

test()
