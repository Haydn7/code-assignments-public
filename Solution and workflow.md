# Solution and Workflow Overview

## Problem 1: Multiprocessing
### Thought process
* Write test scripts to check size and values of data and arrays for single and multi process loaders match
* Get the number of rows and headers from the csv.
* Create blank dataset and arrays to store the merged data in.
* Determines rows for each processor.
* Each worker loads data for given rows, acquires lock, copies thread data to merged global data.

### Uses of AI
* The syntax for asserting torch arrays and datasets are equal.
* Setting up the correct datatypes for the global pandas dataframe to unit test failures.

## Problem 2: Neural Network Implementation
### Thought process
* Write test scripts to compare the output of the torch autograd with manually calculated values.
* Write a helper class to calculate and cache autograd gradients of each layer.
* Backprop of linear layers straightforward using einops.
