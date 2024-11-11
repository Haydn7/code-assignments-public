# Solution and Workflow Overview

## Thought process for Problem 1: Multiprocessing
* Write test scripts to check size and values of data and arrays for single and multi process loaders match
* Get the number of rows and headers from the csv.
* Create blank dataset and arrays to store the merged data in.
* Determines rows for each processor
* Each worker loads data for given rows, acquires lock, copies thread data to merged global data.