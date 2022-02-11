# Implementation details and design decisions

Here we describe some design decisions of our library.

## Candidate databases

In the ```data_stuctures.py``` file, an SQLite database (DB) wrapper is implemented, which allows to access the 
relevant information for the model training via a standardized interface. This interface is not entirely transparent 
on the internal structure of the DB, but added this feature isn't that complicated. More information on the 
currently supported layout can be found [here](https://github.com/aalto-ics-kepaco/lcms2struct_experiments/blob/main/data/DB_README.md).  

The main idea of having a wrapper around the DB is to allow:
- loading MS² candidate scores
- getting random candidate subsets for the Structured Support Vector Machine (SSVM) training (speedup)
- loading molecular feature representations
- etc.

To see, what functionality is available, the reader is asked to read the source-code and its inline documentation. 

In order to apply the SSVM to new data, which was not used in our experiments, or support a different input 
data structure, such as CSV-files, one can use the abstract functions in the DB wrapper as starting point for the 
development of wrappers for other file formats. 