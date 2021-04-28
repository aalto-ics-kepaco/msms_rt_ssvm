# Version History

## v1.0.3 (Patch)

Small performance improvement for the MinMax kernel computation

- counting fingerprint matrices are returned with "dtype=int"
- specific minmax-kernel function using 'ufunc' for integer matrices (no type-conversion)

## v1.0.2 (Patch)

Processing large candidate sets occasionally caused the SSVM library to crash due to "MemoryError". That was due to a 
large matrix that is allocated in the '_I_feat_rsvm' function, which computes the constant part of the candidate scores
against the training data. 

- Wrapped the call of '_I_feat_rsvm' into a try-except environment to load an alternative implementation in case of a
  MemoryError
- Alternative implementation uses loops to go over all training data.
- Test added to compare the output of '_I_feat_rsvm' and '_I_feat_rsvm__FOR_LARGE_MEMORY'

## v1.0.1 (Patch)

The static top-k scoring function ('_topk_score' in SSVM class) can be used without labeled spectra sequence object. 
This requires the marginals dictionary to hold the information about the index of the correct candidate.

## v1.0.0

### Default Behavior

- MS2 scores are normalized to (0, 1] considering **all** scores within a sequence
- **Log-transformation** is applied to the MS2 scores
- The **average node** (1 / |V|) and **edge** (1 / |E|) **scores** are used
- There is **global candidate set** used for each spectrum during training  
- A **single spanning tree** is used for training and scoring phase. 

## Semantic Versioning Scheme

Version numbers of this package are of the following form: **vMajor.Minor.Patch**. A version is always tied to a 
commit on the **master** branch and typically indicated using git-tag. 

### Major

The major version when the code base has changes such that re-running the *publication* experiments is required as 
it is expected that the results will change. The following are example changes that could trigger the major version 
to increase:

- The solver of Structure Support Vector Machine (SSVM) changes or a bug was fixed.
- The training sequence strategy changes 
- ...

### Minor

When a new (additional) feature is implemented, the minor version increases. The new feature **should not require**
previously run publication experiments to be re-run. Example features for which the minor version increases:

- Implementing support for multiple spanning trees in training and scoring phase
- Support for new molecular representations
- Performance improvements
- Implement support for new experimental data
- ...

### Patch

Bug fixes typically cause the patch version to increase. This considers bugs that are in the production code, that 
means on the master branch, and that would prevent any experiment to even finish. Patches are implemented using a 
hot-fix branch from the master branch. 

- An experiment could not finish due to a memory error, or a small bug in the code
- ...

### Version of the results

The experiment results are stored in a sub-directory indicating the **major version**. The triton scripts, used to 
produce publication ready results, always install the latest version "minor.patch" version of the selected major 
version. For example, lets say we have the following tags on the master branch: *v1.0.0, v1.0.1, v1.1.0, v1.1.1*, and 
*v2.0.0, v2.0.1*. If we specify "v1", then the triton script will load the code with the version v1.1.1 and score in 
the sub-directory "v1". 

##### Potential exceptions

The triton scripts can actually be used such that they store the results in "v1.1", as grep is used to find the best 
fitting master branch version.

### Some further considerations

##### Bug fixes should consider the current experiments

As bug fixes are done on the master branch, they typically should consider the current experiment. Current experiment 
means for example redoing the Bioinformatics experiments using the SSVM. Parallel to that, we can develop the 
experiments (and the related code) on the develop branch. If there are bug fixes needed for that developments, but 
they do not affect code on the master branch they should not be implemented there. 

##### Try to run only one experiment at the time using the master branch code

##### Default parameter should be investigated on the develop branch
