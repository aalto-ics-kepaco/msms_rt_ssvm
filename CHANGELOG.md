# Version History

## v2.11.2 (Patch)

- fix in the XLogP3 loader 
- raise an "Imputation error" if all XLogP3 values are NaN (missing)
- new tests added

## v2.11.1 (Patch)

- fix in conda environment

## v2.11.0 (clean up)

- code cleanup only
- remove dependency on tensorflow
- update environment.yml and setup.py

## v2.10.0 (New Feature)

- allow loading of XLogP3 values for the MassBank DB
- remove some old source code

## v2.9.0 (New feature)

- (labeled) sequences can now request the label space with "return_inchikeys"
- (labeled) sequences can now be sorted by the retention time of the spectra upon object construction
- make (labeled) sequences iterable

## v2.8.0 (New feature)

- return inchikey information when loading the molecule labels of the label-space (if requested)

## v2.7.0 (New feature)

- return MAP estimate together with the max-marginals

## v2.6.0 (New feature)

### Support for multiple MS scores

- Allow loading of combination of MS2 scores 
- Scores of different MS2 scorers are combined using a weighted sum

### Support for constant MS score

- for testing purposes a non-informative MS score can be used by specifying "CONST_MS_SCORE"

## v2.5.0 (New feature)

- Implement a Zero-One loss for the margin computation
- The molecule identifier (label) is used to compute the loss

## v2.4.0 (New feature)

- the "k" parameter for the scikit-learn "ndcg_score" function can be passed to the SSVM scoring function

## v2.3.2 (Patch)

- fix constant in MS2 score loading function when accessing directly via the LabeledSequence class

## v2.3.1 (Minor change)

- small performance improvement in the median heuristic 

## v2.3.0 (New feature)

- SSVM class now supports the RBF kernel --> scale parameter can be passed as argument to the SSVM class
- A new 'kernel_loss' (for the labels) was implemented
  - it used the kernel for the molecule features directly for the computation of the label-loss
  
### Minor changes

- Feature transformer in the candidate DB wrapper --> Implementation of getter- and setter-functions 
- Sequence samples now have a parameter storing all spectra IDs associated with it

## v2.2.0 (New feature)

- Added function to compute the RBF scale using the median heuristic

## v2.1.0 (New feature)

- Added support for the molecular descriptors in the MassBank DB
- Added support for feature transformers for the candidate DB wrapper
  - we use the Scikit learn framework for the transformer implementation
  - pipelines are supported
  - transformers respectively pipelines needs to be pre-fitted and passed as parameter to the candidate DB wrapper
  - transformations are applied when features are loaded from the DB

## v2.0.0

### Default Behavior

The default setup has been synchronized with the paper:

- MS2 scores are normalized to [0, 1] for each candidate set separately
- The **average node** (1 / |V|) and **edge** (1 / |E|) **scores** are used

### FIX in edge score normalization

- in the previous implementations only edge scores of the prediction sequence have been normalized
- training (model) sequence edge scores where (wrongly) not normalized
- that has been fixed in the prediction and line-search implementation

### No log-transformation of the node scores

- node potentials are defined by psi = exp(MS2 score). Therefore, the log(psi) = MS2 score
- MS2 scores are normalized to [0, 1]

## v1.2.4 (Minor improvements)

- Allow the computation of the baseline performance for top-k scoring (using max-marginals) based on the MS2 score only
- Candidate wrapper ensures that the requested MS2 scorer is available and raises an error otherwise

### Add debugging feature

- Maximum violating example can be chosen based on the node-scores only

## v1.2.3 (Minor improvements)

- Spectrum labels for "LabeledSequence" can be extracted from the spectrum objects 
  - labels can still be passed via an optional parameter 
  - label-key can be passed as well (default is 'molecule_id')
- Implementation of the generalized tanimoto label loss
- More unittests for the tanimoto label loss

## v1.2.2 (Patch)

### Fixes

- fix in sequence generator: sequence specific candidate sets need to check for the abstract random candidate set
  class
- fix in Structured SVM class: account for interface changes of the candidate DB wrapper class 
  ('_get_feature_dimension' -> '_get_d_feature')
  
### Minor changes

- add 'generalized_tanimoto' as kernel option for the Structured SVM class
- allow a molecular structure two appear more than ones in a sequence, e.g. if it appears in a dataset twice with 
  different adducts
  - change needed due to the MassBank dataset

## v1.2.1 (Minor improvements)

- improve doc-strings
- fix type-hinting

## v1.2.0 (New Feature)

### Support for the MassBank DB Layout

- candidate class wrapper have been implemented using abstract classes to enable easier support for new DB designs
- a new test database for the MassBank layout hase been added

## v1.1.4 (Patch)

- Candidate DB wrapper returns binary fingerprint matrices with dtype = float needed for the numba implementation of the
  tanimoto kernel.
- Remove "jit" decorator from tanimoto implementation  

## v1.1.3 (Patch)

- Candidate DB wrapper is now compatible with the binarized features (via feature transformation)
- How to parse the fingerprint strings is now determined from the fingerprint mode in the DB

## v1.1.2 (Minor improvement)

- Binary feature transformer implements ```__len__```

## v1.1.1 (Minor improvement)

- Binary feature transformer can now also filter empty feature dimension

## v1.1.0 (New feature)

### Encoding of counting fingerprints as binary vectors

- Feature transformer added to encode counting fingerprints as binary vectors
- Counting values are converted (roughly) like: 5 --> 1 1 1 1 1 
- Bin centers can be specified by the user defining which count is encoding separately
  - e.g. bin-centers [1, 4, 5]: cnt_fp = [0, 1, 15, 4] --> bin_fp = [0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0]
- depending on the bin-center definition the conversion is not loss-free in terms of molecule similarity 
- tanimoto kernel can be used to compute molecule similarity

### Faster Tanimoto Kernel using Numba

- added numba jit-decorator to the tanimoto kernel --> huge speedup

## v1.0.4 (Patch)

Include the latest changes to the into the SSVM flavor using sequence specific candidate sets.

- SSVM (sequence specific) updated with latest version of base SSVM class

## v1.0.3 (Minor improvement)

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
