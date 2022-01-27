# LC-MS²Struct

This package implements a [Structured Support Vector Machine (SSVM)](https://en.wikipedia.org/wiki/Structured_support_vector_machine) 
model for the molecule structure prediction of liquid chromatography (LC) tandem mass spectrometry data (MS²). This 
work is part of the publication:

**"Machine learning for joint structural annotation of metabolites using LC retention time order and tandem mass 
spectrometry data"**,

*Eric Bach, Emma L. Schymanski and Juho Rousu*, 2022


We consider the output of an LC-MS² experiment as *structured* output. The structure is thereby assumed to be 
imposed by the observed *retention orders* (RO) of the MS features, i.e. MS¹-information, MS²-spectrum, and 
retention time (RT). We assume, that for each MS feature a set of potential molecular structures, the so-called 
candidate set, can be generated. The idea is to predict a ranking of the candidate structures associated with *each* 
features. The SSVM framework allows us to predict rankings that are not independent of each other, but are taking 
into account the observed ROs, which are assumed to give *structure* respectively additional constraints which 
improve the ranking. 

## Installation

That's how you install the package: 

1) Clone the package and change to the directory:
```bash
git clone https://github.com/aalto-ics-kepaco/msms_rt_ssvm
cd msms_rt_ssvm
```

2) Create a **conda** environment and install dependencies:
```bash
conda env create -f environment.yml
conda activate lcms2struct
```

3) Install the package:
```bash
pip install .
```

4) Leave the package directory:
```bash
cd ..  
```

5) Clone the package-dependency "[msmsrt_scorer](https://github.com/aalto-ics-kepaco/msms_rt_score_integration)", 
   implementing the max-marginal (see Paper) inference, and change to the directory:
```bash
git clone https://github.com/aalto-ics-kepaco/msms_rt_score_integration
cd msms_rt_score_integration
```

6) Install the "msmsrt_scorer" package (it is assumed that the conda environment is active):
```bash
pip install .
```

7) (Optional) Change back to the msms_rt_ssvm directory and test the package:
```bash
cd ../msms_rt_ssvm

# Unpack test databases
gunzip --keep ssvm/tests/Bach2020_test_db.sqlite.gz
gunzip --keep ssvm/tests/Massbank_test_db.sqlite.gz

# Run the tests
python -m unittest discover -s ssvm/tests -p 'unittests*.py'

## Expected output ##
# .............s................s.....................s...................s.....s..................................s......
# ----------------------------------------------------------------------
# Ran 121 tests in 99.599s
# 
# OK (skipped=6)
```

All code was developed and tested in a Linux environment. 

## Usage

Example usages of the package can be found the [repository of the experiments](https://github.com/aalto-ics-kepaco/lcms2struct_experiments)
done for the manuscript.

## Cite the package

If you use this package, please cite:

```bibtex
TODO 
```