# BandGapPredictor
Predict the bandgap energy for inorganic materials

This package provides a machine learning model trained based on experimetally measurements to predict the bandgap energy (Eg) for inorganic materials via the command-line.

## Table of Contents

- [Citations](#citations)
- [Prerequisites](#prerequisites)
- [Usage](#usage)
  - [Define a customized prediction set](#define-a-customized-prediction-set)
  - [Predict bandgap energy](#predict-bandgap-energy)
- [Authors](#authors)

## Citations

To cite relative permittivity and centroid shift predictions, please reference the following work:

Zhuo. Y, Mansouri Tehrani., and Brgoch. J, Predicting the band gaps of inorganic solids by machine learning, *J. Phys. Chem. Lett.* **2018**, 9, 1668-1673.

##  Prerequisites

This package requires:

- [pymatgen](http://pymatgen.org)
- [scikit-learn](http://scikit-learn.org/stable/)
- [pandas](https://pandas.pydata.org/pandas-docs/stable/index.html)
- [NumPy](https://docs.scipy.org/doc/numpy/index.html)
- [xlrd](https://xlrd.readthedocs.io/en/latest/index.html)

## Usage

### Define a customized prediction set

You should create a `.xlsx` file named `to_predict.xlsx`, in which the compositions that are of interest are listed in the first column with the header "`Composition`". There is an example of the `to_predict.xlsx` file in the repository.

### Predict bandgap energy

You can get the Eg prediction by:

```bash
python Eg_model.py
```

`Eg_model.py` will automatically read `elements.xlsx`, `Training_Set.xlsx`,and `c_pounds.xlsx` to generate a prediction. After running, you will get a `.xlsx` file named `predicted.xlsx` in the same directory, in which the predicted Eg is provided next to the corresponding composition.

## Authors

This software was created by [Ya Zhuo](https://scholar.google.com/citations?user=WacJk1sAAAAJ&hl=en) who is advised by [Prof. Jakoah Brgoch](https://www.brgochchemistry.com/).
