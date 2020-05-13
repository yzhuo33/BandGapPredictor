# BandGapPredictor
Predict the bandgap energy for inorganic materials

This package provides a machine learning model trained based on experimetally measurements to predict the bandgap energy (Eg) for inorganic materials via the command-line.

put compositions that you want to predict in the "to_predict" file, then run "model_uniform" in Jupyter. After running the file, you will get a new .xlsx file name "predicted", where the first column is your composition and the second is predicted Eg.

## Table of Contents

- [Citations](#citations)
- [Prerequisites](#prerequisites)
- [Usage](#usage)
  - [1 Relative permittivity prediction](#1-relative-permittivity-prediction)
    - [1_1 Define a customized prediction set for relative permittivity](#1_1-define-a-customized-prediction-set-for-relative-permittivity)
    - [1_2 Predict relative permittivity](#1_2-predict-relative-permittivity)
  - [2 Centroid shift prediction](#2-centroid-shift-prediction)
    - [2_1 Define a customized prediction set for centroid shift](#2_1-define-a-customized-prediction-set-for-centroid-shift)
    - [2_2 Predict centroid shift](#2_2-predict-centroid-shift)
- [Authors](#authors)

## Citations

To cite relative permittivity and centroid shift predictions, please reference the following work:

Zhuo. Y, Hariyani. S, You. S, Dorenbos. P, and Brgoch. J, Machine learning 5d-level centroid shift of Ce<sup>3+</sup> inorganic phosphors, submitted.

##  Prerequisites

This package requires:

- [pymatgen](http://pymatgen.org)
- [XGBoost](https://xgboost.readthedocs.io/en/latest/#)
- [scikit-learn](http://scikit-learn.org/stable/)
- [pandas](https://pandas.pydata.org/pandas-docs/stable/index.html)
- [NumPy](https://docs.scipy.org/doc/numpy/index.html)
- [xlrd](https://xlrd.readthedocs.io/en/latest/index.html)

## Usage

Note: The centroid shift prediction needs the relative permittivity value as one of the inputs. If you have it ready, you can jump to [Section 2](#2-Centroid-shift-prediction). Or, you can get a predicted relative permittivity value following [Section 1](#1-Relative-permittivity-prediction).

### 1 Relative permittivity prediction

### 1_1 Define a customized prediction set for relative permittivity

You should create a `.xlsx` file named `c_pounds.xlsx`, in which the compositions that are of interest are listed in the first column with the header "`Composition`".

There is one [example of customized dataset](/examples) in the repository:`examples/c_pounds.xlsx`.

You can get compositional descriptors by:

```bash
python descriptor_generator.py
```

`descriptor_generator.py` will automatically read `elements.xlsx` and `c_pounds.xlsx` to generate descriptors. After running, you will get a `.xlsx` file named `to_predict_relative_permittivity.xlsx`. In this file, the first column is your composition followed by 85 columns of descriptors.

You also need to append another 13 structural descriptors to the compositional descriptors:
- space group number
- unit cell volume (nm<sup>3</sup>)
- density (Mg/m<sup>3</sup>)
- *a*/*b*
- *b*/*c*
- *c*/*a*
- alpha/beta
- beta/gamma
- gamma/alpha
- existance of inversion center (exist:1; nonexist:0)
- existance of polar axis (exist:1; nonexist:0)
- volume per *Z* (nm<sup>3</sup>)
- volume per atom (nm<sup>3</sup>)

This information could be extracted from crystallographic information files (CIFs) and inorganic cystal databases. There is one [example](/examples) of the final `to_predict_relative_permittivity.xlsx` file in the repository:`examples/to_predict_relative_permittivity.xlsx`.

### 1_2 Predict relative permittivity
Before getting a prediction, you will need to:

- [Prepare a customized dataset](#1_1-define-a-customized-prediction-set-for-relative-permittivity) named after `to_predict_relative_permittivity.xlsx` to store the composition-structure-property relations of interest.

Then, you can predict the relative permittivity by:

```bash
python relative_permittivity_predictor.py
```

`relative_permittivity_predictor.py` will automatically read `relative_permittivity_training_set.xlsx` and `to_predict_relative_permittivity.xlsx` to generate a prediction. You will then get a `predicted_relative_permittivity.xlsx` file in the same directory, in which the predicted relative_permittivity is provided next to the corresponding composition.

### 2 Centroid shift prediction

### 2_1 Define a customized prediction set for centroid shift

You should create a `.xlsx` file named `to_predict_centroid_shift.xlsx` in the format as:

| A | B | C | D | E | F | G | H | I |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| Composition | Relative permittivity | Avg. cation electronegativity | Avg. anion polarizability | R<sub>m</sub> | DeltaR (R<sub>m</sub>-R<sub>Ce</sub> | Avg. bond length | Coord. no. | Condensation |

There is one [example of customized dataset](/examples) in the repository:`examples/to_predict_centroid_shift.xlsx`.

### 2_2 Predict centroid shift
Before getting a prediction, you will need to:

- [Prepare a customized dataset](#2_1-define-a-customized-prediction-set-for-centroid-shift) named after `to_predict_centroid_shift.xlsx` to store the composition-structure-property relations of interest.

Then, you can predict the relative permittivity by:

```bash
python centroid_shift_predictor.py
```

`centroid_shift_predictor.py` will automatically read `centroid_shift_training_set.xlsx` and `to_predict_centroid_shift.xlsx` to generate a prediction. You will then get a `predicted_centroid_shift.xlsx` file in the same directory, in which the predicted centroid shift is provided next to the corresponding composition.

## Authors

This software was created by [Ya Zhuo](https://scholar.google.com/citations?user=WacJk1sAAAAJ&hl=en) who is advised by [Prof. Jakoah Brgoch](https://www.brgochchemistry.com/).
