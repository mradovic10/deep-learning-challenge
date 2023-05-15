# deep-learning-challenge
Module 21 Challenge

The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. Using machine learning, neural networks, and a CSV file containing more than 34,000 organizations that have received funding over the years, I created a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.

## Overview of the Analysis

The first step was preprocessing the data. The 'EIN' and 'NAME' columns were dropped for all models due to them not being quantifiable features of the data. For the third and fourth optimized models, two more columns were dropped as features: 'STATUS' and 'SPECIAL_CONSIDERATIONS'.

The 'APPLICATION_TYPE' and 'CLASSIFICATION' columns were processed so that any data below a notable value cutoff point were binned in an "Other" category to aid effective analyzation.

Next, all categorical data that held 'object' data types was converted to numeric using the 'get_dummies' method. From here, the data was split into 'features' and 'target' arrays, and then further split into 'training' and 'testing' datasets.

Finally, I scaled the data using 'StandardScaler' to put the finishing touch on the preprocessing stage. The data was ready to be compiled, trained, and evaluated using deep learning models.

## Results

The repository is divided up into five models: the initial one and then four attempts at optimizing the deep learning model. The folder for each model includes a Jupyter Notebook file that was downloaded from Google Colab (where the analysis was performed) and an HDF5 file. The following information summarizes each model:

* Initial Model:
    * Layers: 3
    * First hidden layer: 12 units, 'relu' activation function
    * Second hidden layer: 16 units, 'relu' activation function
    * Number of epochs: 50
    * Model accuracy: 0.726
    * Model loss: 0.554
    * Other adjustments: N/A
    * Target performance achieved: No

* Optimized Model 1:
    * Layers: 3
    * First hidden layer: 16 units, 'relu' activation function
    * Second hidden layer: 16 units, 'relu' activation function
    * Number of epochs: 100
    * Model accuracy: 0.729
    * Model loss: 0.553
    * Other adjustments: N/A
    * Target performance achieved: No

* Optimized Model 2:
    * Layers: 3
    * First hidden layer: 12 units, 'tanh' activation function
    * Second hidden layer: 16 units, 'tanh' activation function
    * Number of epochs: 100
    * Model accuracy: 0.728
    * Model loss: 0.553
    * Other adjustments: N/A
    * Target performance achieved: No

* Optimized Model 3:
    * Layers: 3
    * First hidden layer: 16 units, 'relu' activation function
    * Second hidden layer: 16 units, 'relu' activation function
    * Number of epochs: 50
    * Model accuracy: 0.729
    * Model loss: 0.555
    * Other adjustments: 'STATUS' and "SPECIAL_CONSIDERATIONS' columns dropped
    * Target performance achieved: No

* Optimized Model 4:
    * Layers: 3
    * First hidden layer: 16 units, 'relu' activation function
    * Second hidden layer: 16 units, 'relu' activation function
    * Number of epochs: 50
    * Model accuracy: 0.739
    * Model loss: 0.541
    * Other adjustments: 'STATUS' and "SPECIAL_CONSIDERATIONS' columns dropped; outliers dropped based on 'ASK_AMT' column
    * Target performance achieved: No

## Summary

The initial model and the first three optimized ones have very similar accuracy and loss scores. It was not until some outliers (based on the 'ASK_AMT' column) were dropped from the data during preprocessing that notable improvement was seen. The loss score went down and the accuracy score nearly reached 74%, which means that it was only 1% away from 75% target performance.

I believe that the removal of further outliers in the data could aid in improving performance of a deep learning model built to predict whether applicants will be successful if funded by Alphabet Soup. Regardless of differences in the five models included here, there is not much difference in their performance, with the exception of Optimized Model 4, which left out some outliers. This means that there has to be some further tweaking done with the dataset in order to better predict an applicant's success.