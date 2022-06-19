# Venture_Funding_With_Deep_Learning

## Overview of the Analysis

I'm working as a risk management associate at Alphabet Soup, a venture capital firm. Alphabet Soup’s business team receives many funding applications from startups every day. This team has asked me to help them create a model that predicts whether applicants will be successful if funded by Alphabet Soup.

The business team has given me a CSV file containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. The CSV file contains a variety of information about each business, including whether or not it ultimately became successful. With my knowledge of machine learning and neural networks, I decide to use the features in the provided dataset to create a binary classifier model that will predict whether an applicant will become a successful business.

To predict whether Alphabet Soup funding applicants will be successful, I will create a binary classification model using a deep neural network.

## Prepare the Data for Use on a Neural Network Model 

1. Read the `applicants_data.csv` file into a Pandas DataFrame. Review the DataFrame, looking for categorical variables that will need to be encoded, as well as columns that could eventually define my features and target variables.   

2. Drop the “EIN” (Employer Identification Number) and “NAME” columns from the DataFrame by using `.drop(colunms=['EIN', 'NAME'])`, because they are not relevant to the binary classification model.
 
3. Encode the dataset’s categorical variables using `OneHotEncoder(sparse=False)`, and then place the encoded variables into a new DataFrame.

4. Add the original DataFrame’s numerical variables to the DataFrame containing the encoded variables usiing   `concat()`.

5. Using the preprocessed data, create the features (`X`) and target (`y`) datasets. The target dataset should be defined by the preprocessed DataFrame column “IS_SUCCESSFUL”. The remaining columns should define the features dataset. 

6. Split the features and target sets into training and testing datasets.

7. Use scikit-learn's `StandardScaler` to scale the features data.

## Compile and Evaluate a Binary Classification Model Using a Neural Network

1. Create a deep neural network by assigning the number of input features (116), the number of output layers (1), and the number of hidden nodes for first hidden layer (58), number of hidden nodes for second hidden layeer (29), on each layer using Tensorflow’s Keras.

2. Compile and fit the model using the `binary_crossentropy` loss function, the `adam` optimizer, and the `accuracy` evaluation metric.

3. Evaluate the model using the test data to determine the model’s loss and accuracy.

4. Save and export your model to an HDF5 file, and name the file `AlphabetSoup.h5`. 

## Optimize the Neural Network Model

*Define two new deep neural network models to improve on my first model's predictive accuracy.*

**Alternative Model 1**
1. Number of input features (116), number of output neurons layer (1), added three hidden nodes for three hidden layers, activation = `linear`, and fit the model using `50` epochs.

**Alternative Model 2**
1. Number of input features (116), number of output neurons layer (1), added two hidden nodes for two hidden layers, activation = `softmax`, compile the model with loss = `categorical_crossentropy`, and fit the model using `20` epochs.


## Results

* Original Model

Original Model Results
268/268 - 1s - loss: 0.5531 - accuracy: 0.7306 - 591ms/epoch - 2ms/step
Loss: 0.5530970692634583, Accuracy: 0.7306122183799744

* Alternative Model 1

Alternative Model 1 Results
268/268 - 1s - loss: 0.5626 - accuracy: 0.7297 - 590ms/epoch - 2ms/step
Loss: 0.5626217722892761, Accuracy: 0.72967928647995

* Alternative Model 2

Alternative Model 2 Results
268/268 - 1s - loss: 0.0000e+00 - accuracy: 0.5292 - 608ms/epoch - 2ms/step
Loss: 0.0, Accuracy: 0.5292128324508667

## Summary

The result shows that comparing to the original model, adding one more layer and hidden node does not help with the loss and accuracy. However, change the activation and compile loss to `categorical_crossentropy` does make the loss decrease to 0. 

## Contributor
Feier Ou
ffeierou1003@gmail.com