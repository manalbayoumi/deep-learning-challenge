# Module 21 deep-learning-challenge Analysis Report
## Overview 
1. **The purpose of the analysis**
	The analysis is for the nonprofit foundation Alphabet Soup that wants to use a tool that can help it select the applicants for funding with the best chance of success in their ventures. With our knowledge of machine learning and neural networks, we will use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.

2. **About the Data**
	The data is a large one, it has about twelve columns and about 34,299 rows. From Alphabet Soup’s business team, you have received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. 
	Within this dataset are a number of columns that capture metadata about each organization, such as:
	* EIN and NAME—Identification columns
	* APPLICATION_TYPE—Alphabet Soup application type
	* AFFILIATION—Affiliated sector of industry
	* CLASSIFICATION—Government organization classification
	* USE_CASE—Use case for funding
	* ORGANIZATION—Organization type
	* STATUS—Active status
	* INCOME_AMT—Income classification
	* SPECIAL_CONSIDERATIONS—Special considerations for application
	* ASK_AMT—Funding amount requested
	* IS_SUCCESSFUL—Was the money used effectively

3. **Deep Learing Process**
	The deep Learning Process has 3 steps. Using your knowledge of Pandas and scklearn, you’ll need to preprocess the dataset. This step prepares you for Step 2, where you'll compile, train, and evaluate the neural network model.
	Start by uploading the starter file to Google Colab, then using the information we provided in the Challenge files, follow the instructions to complete the preprocessing steps.
	* Step 1: Preprocess the Data
	1. Read in the charity_data.csv to a Pandas DataFrame, and be sure to identify the following in your dataset:
	2. What variable(s) are the target(s) for your model?
	3. What variable(s) are the feature(s) for your model? Drop the EIN and NAME columns.
	4. Determine the number of unique values for each column.
	5. For columns that have more than 10 unique values, determine the number of data points for each unique value.
	6. Use the number of data points for each unique value to pick a cutoff point to bin "rare" categorical variables together in a new value, Other, and then check if the binning was successful.
	7. Use pd.get_dummies() to encode categorical variables.
	8. Split the preprocessed data into a features array, X, and a target array, y. Use these arrays and the train_test_split function to split the data into training and testing datasets.
	9. Scale the training and testing features datasets by creating a StandardScaler instance, fitting it to the training data, then using the transform function.

	* Step 2: Compile, Train, and Evaluate the Model
	Using your knowledge of TensorFlow, you’ll design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup-funded organization will be successful based on the features in the dataset. You’ll need to think about how many inputs there are before determining the number of neurons and layers in your model. Once you’ve completed that step, you’ll compile, train, and evaluate your binary classification model to calculate the model’s loss and accuracy.
	1. Continue using the file in Google Colab in which you performed the preprocessing steps from Step 1.
	2. Create a neural network model by assigning the number of input features and nodes for each layer using TensorFlow and Keras.
	3. Create the first hidden layer and choose an appropriate activation function.
	4. If necessary, add a second hidden layer with an appropriate activation function.
	5. Create an output layer with an appropriate activation function.
	6. Check the structure of the model.
	7. Compile and train the model.
	8. Create a callback that saves the model's weights every five epochs.
	9. Evaluate the model using the test data to determine the loss and accuracy.
	10. Save and export your results to an HDF5 file. Name the file AlphabetSoupCharity.h5.
	
	* Step 3: Optimize the Model
	Using your knowledge of TensorFlow, optimize your model to achieve a target predictive accuracy higher than 75%. Use any or all of the following methods to optimize your model:
	Adjust the input data to ensure that no variables or outliers are causing confusion in the model, such as: 
	1. Dropping more or fewer columns.
	2. Creating more bins for rare occurrences in columns.
	3. Increasing or decreasing the number of values for each bin.
	4. Add more neurons to a hidden layer.
	5. Add more hidden layers.
	6. Use different activation functions for the hidden layers.
	7. Add or reduce the number of epochs to the training regimen.

	* Step 4: Write a Report on the Neural Network Model
	For this part of the assignment, you’ll write a report on the performance of the deep learning model you created for Alphabet Soup.
	1. Overview of the analysis: The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. Using machine learning and neural networks, we use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.

	2. Results: Using bulleted lists and images to support your answers, address the following questions:
	* Data Preprocessing
	* What variable(s) are the target(s) for your model?
	Our target is the "y" which is the "IS_SUCCESSFUL" column

	* What variable(s) are the features for your model?
	Our features is the "X" which is everything but the "IS_SUCCESSFUL" column as: "APPLICATION_TYPE" , "AFFILIATION", "CLASSIFICATION", "USE_CASE", "ORGANIZATION", "STATUS", "INCOME_AMT", "SPECIAL_CONSIDERATIONS", "ASK_AMT"

	* What variable(s) should be removed from the input data because they are neither targets nor features?
	"EIN & "NAME" were removed from variables, since they are neither targets nor features.

	* Compiling, Training, and Evaluating the Model
	* How many neurons, layers, and activation functions did you select for your neural network model, and why?
 	I have 4 layers, 3 input layers, 2 hidden layers and 1 output layers. I have 43+80+30+1=154 neurons. I have 2 activitation functions. I am using the relu function because X is non linear relationship. For the output, I am using the Sigmoid because the result is two value classifications. 

	* Were you able to achieve the target model performance? The best I got an accuracy of 0.7402.

	* What steps did you take in your attempts to increase model performance? There are four ways to increase the performance:
	1. Increase layer of neural network and check if performance is getting better.
	2. Adjusting the neural unit number. Using too few neural numbers may lead to underfitting. On other hand, using too many neural numbers may result in overfitting.
	3. You can change the activitation function, for hidden layer use `relu` and for classification we can use `Sigmoid`, for multiple classification use `softmax`, for regression use `linear`
	4. Increase the number of epochs.


	3. Summary: Summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and then explain your recommendation.
	 For this model, the loss is 56% and accuracy is 72% after 100 epoch run. When the neural network model is running, the loss should drop and approach zero, at the same time the accuracy should rise. So, 72% of accuracy is good enough, but 56% of loss is not good and far away from having a successful model. We have to retrain the model to lower the loss using the techniques listed above. 

		
4. **Machine Learning Packages**     
```python 
# Import our dependencies
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense
``` 
5. Copy Files Into Your Repository
Now that you're finished with your analysis in Google Colab, you need to get your files into your repository for final submission.
	1. Download your Colab notebooks to your computer.
	2. Move them into your Deep Learning Challenge directory in your local repository.
	3. Push the added files to GitHub.

## Deep Learning Results     
![model.PNG](./Alphabet%20Soup/model.PNG)    

![report.PNG](./Alphabet%20Soup/report.PNG)    

![Epoch.PNG](./Alphabet%20Soup/Epoch.PNG)

## Conclusion
For this model, the loss is 56% and accuracy is 72% after 100 epoch run. When the neural network model is running, the loss should drop and approach zero, at the same time the accuracy should rise. So, 72% of accuracy is good enough, but 56% of loss is not good and far away from having a successful model. We have to retrain the model to lower the loss using the techniques listed above. 