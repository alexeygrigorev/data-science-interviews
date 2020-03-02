# Theoretical interview questions

The list of questions is based on this post: https://hackernoon.com/160-data-science-interview-questions-415s3y2a

Legend: 👶 easy ‍⭐️ medium 🚀 expert

## Supervised machine learning

> What is supervised machine learning? 👶

A case when we have both features (the matrix X) and the labels (the vector y) 

## Linear regression

> What is regression? Which models can you use to solve a regression problem? 👶

Regression is a part of supervised ML. Regression models predict a real number

> What is linear regression? When do we use it? 👶

Answer here

> What’s the normal distribution? Why do we care about it? 👶

Answer here

> How do we check if a variable follows the normal distribution? ‍⭐️

Answer here

> What if we want to build a model for predicting prices? Are prices distributed normally? Do we need to do any pre-processing for prices? ‍⭐️

Answer here

> What are the methods for solving linear regression do you know? ‍⭐️

Answer here

> What is gradient descent? How does it work? ‍⭐️

Answer here

> What is the normal equation? ‍⭐️

Answer here

> What is SGD  —  stochastic gradient descent? What’s the difference with the usual gradient descent? ‍⭐️

Answer here

> Which metrics for evaluating regression models do you know? 👶

Answer here

> What are MSE and RMSE? 👶

Answer here


## Validation

> What is overfitting? 👶

Answer here

> How to validate your models? 👶

Answer here

> Why do we need to split our data into three parts: train, validation, and test? 👶

Answer here

> Can you explain how cross-validation works? 👶

Answer here

> What is K-fold cross-validation? 👶

Answer here

> How do we choose K in K-fold cross-validation? What’s your favorite K? 👶

Answer here


## Classification

> What is classification? Which models would you use to solve a classification problem? 👶

Answer here

> What is logistic regression? When do we need to use it? 👶

Answer here

> Is logistic regression a linear model? Why? 👶

Answer here

> What is sigmoid? What does it do? 👶

Answer here

> How do we evaluate classification models? 👶

Answer here

> What is accuracy? 👶

Accuracy is a metric for evaluating classification models. It is calculated by dividing the number of correct predictions by the number of total predictions.

> Is accuracy always a good metric? 👶

Answer here

> What is the confusion table? What are the cells in this table? 👶

Answer here

> What is precision, recall, and F1-score? 👶

Answer here

> Precision-recall trade-off ‍⭐️

Answer here

> What is the ROC curve? When to use it? ‍⭐️

Answer here

> What is AUC (AU ROC)? When to use it? ‍⭐️

Answer here

> How to interpret the AU ROC score? ‍⭐️

Answer here

> What is the PR (precision-recall) curve? ‍⭐️

Answer here

> What is the area under the PR curve? Is it a useful metric? ‍⭐️I

Answer here

> In which cases AU PR is better than AU ROC? ‍⭐️

Answer here

> What do we do with categorical variables? ‍⭐️

Answer here

> Why do we need one-hot encoding? ‍⭐️

Answer here


## Regularization

> What happens to our linear regression model if we have three columns in our data: x, y, z  —  and z is a sum of x and y? ‍⭐️

Answer here

> What happens to our linear regression model if the column z in the data is a sum of columns x and y and some random noise? ‍⭐️

Answer here

> What is regularization? Why do we need it? 👶

Answer here

> Which regularization techniques do you know? ‍⭐️

Answer here

> What kind of regularization techniques are applicable to linear models? ‍⭐️

Answer here

> How does L2 regularization look like in a linear model? ‍⭐️

Answer here

> How do we select the right regularization parameters? 👶

Answer here

> What’s the effect of L2 regularization on the weights of a linear model? ‍⭐️

Answer here

> How L1 regularization looks like in a linear model? ‍⭐️

Answer here

> What’s the difference between L2 and L1 regularization? ‍⭐️

Answer here

> Can we have both L1 and L2 regularization components in a linear model? ‍⭐️

Answer here

> What’s the interpretation of the bias term in linear models? ‍⭐️

Answer here

> How do we interpret weights in linear models? ‍⭐️

Answer here

> If a weight for one variable is higher than for another  —  can we say that this variable is more important? ‍⭐️

Answer here

> When do we need to perform feature normalization for linear models? When it’s okay not to do it? ‍⭐️

Answer here


## Feature selection

> What is feature selection? Why do we need it? 👶

Answer here

> Is feature selection important for linear models? ‍⭐️

Answer here

> Which feature selection techniques do you know? ‍⭐️

Answer here

> Can we use L1 regularization for feature selection? ‍⭐️

Answer here

> Can we use L2 regularization for feature selection? ‍⭐️

Answer here


## Decision trees

> What are the decision trees? 👶

Answer here

> How do we train decision trees? ‍⭐️

Answer here

> What are the main parameters of the decision tree model? 👶

Answer here

> How do we handle categorical variables in decision trees? ‍⭐️

Answer here

> What are the benefits of a single decision tree compared to more complex models? ‍⭐️

Answer here

> How can we know which features are more important for the decision tree model? ‍⭐️

Answer here


## Random forest

> What is random forest? 👶

Answer here

> Why do we need randomization in random forest? ‍⭐️

Answer here

> What are the main parameters of the random forest model? ‍⭐️

Answer here

> How do we select the depth of the trees in random forest? ‍⭐️

Answer here

> How do we know how many trees we need in random forest? ‍⭐️

Answer here

> Is it easy to parallelize training of a random forest model? How can we do it? ‍⭐️

Answer here

> What are the potential problems with many large trees? ‍⭐️

Answer here

> What if instead of finding the best split, we randomly select a few splits and just select the best from them. Will it work? 🚀

Answer here

> What happens when we have correlated features in our data? ‍⭐️

Answer here


## Gradient boosting

> What is gradient boosting trees? ‍⭐️

Answer here

> What’s the difference between random forest and gradient boosting? ‍⭐️

Answer here

> Is it possible to parallelize training of a gradient boosting model? How to do it? ‍⭐️

Answer here

> Feature importance in gradient boosting trees  —  what are possible options? ‍⭐️

Answer here

> Are there any differences between continuous and discrete variables when it comes to feature importance of gradient boosting models? 🚀

Answer here

> What are the main parameters in the gradient boosting model? ‍⭐️

Answer here

> How do you approach tuning parameters in XGBoost or LightGBM? 🚀

Answer here

> How do you select the number of trees in the gradient boosting model? ‍⭐️

Answer here



## Parameter tuning

> Which parameter tuning strategies (in general) do you know? ‍⭐️

Answer here

> What’s the difference between grid search parameter tuning strategy and random search? When to use one or another? ‍⭐️

Answer here


## Neural networks

> What kind of problems neural nets can solve? 👶

Answer here

> How does a usual fully-connected feed-forward neural network work? ‍⭐️

Answer here

> Why do we need activation functions? 👶

Answer here

> What are the problems with sigmoid as an activation function? ‍⭐️

Answer here

> What is ReLU? How is it better than sigmoid or tanh? ‍⭐️

Answer here

> How we can initialize the weights of a neural network? ‍⭐️

Answer here

> What if we set all the weights of a neural network to 0? ‍⭐️

Answer here

> What regularization techniques for neural nets do you know? ‍⭐️

Answer here

> What is dropout? Why is it useful? How does it work? ‍⭐️

Answer here


## Optimization in neural networks

> What is backpropagation? How does it work? Why do we need it? ‍⭐️

Answer here

> Which optimization techniques for training neural nets do you know? ‍⭐️

Answer here

> How do we use SGD (stochastic gradient descent) for training a neural net? ‍⭐️

Answer here

> What’s the learning rate? 👶

Answer here

> What happens when the learning rate is too large? Too small? 👶

Answer here

> How to set the learning rate? ‍⭐️

Answer here

> What is Adam? What’s the main difference between Adam and SGD? ‍⭐️

Answer here

> When would you use Adam and when SGD? ‍⭐️

Answer here

> Do we want to have a constant learning rate or we better change it throughout training? ‍⭐️

Answer here

> How do we decide when to stop training a neural net? 👶

Answer here

> What is model checkpointing? ‍⭐️

Answer here

> Can you tell us how you approach the model training process? ‍⭐️

Answer here


## Neural networks for computer vision

> How we can use neural nets for computer vision? ‍⭐️

Answer here

> What’s a convolutional layer? ‍⭐️

Answer here

> Why do we actually need convolutions? Can’t we use fully-connected layers for that? ‍⭐️

Answer here

> What’s pooling in CNN? Why do we need it? ‍⭐️

Answer here

> How does max pooling work? Are there other pooling techniques? ‍⭐️

Answer here

> Are CNNs resistant to rotations? What happens to the predictions of a CNN if an image is rotated? 🚀

Answer here

> What are augmentations? Why do we need them? 👶What kind of augmentations do you know? 👶How to choose which augmentations to use? ‍⭐️

Answer here

> What kind of CNN architectures for classification do you know? 🚀

Answer here

> What is transfer learning? How does it work? ‍⭐️

Answer here

> What is object detection? Do you know any architectures for that? 🚀

Answer here

> What is object segmentation? Do you know any architectures for that? 🚀

Answer here


## Text classification

> How can we use machine learning for text classification? ‍⭐️

Answer here

> What is bag of words? How we can use it for text classification? ‍⭐️

Answer here

> What are the advantages and disadvantages of bag of words? ‍⭐️

Answer here

> What are N-grams? How can we use them? ‍⭐️

Answer here

> How large should be N for our bag of words when using N-grams? ‍⭐️

Answer here

> What is TF-IDF? How is it useful for text classification? ‍⭐️

Answer here

> Which model would you use for text classification with bag of words features? ‍⭐️

Answer here

> Would you prefer gradient boosting trees model or logistic regression when doing text classification with bag of words? ‍⭐️

Answer here

> What are word embeddings? Why are they useful? Do you know Word2Vec? ‍⭐️

Answer here

> Do you know any other ways to get word embeddings? 🚀

Answer here

> If you have a sentence with multiple words, you may need to combine multiple word embeddings into one. How would you do it? ‍⭐️

Answer here

> Would you prefer gradient boosting trees model or logistic regression when doing text classification with embeddings? ‍⭐️

Answer here

> How can you use neural nets for text classification? 🚀

Answer here

> How can we use CNN for text classification? 🚀

Answer here


## Clustering

> What is unsupervised learning? 👶

Answer here

> What is clustering? When do we need it? 👶

Answer here

> Do you know how K-means works? ‍⭐️

Answer here

> How to select K for K-means? ‍⭐️

Answer here

> What are the other clustering algorithms do you know? ‍⭐️

Answer here

> Do you know how DBScan works? ‍⭐️

Answer here

> When would you choose K-means and when DBScan? ‍⭐️

Answer here


## Dimensionality reduction
> What is the curse of dimensionality? Why do we care about it? ‍⭐️

Answer here

> Do you know any dimensionality reduction techniques? ‍⭐️

Answer here

> What’s singular value decomposition? How is it typically used for machine learning? ‍⭐️

Answer here


## Ranking and search

> What is the ranking problem? Which models can you use to solve them? ‍⭐️

Answer here

> What are good unsupervised baselines for text information retrieval? ‍⭐️

Answer here

> How would you evaluate your ranking algorithms? Which offline metrics would you use? ‍⭐️

Answer here

> What is precision and recall at k? ‍⭐️

Answer here

> What is mean average precision at k? ‍⭐️

Answer here

> How can we use machine learning for search? ‍⭐️

Answer here

> How can we get training data for our ranking algorithms? ‍⭐️

Answer here

> Can we formulate the search problem as a classification problem? How? ‍⭐️

Answer here

> How can we use clicks data as the training data for ranking algorithms? 🚀

Answer here

> Do you know how to use gradient boosting trees for ranking? 🚀

Answer here

> How do you do an online evaluation of a new ranking algorithm? ‍⭐️

Answer here


## Recommender systems

> What is a recommender system? 👶

Answer here

> What are good baselines when building a recommender system? ‍⭐️

Answer here

> What is collaborative filtering? ‍⭐️

Answer here

> How we can incorporate implicit feedback (clicks, etc) into our recommender systems? ‍⭐️

Answer here

> What is the cold start problem? ‍⭐️

Answer here

> Possible approaches to solving the cold start problem? ‍⭐️🚀

Answer here


## Time series

> What is a time series? 👶

Answer here

> How is time series different from the usual regression problem? 👶

Answer here

> Which models do you know for solving time series problems? ‍⭐️

Answer here

> If there’s a trend in our series, how we can remove it? And why would we want to do it? ‍⭐️

Answer here

> You have a series with only one variable “y” measured at time t. How do predict “y” at time t+1? Which approaches would you use? ‍⭐️

Answer here

> You have a series with a variable “y” and a set of features. How do you predict “y” at t+1? Which approaches would you use? ‍⭐️

Answer here

> What are the problems with using trees for solving time series problems? ‍⭐️

Answer here


