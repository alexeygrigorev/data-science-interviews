# Theoretical interview questions

* The list of questions is based on this post: https://hackernoon.com/160-data-science-interview-questions-415s3y2a
* Legend: 👶 easy ‍⭐️ medium 🚀 expert
* Do you know how to answer questions without answers? Please create a PR
* See an error? Please create a PR with fix

## Supervised machine learning

**What is supervised machine learning? 👶**

A case when we have both features (the matrix X) and the labels (the vector y) 

<br/>

## Linear regression

**What is regression? Which models can you use to solve a regression problem? 👶**

Regression is a part of supervised ML. Regression models predict a real number

<br/>

**What is linear regression? When do we use it? 👶**

Linear regression is a model that assumes a linear relationship between the input variables (X) and the single output variable (y). 

With a simple equation:

```
y = B0 + B1*x1 + ... + Bn * xN
```

B is regression coefficients, x values are the independent (explanatory) variables  and y is dependent variable. 

The case of one explanatory variable is called simple linear regression. For more than one explanatory variable, the process is called multiple linear regression.

Simple linear regression:

```
y = B0 + B1*x1
```

Multiple linear regression:

```
y = B0 + B1*x1 + ... + Bn * xN
```

<br/>

**What’s the normal distribution? Why do we care about it? 👶**

Answer here

<br/>

**How do we check if a variable follows the normal distribution? ‍⭐️**

Answer here

<br/>

**What if we want to build a model for predicting prices? Are prices distributed normally? Do we need to do any pre-processing for prices? ‍⭐️**

Answer here

<br/>

**What are the methods for solving linear regression do you know? ‍⭐️**

Answer here

<br/>

**What is gradient descent? How does it work? ‍⭐️**

Answer here

<br/>

**What is the normal equation? ‍⭐️**

Normal equations are equations obtained by setting equal to zero the partial derivatives of the sum of squared errors (least squares); normal equations allow one to estimate the parameters of a multiple linear regression.

<br/>

**What is SGD  —  stochastic gradient descent? What’s the difference with the usual gradient descent? ‍⭐️**

Answer here

<br/>

**Which metrics for evaluating regression models do you know? 👶**

Answer here

<br/>

**What are MSE and RMSE? 👶**

Answer here

<br/>


## Validation

**What is overfitting? 👶**

When your model perform very well on your training set but can't generalize the test set, because it adjusted a lot to the training set.

<br/>

**How to validate your models? 👶**

Answer here

<br/>

**Why do we need to split our data into three parts: train, validation, and test? 👶**

The training set is used to fit the model, i.e. to train the model with the data. The validation set is then used to provide an unbiased evaluation of a model while fine-tuning hyperparameters. This improves the generalization of the model. Finally, a test data set which the model has never "seen" before should be used for the final evaluation of the model. This allows for an unbiased evaluation of the model. The evaluation should never be performed on the same data that is used for training. Otherwise the model performance would not be representative.

<br/>

**Can you explain how cross-validation works? 👶**

Cross-validation is the process to separate your total training set into two subsets: training and validation set, and evaluate your model to choose the hyperparameters. But you do this process iteratively, selecting differents training and validation set, in order to reduce the bias that you would have by selecting only one validation set.

<br/>

**What is K-fold cross-validation? 👶**

Answer here

<br/>

**How do we choose K in K-fold cross-validation? What’s your favorite K? 👶**

Answer here

<br/>


## Classification

**What is classification? Which models would you use to solve a classification problem? 👶**

Answer here

<br/>

**What is logistic regression? When do we need to use it? 👶**

Answer here

<br/>

**Is logistic regression a linear model? Why? 👶**

Answer here

<br/>

**What is sigmoid? What does it do? 👶**

Answer here

<br/>

**How do we evaluate classification models? 👶**

Answer here

<br/>

**What is accuracy? 👶**

Accuracy is a metric for evaluating classification models. It is calculated by dividing the number of correct predictions by the number of total predictions.

<br/>

**Is accuracy always a good metric? 👶**

Accuracy is not a good performance metric when there is imbalance in the dataset. For example, in binary classification with 95% of A class and 5% of B class, prediction accuracy can be 95%. In case of imbalance dataset, we need to choose Precision, recall, or F1 Score depending on the problem we are trying to solve. 

<br/>

**What is the confusion table? What are the cells in this table? 👶**

Confusion table (or confusion matrix) shows how many True positives (TP), True Negative (TN), False Positive (FP) and False Negative (FN) model has made. 

||                |     Actual   |        Actual |
|:---:|   :---:        |     :---:    |:---:          |
||                | Positive (1) | Negative (0)  |
|Predicted|   Positive (1) | TP           | FP            |
|Predicted|   Negative (0) | FN           | TN            |

* True Positives (TP): When the actual class of the observation is 1 (True) and the prediction is 1 (True)
* True Negative (TN): When the actual class of the observation is 0 (False) and the prediction is 0 (False)
* False Positive (FP): When the actual class of the observation is 0 (False) and the prediction is 1 (True)
* False Negative (FN): When the actual class of the observation is 1 (True) and the prediction is 0 (False)

Most of the performance metrics for classification models are based on the values of the confusion matrix. 

<br/>

**What are precision, recall, and F1-score? 👶**

* Precision and recall are classification evaluation metrics:
* P = TP / (TP + FP) and R = TP / (TP + FN).
* Where TP is true positives, FP is false positives and FN is false negatives
* In both cases the score of 1 is the best: we get no false positives or false negatives and only true positives.
* F1 is a combination of both precision and recall in one score:
* F1 = 2 * PR / (P + R). 
* Max F score is 1 and min is 0, with 1 being the best.

<br/>

**Precision-recall trade-off ‍⭐️**

Answer here

<br/>

**What is the ROC curve? When to use it? ‍⭐️**

Answer here

<br/>

**What is AUC (AU ROC)? When to use it? ‍⭐️**

Answer here

<br/>

**How to interpret the AU ROC score? ‍⭐️**

Answer here

<br/>

**What is the PR (precision-recall) curve? ‍⭐️**

Answer here

<br/>

**What is the area under the PR curve? Is it a useful metric? ‍⭐️I**

Answer here

<br/>

**In which cases AU PR is better than AU ROC? ‍⭐️**

Answer here

<br/>

**What do we do with categorical variables? ‍⭐️**

Answer here

<br/>

**Why do we need one-hot encoding? ‍⭐️**

Answer here

<br/>


## Regularization

**What happens to our linear regression model if we have three columns in our data: x, y, z  —  and z is a sum of x and y? ‍⭐️**

Answer here

<br/>

**What happens to our linear regression model if the column z in the data is a sum of columns x and y and some random noise? ‍⭐️**

Answer here

<br/>

**What is regularization? Why do we need it? 👶**

Answer here

<br/>

**Which regularization techniques do you know? ‍⭐️**

Answer here

<br/>

**What kind of regularization techniques are applicable to linear models? ‍⭐️**

Answer here

<br/>

**How does L2 regularization look like in a linear model? ‍⭐️**

Answer here

<br/>

**How do we select the right regularization parameters? 👶**

Answer here

<br/>

**What’s the effect of L2 regularization on the weights of a linear model? ‍⭐️**

Answer here

<br/>

**How L1 regularization looks like in a linear model? ‍⭐️**

Answer here

<br/>

**What’s the difference between L2 and L1 regularization? ‍⭐️**

Answer here

<br/>

**Can we have both L1 and L2 regularization components in a linear model? ‍⭐️**

Answer here

<br/>

**What’s the interpretation of the bias term in linear models? ‍⭐️**

Answer here

<br/>

**How do we interpret weights in linear models? ‍⭐️**

If the variables are normalized, we can interpret weights in linear models like the importance of this variable in the predicted result.

<br/>

**If a weight for one variable is higher than for another  —  can we say that this variable is more important? ‍⭐️**

Answer here

<br/>

**When do we need to perform feature normalization for linear models? When it’s okay not to do it? ‍⭐️**

Answer here

<br/>


## Feature selection

**What is feature selection? Why do we need it? 👶**

Answer here

<br/>

**Is feature selection important for linear models? ‍⭐️**

Answer here

<br/>

**Which feature selection techniques do you know? ‍⭐️**

Answer here

<br/>

**Can we use L1 regularization for feature selection? ‍⭐️**

Answer here

<br/>

**Can we use L2 regularization for feature selection? ‍⭐️**

Answer here

<br/>


## Decision trees

**What are the decision trees? 👶**

Answer here

<br/>

**How do we train decision trees? ‍⭐️**

Answer here

<br/>

**What are the main parameters of the decision tree model? 👶**

Answer here

<br/>

**How do we handle categorical variables in decision trees? ‍⭐️**

Answer here

<br/>

**What are the benefits of a single decision tree compared to more complex models? ‍⭐️**

Answer here

<br/>

**How can we know which features are more important for the decision tree model? ‍⭐️**

Answer here

<br/>


## Random forest

**What is random forest? 👶**

Answer here

<br/>

**Why do we need randomization in random forest? ‍⭐️**

Answer here

<br/>

**What are the main parameters of the random forest model? ‍⭐️**

Answer here

<br/>

**How do we select the depth of the trees in random forest? ‍⭐️**

Answer here

<br/>

**How do we know how many trees we need in random forest? ‍⭐️**

Answer here

<br/>

**Is it easy to parallelize training of a random forest model? How can we do it? ‍⭐️**

Answer here

<br/>

**What are the potential problems with many large trees? ‍⭐️**

Answer here

<br/>

**What if instead of finding the best split, we randomly select a few splits and just select the best from them. Will it work? 🚀**

Answer here

<br/>

**What happens when we have correlated features in our data? ‍⭐️**

Answer here

<br/>


## Gradient boosting

**What is gradient boosting trees? ‍⭐️**

Answer here

<br/>

**What’s the difference between random forest and gradient boosting? ‍⭐️**

Answer here

<br/>

**Is it possible to parallelize training of a gradient boosting model? How to do it? ‍⭐️**

Answer here

<br/>

**Feature importance in gradient boosting trees  —  what are possible options? ‍⭐️**

Answer here

<br/>

**Are there any differences between continuous and discrete variables when it comes to feature importance of gradient boosting models? 🚀**

Answer here

<br/>

**What are the main parameters in the gradient boosting model? ‍⭐️**

Answer here

<br/>

**How do you approach tuning parameters in XGBoost or LightGBM? 🚀**

Answer here

<br/>

**How do you select the number of trees in the gradient boosting model? ‍⭐️**

Answer here

<br/>



## Parameter tuning

**Which parameter tuning strategies (in general) do you know? ‍⭐️**

Answer here

<br/>

**What’s the difference between grid search parameter tuning strategy and random search? When to use one or another? ‍⭐️**

Answer here

<br/>


## Neural networks

**What kind of problems neural nets can solve? 👶**

Answer here

<br/>

**How does a usual fully-connected feed-forward neural network work? ‍⭐️**

Answer here

<br/>

**Why do we need activation functions? 👶**

Answer here

<br/>

**What are the problems with sigmoid as an activation function? ‍⭐️**

Answer here

<br/>

**What is ReLU? How is it better than sigmoid or tanh? ‍⭐️**

Answer here

<br/>

**How we can initialize the weights of a neural network? ‍⭐️**

Answer here

<br/>

**What if we set all the weights of a neural network to 0? ‍⭐️**

Answer here

<br/>

**What regularization techniques for neural nets do you know? ‍⭐️**

Answer here

<br/>

**What is dropout? Why is it useful? How does it work? ‍⭐️**

Answer here

<br/>


## Optimization in neural networks

**What is backpropagation? How does it work? Why do we need it? ‍⭐️**

Answer here

<br/>

**Which optimization techniques for training neural nets do you know? ‍⭐️**

Answer here

<br/>

**How do we use SGD (stochastic gradient descent) for training a neural net? ‍⭐️**

Answer here

<br/>

**What’s the learning rate? 👶**

The learning rate is an important hyperparameter that controls how quickly the model is adapted to the problem during the training. It can be seen as the "step width" during the parameter updates, i.e. how far the weights are moved into the direction of the minimum of our optimization problem.

<br/>

**What happens when the learning rate is too large? Too small? 👶**

A large learning rate can accelerate the training. However, it is possible that we "shoot" too far and miss the minimum of the function that we want to optimize, which will not result in the best solution. On the other hand, training with a small learning rate takes more time but it is possible to find a more precise minimum. The downside can be that the solution is stuck in a local minimum, and the weights won't update even if it is not the best possible global solution.

<br/>

**How to set the learning rate? ‍⭐️**

Answer here

<br/>

**What is Adam? What’s the main difference between Adam and SGD? ‍⭐️**

Answer here

<br/>

**When would you use Adam and when SGD? ‍⭐️**

Answer here

<br/>

**Do we want to have a constant learning rate or we better change it throughout training? ‍⭐️**

Answer here

<br/>

**How do we decide when to stop training a neural net? 👶**

Answer here

<br/>

**What is model checkpointing? ‍⭐️**

Answer here

<br/>

**Can you tell us how you approach the model training process? ‍⭐️**

Answer here

<br/>


## Neural networks for computer vision

**How we can use neural nets for computer vision? ‍⭐️**

Answer here

<br/>

**What’s a convolutional layer? ‍⭐️**

Answer here

<br/>

**Why do we actually need convolutions? Can’t we use fully-connected layers for that? ‍⭐️**

Answer here

<br/>

**What’s pooling in CNN? Why do we need it? ‍⭐️**

Answer here

<br/>

**How does max pooling work? Are there other pooling techniques? ‍⭐️**

Answer here

<br/>

**Are CNNs resistant to rotations? What happens to the predictions of a CNN if an image is rotated? 🚀**

Answer here

<br/>

**What are augmentations? Why do we need them? 👶What kind of augmentations do you know? 👶How to choose which augmentations to use? ‍⭐️**

Answer here

<br/>

**What kind of CNN architectures for classification do you know? 🚀**

Answer here

<br/>

**What is transfer learning? How does it work? ‍⭐️**

Answer here

<br/>

**What is object detection? Do you know any architectures for that? 🚀**

Answer here

<br/>

**What is object segmentation? Do you know any architectures for that? 🚀**

Answer here

<br/>


## Text classification

**How can we use machine learning for text classification? ‍⭐️**

Answer here

<br/>

**What is bag of words? How we can use it for text classification? ‍⭐️**

Answer here

<br/>

**What are the advantages and disadvantages of bag of words? ‍⭐️**

Answer here

<br/>

**What are N-grams? How can we use them? ‍⭐️**

Answer here

<br/>

**How large should be N for our bag of words when using N-grams? ‍⭐️**

Answer here

<br/>

**What is TF-IDF? How is it useful for text classification? ‍⭐️**

Answer here

<br/>

**Which model would you use for text classification with bag of words features? ‍⭐️**

Answer here

<br/>

**Would you prefer gradient boosting trees model or logistic regression when doing text classification with bag of words? ‍⭐️**

Answer here

<br/>

**What are word embeddings? Why are they useful? Do you know Word2Vec? ‍⭐️**

Answer here

<br/>

**Do you know any other ways to get word embeddings? 🚀**

Answer here

<br/>

**If you have a sentence with multiple words, you may need to combine multiple word embeddings into one. How would you do it? ‍⭐️**

Answer here

<br/>

**Would you prefer gradient boosting trees model or logistic regression when doing text classification with embeddings? ‍⭐️**

Answer here

<br/>

**How can you use neural nets for text classification? 🚀**

Answer here

<br/>

**How can we use CNN for text classification? 🚀**

Answer here

<br/>


## Clustering

**What is unsupervised learning? 👶**

Unsupervised learning aims to detect paterns in data where no labels are given.

<br/>

**What is clustering? When do we need it? 👶**

Clustering algorithms group objects such that similar feature points are put into the same groups (clusters) and dissimilar feature points are put into different clusters. 

<br/>

**Do you know how K-means works? ‍⭐️**

1. Partition points into k subsets.
2. Compute the seed points as the new centroids of the clusters of the current partitioning.
3. Assign each point to the cluster with the nearest seed point.
4. Go back to step 2 or stop when the assignment does not change.

<br/>

**How to select K for K-means? ‍⭐️**

* Domain knowledge, i.e. an expert knows the value of k
* Elbow method: compute the clusters for different values of k, for each k, calculate the total within-cluster sum of square, plot the sum according to the number of clusters and use the band as the number of clusters.
* Average silhouette method: compute the clusters for different values of k, for each k, calculate the average silhouette of observations, plot the silhouette according to the number of clusters and select the maximum as the number of clusters.

<br/>

**What are the other clustering algorithms do you know? ‍⭐️**

* k-medoids: Takes the most central point instead of the mean value as the center of the cluster. This makes it more robust to noise.
* Agglomerative Hierarchical Clustering (AHC): hierarchical clusters combining the nearest clusters starting with each point as its own cluster.
* DIvisive ANAlysis Clustering (DIANA): hierarchical clustering starting with one cluster containing all points and splitting the clusters until each point describes its own cluster.
* Density-Based Spatial Clustering of Applications with Noise (DBSCAN): Cluster defined as maximum set of density-connected points.

<br/>

**Do you know how DBScan works? ‍⭐️**

* Two input parameters epsilon (neighborhood radius) and minPts (minimum number of points in an epsilon-neighborhood)
* Cluster defined as maximum set of density-connected points.
* Points p_j and p_i are density-connected w.r.t. epsilon and minPts if there is a point o such that both, i and j are density-reachable from o w.r.t. epsilon and minPts.
* p_j is density-reachable from p_i w.r.t. epsilon, minPts if there is a chain of points p_i -> p_i+1 -> p_i+x = p_j such that p_i+x is directly density-reachable from p_i+x-1.
* p_j is a directly density-reachable point of the neighborhood of p_i if dist(p_i,p_j) <= epsilon.

<br/>

**When would you choose K-means and when DBScan? ‍⭐️**

* DBScan is more robust to noise.
* DBScan is better when the amount of clusters is difficult to guess.
* K-means has a lower complexity, i.e. it will be much faster, especially with a larger amount of points.

<br/>


## Dimensionality reduction
**What is the curse of dimensionality? Why do we care about it? ‍⭐️**

Answer here

<br/>

**Do you know any dimensionality reduction techniques? ‍⭐️**

Answer here

<br/>

**What’s singular value decomposition? How is it typically used for machine learning? ‍⭐️**

Answer here

<br/>


## Ranking and search

**What is the ranking problem? Which models can you use to solve them? ‍⭐️**

Answer here

<br/>

**What are good unsupervised baselines for text information retrieval? ‍⭐️**

Answer here

<br/>

**How would you evaluate your ranking algorithms? Which offline metrics would you use? ‍⭐️**

Answer here

<br/>

**What is precision and recall at k? ‍⭐️**

Answer here

<br/>

**What is mean average precision at k? ‍⭐️**

Answer here

<br/>

**How can we use machine learning for search? ‍⭐️**

Answer here

<br/>

**How can we get training data for our ranking algorithms? ‍⭐️**

Answer here

<br/>

**Can we formulate the search problem as a classification problem? How? ‍⭐️**

Answer here

<br/>

**How can we use clicks data as the training data for ranking algorithms? 🚀**

Answer here

<br/>

**Do you know how to use gradient boosting trees for ranking? 🚀**

Answer here

<br/>

**How do you do an online evaluation of a new ranking algorithm? ‍⭐️**

Answer here

<br/>


## Recommender systems

**What is a recommender system? 👶**

Answer here

<br/>

**What are good baselines when building a recommender system? ‍⭐️**

Answer here

<br/>

**What is collaborative filtering? ‍⭐️**

Answer here

<br/>

**How we can incorporate implicit feedback (clicks, etc) into our recommender systems? ‍⭐️**

Answer here

<br/>

**What is the cold start problem? ‍⭐️**

Answer here

<br/>

**Possible approaches to solving the cold start problem? ‍⭐️🚀**

Answer here

<br/>


## Time series

**What is a time series? 👶**

Answer here

<br/>

**How is time series different from the usual regression problem? 👶**

Answer here

<br/>

**Which models do you know for solving time series problems? ‍⭐️**

Answer here

<br/>

**If there’s a trend in our series, how we can remove it? And why would we want to do it? ‍⭐️**

Answer here

<br/>

**You have a series with only one variable “y” measured at time t. How do predict “y” at time t+1? Which approaches would you use? ‍⭐️**

Answer here

<br/>

**You have a series with a variable “y” and a set of features. How do you predict “y” at t+1? Which approaches would you use? ‍⭐️**

Answer here

<br/>

**What are the problems with using trees for solving time series problems? ‍⭐️**

Answer here

<br/>


