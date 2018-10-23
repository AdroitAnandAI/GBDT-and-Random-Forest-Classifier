# Gradient Boosted Decision Tree &amp; Random Forest Classifier on high dimensional data (Part I) #

## Amazon Fine Food Review Dataset ##

Data Source: https://www.kaggle.com/snap/amazon-fine-food-reviews

The Amazon Fine Food Reviews dataset consists of reviews of fine foods from Amazon. <br/>
Number of reviews                   : 568,454  <br/>
Number of users                     : 256,059  <br/>
Number of products                  : 74,258  <br/>
Timespan: Oct 1999                  : Oct 2012  <br/>
Number of Attributes/Columns in data: 10 <br/>

### Attribute Information ###
1. Id <br/>
2. ProductId - unique identifier for the product <br/>
3. UserId - unqiue identifier for the user <br/>
4. ProfileName <br/>
5. HelpfulnessNumerator - number of users who found the review helpful <br/>
6. HelpfulnessDenominator - number of users who indicated whether they found the review helpful or not <br/>
7. Score - rating between 1 and 5 <br/>
8. Time - timestamp for the review <br/>
9. Summary - brief summary of the review <br/>
10. Text - text of the review <br/>

## Objective ##

The code below would **clean the review text from html tags and punctuations and write it as a new column in the database and write it to disk**. This is further taken up in Part 2 to find accuracy of 10-fold cross validation KNN on vectorized input data, for each of the 4 featurizations, namely BoW, tf-IDF, W2V, tf-IDF weighted W2V.

## Significant Points ##

1. **Duplication of reviews** are found with same userid and timestamp (Cleaned).
2. Found discrepancy issues with HelpfulnessDenominator (Cleaned).
3. final.sqlite db is to be **used for further processing** such as Text to Vector operations.
4. The preprocessing step is one time effort but the training & visualization steps require multiple runs. Hence, it is prudent to make reprocessing step independant, to avoid multiple runs.

# Gradient Boosted Decision Tree &amp; Random Forest Classifier on high dimensional data (Part II) #

## Data Source ##

The preprocessing step has produced final.sqlite file after doing the data preparation & cleaning. The review text is now devoid of punctuations, HTML markups and stop words.

## Objective ##

To find number of trees of Random Forest Classifier using GridSearchCV or iterated cross validation on standardized feature vectors obtained from BoW, tf-idf, W2V and tf-idf weighted W2V featurizations.

Use Gradient Boosted Decision Trees alongsize RF for comparison. GridSearchCV is to be done in order to tune 3 hyperparameters: # of trees, depth of the tree as well as learning rate.

Find Precision, Recall, F1 Score, Confusion Matrix, Accuracy of the optimal model obstained with GridSearch or Cross Validation, on vectorized input data, for BoW, tf-idf, W2V and tf-idf weighted W2V featurizations. TPR, TNR, FPR and FNR is calculated for all.

## At a glance ##

Tail end data is taken after sorting the data, to conserve the timing info & time Series based cross validation is done, as it is time series data. The optimal number of trees for Random Forest Classifier is found using GridSearchCV (wrote code for cross validation also), by searching for # of trees between 9 - 135, with step size 9.

GBDT is also done along with RF. The optimal parameters for GBDT is found using GridSearchCV. Parameter tuning of 3 hyperparameters is done: # of trees between 20-81 (with step size 10), max_depth between 5-16 (with step size 2) & learning_rate between 0.05 - 0.2 (with step size 0.05).

The Precision, Recall, F1 Score, Confusion Matrix, Accuracy metrics are found out for all 4 featurizations.

## Custom Defined Functions ##

4 user defined functions are written to

a) Random Forest Hyperparameter Tuning

b) Gradient Boosted Decision Tree (GBDT) Hyperparameter Tuning

c) Compute Performance Metrics for RF & GBDT

d) Generate Word Cloud based on Feature Importance

## BoW ##

BoW will result in a sparse matrix with huge number of features as it creates a feature for each unique word in the review.

For Binary BoW feature representation, CountVectorizer is declared as float, as the values can take non-integer values on further processing.

![1](https://github.com/AdroitAnandAI/GBDT-and-Random-Forest-Classifier/blob/master/images/1.PNG)

![2](https://github.com/AdroitAnandAI/GBDT-and-Random-Forest-Classifier/blob/master/images/2.PNG)

<p align="center">
    <img src="https://github.com/AdroitAnandAI/GBDT-and-Random-Forest-Classifier/blob/master/images/3.PNG">
</p>

<p align="center">
    <img src="https://github.com/AdroitAnandAI/GBDT-and-Random-Forest-Classifier/blob/master/images/4.PNG">
</p>

![5](https://github.com/AdroitAnandAI/GBDT-and-Random-Forest-Classifier/blob/master/images/5.PNG)

![6](https://github.com/AdroitAnandAI/GBDT-and-Random-Forest-Classifier/blob/master/images/6.PNG)

<p align="center">
    <img src="https://github.com/AdroitAnandAI/GBDT-and-Random-Forest-Classifier/blob/master/images/7.PNG">
</p>

<p align="center">
    <img src="https://github.com/AdroitAnandAI/GBDT-and-Random-Forest-Classifier/blob/master/images/8.PNG">
</p>

## tf-IDF ##

Sparse matrix generated from tf-IDF is fed in to GridSearch GBDT Cross Validator & RF Cross Validator to find the optimal depth value. Performance metrics of optimal GBDT with tf-idf featurization is found.

![9](https://github.com/AdroitAnandAI/GBDT-and-Random-Forest-Classifier/blob/master/images/9.PNG)

![10](https://github.com/AdroitAnandAI/GBDT-and-Random-Forest-Classifier/blob/master/images/10.PNG)

<p align="center">
    <img src="https://github.com/AdroitAnandAI/GBDT-and-Random-Forest-Classifier/blob/master/images/11.PNG">
</p>

<p align="center">
    <img src="https://github.com/AdroitAnandAI/GBDT-and-Random-Forest-Classifier/blob/master/images/12.PNG">
</p>

![13](https://github.com/AdroitAnandAI/GBDT-and-Random-Forest-Classifier/blob/master/images/13.PNG)

![14](https://github.com/AdroitAnandAI/GBDT-and-Random-Forest-Classifier/blob/master/images/14.PNG)

<p align="center">
    <img src="https://github.com/AdroitAnandAI/GBDT-and-Random-Forest-Classifier/blob/master/images/15.PNG">
</p>

<p align="center">
    <img src="https://github.com/AdroitAnandAI/GBDT-and-Random-Forest-Classifier/blob/master/images/16.PNG">
</p>

## Word2Vec ##

**Dense matrix generated from Word2Vec** is fed in to GridSearch GBDT Cross Validator & RF Cross Validator to find the optimal depth value. Performance metrics of GBDT and RF with W2V featurization is found.

![17](https://github.com/AdroitAnandAI/GBDT-and-Random-Forest-Classifier/blob/master/images/17.PNG)

![18](https://github.com/AdroitAnandAI/GBDT-and-Random-Forest-Classifier/blob/master/images/18.PNG)

![18-2](https://github.com/AdroitAnandAI/GBDT-and-Random-Forest-Classifier/blob/master/images/18-2.PNG)

![18-3](https://github.com/AdroitAnandAI/GBDT-and-Random-Forest-Classifier/blob/master/images/18-3.PNG)

<p align="center">
    <img src="https://github.com/AdroitAnandAI/GBDT-and-Random-Forest-Classifier/blob/master/images/19.PNG">
</p>

<p align="center">
    <img src="https://github.com/AdroitAnandAI/GBDT-and-Random-Forest-Classifier/blob/master/images/20.PNG">
</p>

## TF-ID Weighted W2V ##

![21](https://github.com/AdroitAnandAI/GBDT-and-Random-Forest-Classifier/blob/master/images/21.PNG)

![22](https://github.com/AdroitAnandAI/GBDT-and-Random-Forest-Classifier/blob/master/images/22.PNG)

![22-2](https://github.com/AdroitAnandAI/GBDT-and-Random-Forest-Classifier/blob/master/images/22-2.PNG)

![22-3](https://github.com/AdroitAnandAI/GBDT-and-Random-Forest-Classifier/blob/master/images/22-3.PNG)

<p align="center">
    <img src="https://github.com/AdroitAnandAI/GBDT-and-Random-Forest-Classifier/blob/master/images/23.PNG">
</p>

<p align="center">
    <img src="https://github.com/AdroitAnandAI/GBDT-and-Random-Forest-Classifier/blob/master/images/24.PNG">
</p>

## Summary Statistics ##

<p align="center">
    <img src="https://github.com/AdroitAnandAI/GBDT-and-Random-Forest-Classifier/blob/master/images/25.PNG">
</p>

## Observations ##

1. The **best model** based on test metrics is found to be **GBDT on tf-idf**. The F1 Score is **94.12**, while **98% of positive points and around 50% of negative points** are detected correctly.

2. The **classification accuracy of GBDT and RF is found to be less than linear models** like logistic regression. This is possibly because the **separation at high dimensional space using hyperplanes is easier** than doing a decision tree based approach.

3. **GBDT consistently performs better than RF for Amazon review classiciation problem.

4. The **wordcloud figure of GBDT is much clearer than RF**. RF wordcloud is too cluttered and hence less suitable.

5. **Words such as “dissapoint”, “horrible” etc have high feature importance**, as evident from the wordcloud. While these words are important **to classify negative reviews, words like “love”, “good”, “best” etc. also have high feature importance, as they intuitively denote positive reviews**.
