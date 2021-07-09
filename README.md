### ML Test Assignment

#### Kateryna Nekhomiazh 

'heart_disease_uci.py' file contains an implementation of Logistic Regression algorithm, preparation of Heart Disease UCI dataset and application of the algorithm on the dataset.
'medical_cost_personal.py' file contains an implementation of Linear Regression algorithm, preparation of Medical Cost Personal dataset and application of the algorithm on the dataset.


### 1. For which tasks is it better to use Logistic Regression instead of other models? 
For binary classification problems.
   
### 2. What are the most important parameters that you can tune in Linear Regression / Logistic Regression / SVM?
#### Linear Regression: 
- regularisation
- learning rate
#### Logistic Regression:
- penalty 
- C
- solver
#### SVM:
- kernel
- C

### 3. How does parameter C influence regularisation in Logistic Regression?
The parameter C is the inverse of regularization strength. Higher values of C specify less regularization — a high value of C tells the model to give high weight to the training data and a lower weight to the complexity penalty, it tells a model to "trust" the training data. A low value of C tells the model to give more weight to this complexity penalty at the expense of fitting to the training data.

### 4. Which top 3 features are the most important for each data sets?
To answer this question I used seaborn library, and a correlogram of features

#### Data set 1. Heart Disease UCI: 
The most important features are: 
1. thal: thalassemia, reversable defect
2. ca (number of major vessels colored by fluoroscopy)
3. asymptomatic chest pains

#### Data set 2. Medical Cost Personal Datasets:
1. smoker
2. age
3. body mass index


### 5. What accuracy metrics did you receive on train and test sets for Heart Disease UCI dataset?
 For train set accuracy is 0.8347457627118644
\
 For test set accuracy is 0.8166666666666667

For train set precision is 0.8165137614678899
\
For test set precision is 0.8461538461538461

(The accuracy of the LogisticRegression model of the sklearn library with basic configurations for a train set is 0.8601694915254238, for a test set  is 0.8166666666666667;\
precision for a train set is 0.8712871287128713, for a test set  is 0.8461538461538461)


### 6. What MSE did you receive on train and test datasets for Medical Cost Personal?
For train set MSE is 123239439.85483362
\
For test set MSE is 124235844.7911461

(The MSE of the LinearRegression model of the sklearn library with basic configurations for a train set is 34374619.15699605, for a test set  is 37137579.17188956)


### (additional)* Try a few different models and compare them. (You can use scikit-learn models for this)

#### For Heart Disease UCI dataset I tried 5 sklearn library models:
- SGDClassifier
- SVC
- MLPClassifier
- DecisionTreeClassifier
- RandomForestClassifier

SGDClassifier had the best accuracy score on test set: 0.8666666666666667 \
MLPClassifier (with parameters: solver = 'sgd', learning_rate = 'adaptive') had the best precision score on test set: 0.9285714285714286

#### For Medical Cost Personal Dataset I tried 5 sklearn library models:
- DecisionTreeRegressor
- RandomForestRegressor
- GradientBoostingRegressor
- KNeighborsRegressor
- AdaBoostRegressor

GradientBoostingRegressor (with loss parameter 'huber') had the best MSE on test set: 18911429.886206478
