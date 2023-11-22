# Implementation of Logistic and Linear Regression and their application on two different datasets

### File Descriptions
- `heart_disease_uci.py`: Implements Logistic Regression on the Heart Disease UCI dataset.
- `medical_cost_personal.py`: Implements Linear Regression on the Medical Cost Personal dataset.

## Questions:
### 1. For which tasks is it better to use Logistic Regression?
Logistic Regression is best suited for binary classification tasks.

### 2. What are the most important parameters that you can tune in Linear Regression / Logistic Regression / SVM?
- **Linear Regression**: Regularisation, Learning Rate
- **Logistic Regression**: Penalty, C (Inverse Regularization Strength), Solver
- **SVM (Support Vector Machine)**: Kernel, C (Regularization Parameter)

### 3. How does parameter C influence regularisation in Logistic Regression?
The parameter C inversely influences regularization strength in Logistic Regression. High values of C indicate lower regularization, prioritizing the fit to training data, while low values emphasize regularization, reducing overfitting.

### 4. Which top 3 features are the most important for each data set?
- **Heart Disease UCI Dataset**:
  1. Thalassemia (Reversable Defect)
  2. Number of Major Vessels Colored by Fluoroscopy (ca)
  3. Asymptomatic Chest Pains
- **Medical Cost Personal Dataset**:
  1. Smoker
  2. Age
  3. Body Mass Index

### 5. What accuracy metrics did you receive on train and test sets for the Heart Disease UCI dataset?
- Train Set: 
  - Accuracy: 83.47%
  - Precision: 81.65%
- Test Set: 
  - Accuracy: 81.67%
  - Precision: 84.62%
- Comparison with sklearn's LogisticRegression defaults: 
  - Train Accuracy: 86.02%, Precision: 87.13%
  - Test Accuracy: 81.67%, Precision: 84.62%

### 6. What MSE did you receive on train and test datasets for Medical Cost Personal?
- Train Set MSE: 123,239,439.85
- Test Set MSE: 124,235,844.79
- Comparison with sklearn's LinearRegression defaults:
  - Train MSE: 34,374,619.16
  - Test MSE: 37,137,579.17

### Additional: Model Comparisons

#### Heart Disease UCI Dataset
- **SGDClassifier**: Best Test Accuracy: 86.67%
- **MLPClassifier** (sgd, adaptive learning rate): Best Test Precision: 92.86%

#### Medical Cost Personal Dataset
- **GradientBoostingRegressor** (Huber loss): Best Test MSE: 18,911,429.89

---

This project was a test task for an internship.
