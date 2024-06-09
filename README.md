## Heart Attack Data Set Analysis and Machine Learning

The main aim of this project is to develop a machine learning model capable of predicting the risk of a heart attack.

Within this scope, the "Heart Attack Analysis & Prediction Dataset" available on Kaggle will be utilized to analyze various health parameters of individuals, and the impact of these parameters on the risk of heart attack will be evaluated.

# Project Stages

1 - Data Exploration and Preprocessing - Consists of 3 steps: Understanding the Dataset, Data Cleaning, and Data Transformation.

2 - Data Analysis and Visualization - Consists of 2 stages: Exploratory Data Analysis (EDA) and Correlation Analysis.

3 - Detection of Outliers and Data Cleaning

4 - Model Development - Consists of 3 stages: Model Selection, Model Training, and Hyperparameter Optimization.

5 - Model Evaluation and Testing - Consists of 3 stages: Model Performance Metrics, Cross-Validation, and Final model selection.

6 - Results - The applicability of the model is verified. If the model is applicable in the health domain, it is selected.

<div align="center">
<img width="133" alt="path" src="https://github.com/sensoyyasin/heartdisease_prediction/assets/73845925/e7a9874e-de57-44ee-aeaf-8db913297e9b">
</div>

# Information about the Dataset

There are a total of 13 different variables in our dataset. These are:

1 - age - age (in years)

2 - sex - gender (1 = male; 0 = female)

3 - cp - chest pain type (1 = typical angina; 2 = atypical angina; 3 = non-anginal pain; 0 = asymptomatic)

4 - trestbps - resting blood pressure (in mm Hg on admission to the hospital)

5 - chol - serum cholesterol (in mg/dl)

6 - fbs - fasting blood sugar > 120 mg/dl (1 = true; 0 = false)

7 - restecg - resting electrocardiographic results (1 = normal; 2 = having ST-T wave abnormality; 0 = hypertrophy)

8 - thalach - maximum heart rate achieved

9 - exang - exercise-induced angina (1 = yes; 0 = no)

10 - oldpeak - ST depression induced by exercise relative to rest

11 - slope - the slope of the peak exercise ST segment (2 = upsloping; 1 = flat; 0 = downsloping)

12 - ca - number of major vessels colored by fluoroscopy (0-3)

13 - thal - 2 = normal; 1 = fixed defect; 3 = reversible defect

14 - num - target feature - diagnosis of heart disease (angiographic disease status) (Value 0 = < 50% diameter narrowing; Value 1 = > 50% diameter narrowing)


# Model Comparison

<img width="796" alt="Ekran Resmi 2024-06-09 16 33 04" src="https://github.com/sensoyyasin/heartdisease_prediction/assets/73845925/9d6d5f1a-bc3a-48c9-8b16-240932e8f080">

# Project Outcome

Models are created using Logistic Regression, Decision Trees, Support Vector Machines, and Random Forest algorithms, and accuracy scores are calculated. We evaluated the performance of the models using cross-validation.
List of activities carried out within the scope of the project:

1 - Firstly, we prepared the dataset for Exploratory Data Analysis (EDA).

2 - We conducted Exploratory Data Analysis (EDA).

3 - Within the scope of univariate analysis, we analyzed numerical and categorical variables using Distplot and Circular Charts.

4 - Within the scope of bivariate analysis, we analyzed variables using FacetGrid, Countplot, Pairplot, Swarmplot, Boxplot, and Heatmap.

5 - We prepared the dataset for modeling. In this context, we dealt with missing and outlier values.

6 - We used four different algorithms in the modeling stage.

7 - We achieved <b>87% accuracy and 88% AUC with the Logistic Regression model</b>.

8 - We achieved <b>83% accuracy and 85% AUC with the Decision Tree Model</b>.

9 - We achieved <b>83% accuracy and 89% AUC with the Support Vector Classifier Model</b>.

10 - And we achieved <b>90.3% accuracy and 93% AUC with the Random Forest Classifier Model </b>.

11 - Considering all these model outputs, <b>we preferred the model created with the Random Forest Algorithm, which provided the best results</b>.
