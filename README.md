<div align="center">
  <h1>Insurance Claims Analysis using Naive Bayes</h1>
</div>


<div align="center">
Contains an analysis of insurance claims data, focusing on incident causes, claim types, and potential fraud detection, utilizing data manipulation and visualization techniques.
</div>



<div align="center">
  <img src="https://github.com/Ras-codes/Insurance-Claims-Naive-Bayes/assets/164164852/0116f68e-ca4a-432a-902b-42517207de45">
</div>


## Tools

- **Programming Language**: Python 🐍
- **IDE**: Jupyter Notebook 📓
- **Data Manipulation and Analysis**:
  - NumPy 📊
  - pandas 🐼
  - SciPy 📐
  - statsmodels 📊
- **Data Visualization**:
  - Matplotlib 📊
  - Seaborn 📈
- **Date and Time Handling**: datetime ⏰


## Dataset Description: 

### Table- Claims


### Variables-

| Column Name         | Data Type | Description                         |
|---------------------|-----------|-------------------------------------|
| claim_id            | int       | Unique identifier for each claim    |
| customer_id         | int       | Unique identifier for each customer |
| incident_cause      | chr       | Cause of the incident               |
| claim_date          | date      | Date of the claim                   |
| claim_area          | chr       | Area where the claim occurred       |
| police_report       | chr       | Indicates if a police report exists |
| claim_type          | chr       | Type of claim                       |
| claim_amount        | chr       | Amount claimed                      |
| total_policy_claims | float     | Total claims made on the policy     |
| fraudulent          | chr       | Indicates if the claim is fraudulent|


# ------------------------------------------------------------------------------


# Naive Bayes Classifier

This project implements a Naive Bayes classifier on an Insurance Claims dataset to predict the likelihood of fraudulent claims. It involves data preprocessing, model training using Python's scikit-learn library, evaluation of model performance metrics, and visualization of results. The goal is to enhance fraud detection capabilities in insurance operations through machine learning techniques.


# ------------------------------------------------------------------------------


# Insights from the Dataset

- After importing the dataset, our first step is to check if the data is imported properly, we can use `Claims.shape` to check the number of observations (rows) and features (columns) in the dataset
- Output will be : ![image](https://github.com/Ras-codes/Insurance-Claims-Naive-Bayes/assets/164164852/d7ddb9e5-d416-4fe3-8b05-04fab2fc8ebc)
- which means that the dataset contains 1100 records and 10 variables.
- We will now use `Claims.head()` to display the top 5 observations of the dataset
- ![image](https://github.com/Ras-codes/Insurance-Claims-Naive-Bayes/assets/164164852/2f9d793c-b957-41d8-8e4f-e5ec984f3368)
- To understand more about the data, including the number of non-null records in each columns, their data types, the memory usage of the dataset, we use `Claims.info()`
- ![image](https://github.com/Ras-codes/Insurance-Claims-Naive-Bayes/assets/164164852/a8aa8a67-f72f-4232-8fe8-75e24df2558e)


# ------------------------------------------------------------------------------


# Data Preparation:

- Data can have different sorts of quality issues. It is essential that you clean and preperate your data to be analyzed.  Therefore, these issues must be addressed before data can be analyzed.
- Data preparation involved essential tasks such as converting data types to ensure consistency (e.g., numeric or categorical), cleaning by removing extraneous characters like '$', validating and formatting dates for consistency, removing irrelevant columns to focus on relevant variables, and employing binning techniques to categorize continuous variables, enhancing their interpretability in analyses.


# ------------------------------------------------------------------------------


# Data Interpretation

 This involves visually inspecting the bar heights and patterns to derive insights into the distribution, frequency, or comparison of categorical data represented in the plot.

- ![image](https://github.com/Ras-codes/Insurance-Claims-Naive-Bayes/assets/164164852/48729375-2138-4ad7-bb2f-453f7b0674c4)
- By seeing this bar chart we can say that the number of Normal Customers are more than the fraud customers.


# ------------------------------------------------------------------------------


# One-Hot Encoding 

One-hot encoding is a technique used to convert categorical variables into a binary matrix representation. Each category is transformed into a separate binary column, where a value of 1 indicates the presence of the category, and 0 indicates its absence. This approach allows categorical data to be used effectively in machine learning models.
![image](https://github.com/Ras-codes/Insurance-Claims-Naive-Bayes/assets/164164852/3dc668ab-8cc2-41c7-98ed-beabc4a0fddc)


# ------------------------------------------------------------------------------


# Handling Missing Values:

Next step is to check for missing values in the dataset. It is very common for a dataset to have missing values.

- `data.isna().sum()` isna() is used for detecting missing values in the dataframe, paired with sum() will return the number of missing values in each column.
- ![image](https://github.com/Ras-codes/Insurance-Claims-Naive-Bayes/assets/164164852/22c51cc9-0e33-4e21-8bcf-0c042a2ce745)
````
Claims_1['claim_amount'] = Claims_1['claim_amount'].fillna(Claims_1['claim_amount'].median())
Claims_1['total_policy_claims'] = Claims_1['total_policy_claims'].fillna(Claims_1['total_policy_claims'].median())
````
- We have filled the missing values by using median. (The values of median and mode are equal so we can use any of them to fill the missing values)


# ------------------------------------------------------------------------------


# Outlier Detection

Outliers are data points that deviate significantly from the overall pattern of the dataset and can indicate atypical or rare cases. Outliers in medical data can be indicative of unique cases or anomalies that deviate significantly from the general pattern, and their presence is something to be expected.

- To detect outliers in your dataset, you can use statistical methods or visualizations.
- Visualize the distribution of each numerical feature using box plots.
````
sns.boxplot(Claims_1.total_policy_claims)
````
![image](https://github.com/Ras-codes/Insurance-Claims-Naive-Bayes/assets/164164852/42bdd7e7-c9d0-4aa0-8b96-15dc47072dc1)
- Treating the outliers
````
Claims_3['total_policy_claims'] = Claims_3['total_policy_claims'].clip(lower = -0.5, upper = 3.5)
````


# ------------------------------------------------------------------------------


# Feature Engineering: Correlation Analysis

Correlation Analysis involves examining how different features (variables) in a dataset are related to each other. By calculating correlations, we determine the strength and direction of these relationships, helping us identify which features may influence each other and which ones are redundant. This analysis aids in selecting the most relevant features for predictive models, improving model performance by focusing on the most informative data attributes.

- Calculating the absolute correlation coefficients of each variable in Claims dataset with the 'fraudulent' column.
````
corr = Claims_3.corrwith(Claims_3['fraudulent']).abs().sort_values(ascending = False)
````
- ![image](https://github.com/Ras-codes/Insurance-Claims-Naive-Bayes/assets/164164852/bcb1087f-d506-4b23-9e2f-916c5d36881b)
- Filtering out only those variables whose absolute correlation with 'fraudulent' is greater than 0.03. This helps in selecting variables that have a relatively stronger correlation with the target variable.
- ![image](https://github.com/Ras-codes/Insurance-Claims-Naive-Bayes/assets/164164852/7f801645-0d11-4021-a4b0-f1fcc2ff2217)


# ------------------------------------------------------------------------------


# Split train-test data

Splitting train-test data refers to the process of dividing a dataset into two separate subsets: one for training a machine learning model and another for evaluating its performance.

### Defining Features and Target Variable:

````
x = Claims_3.drop(['fraudulent'], axis = 1)
y = Claims_3['fraudulent']
````
- Define x to contain all features except the target variable ('fraudulent').
- Define y to contain only the target variable ('fraudulent').

### Splitting Data:

````
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
````
- x_train and y_train for training the model.
- x_test and y_test for evaluating the model's performance.

#### Using 20% of the data for testing, leaving 80% for training.


# ------------------------------------------------------------------------------


# Naive Bayes Model:



- Model Selection: Uses Gaussian Naive Bayes (GNB), suitable for features with a Gaussian distribution assumption.
- Model Training: `gnb.fit(x_train, y_train)` trains the model using features `(x_train)` and corresponding target labels `(y_train)`.
- Prediction: `gnb.predict(x_train)` generates predictions for the training data.
- Confusion Matrix: `pd.crosstab(Model_data_train.fraudulent, Model_data_train.y_pred, margins=True)` creates a confusion matrix to evaluate model performance.
- ![image](https://github.com/Ras-codes/Insurance-Claims-Naive-Bayes/assets/164164852/3b525f62-d513-4d53-93c5-5ddfdc5d25d1)
- Metrics Calculation: Computes True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN) from the confusion matrix.
- Visualization: Heatmap visualizes the confusion matrix.
- ![image](https://github.com/Ras-codes/Insurance-Claims-Naive-Bayes/assets/164164852/b75aebe3-f7f0-4c14-9ef8-391c792db1d3)
- Classification Report: Provides precision, recall, F1-score, and support metrics.
- Accuracy Score: Computes the overall accuracy of the model on the training data.

#### Model accuracy score: 0.7716


# ------------------------------------------------------------------------------


# Evaluating the model


- Prediction on Test Data: `gnb.predict(x_test)` predicts the target variable (fraudulent in this case) using the trained Gaussian Naive Bayes model (gnb) on the test data (x_test).
- Comparison DataFrame: Model_data_test is created to compare actual (y_test) vs. predicted (y_test_pred) values.
- Classification Report: `classification_report(y_test, y_test_pred)` generates a report including precision, recall, F1-score, and support metrics for the test data.
- Accuracy Score: `accuracy_score(y_test, y_test_pred)` computes the overall accuracy of the model on the test data.

## Model accuracy score: 0.7773

#### It is not an overfitting model. We have divided the training and testing dataset in 80-20 ratio and we got almost similar training and testing accuracy.

### Conclusion : 
In this project, we set out to improve the accuracy of our model, initially achieving a 73% accuracy rate. By doing some Outlier Treatment and Correlation Analysis the model's accuracy rate has increased but importantly by reintroducing the claim_amount variable into our analysis, we observed a notable increase in accuracy, reaching 77%. This suggests that in this dataset, claim_amount variable plays a significant role in predicting the outcomes of our model.
