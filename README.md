# data-analyst-Harpreet
**Introduction**

The assignment analyzes the "Franchises Dataset," comprising data from 100 different
franchise locations. This dataset has various attributes that can help understand the factors
influencing each franchise's net profit. Specifically, the dataset includes the net profit (in a million
dollars) as the target variable, along with several key predictors: counter sales (in a million dollars),
drive-through sales (in a million dollars), the number of daily customers, the type of franchise, and
the location of the franchise.

In this assignment, I aim to employ Decision Tree and Random Forest Models to explore and
model the relationships between these variables to predict the net profit for each franchise. This
analysis helps forecast future profits and make informed strategic decisions to enhance business
performance across different franchises.

**Data Preprocessing and Analysis**

After getting a snapshot of the dataset, I realized every column except Business Type and Location
is numerical. As these machine learning algorithms cannot handle categorical and alphanumeric
data, I used the “One-hot Encoding” technique to convert both into matrices/vectors of binary
digits. After employing this technique, all columns became numerical, as shown below:

![image](https://github.com/user-attachments/assets/349a1f46-e5e9-4e0a-8465-f2f51932194e)

As evident, there are no missing values to treat. Outlier Removal is not required as these algorithms
are non-parametric in nature. Also, Feature scaling is not required as these algorithms are scale-
invariant.
Below is a quick snapshot of the Descriptive Statistics of the given dataset:

![image](https://github.com/user-attachments/assets/31bd205c-2b4c-4f7a-99bf-8bcc46e53c3f)

**(a) Development and Accuracy Assessment of Decision Tree Model**

**Development of Decision Tree Model**__

**Step 1 (Defining Predictors and Predictand):** After preprocessing the data, it entails eight
predictors and one target variable (Net Profit), as shown below:

![image](https://github.com/user-attachments/assets/2198d758-1d85-43f7-a8ce-5dd6f26c5f47)

I defined “X” as columns indexing from 1 to 8, while “y” as the first column (index 0).

**Step 2 (Splitting of Data):** Afterwards, I split the data into two parts- 70% as a training set and the
remaining 30% as a testing set.

**Step 3 (Model Implementation):** As the predictand is a continuous variable (Net Profit), I used the
Decision Tree “Regressor” to fit the model on the Training set. I did not specify the values of any
hyperparameters so that the model takes the default values, as shown below:

![image](https://github.com/user-attachments/assets/271ff72e-e324-46da-83e7-bf9cbe2e0292)

**Training Accuracy and Testing Accuracy.** After fitting the model, I checked training
accuracy using the R2 score, which came out to be 1.0, which means 100% of the variations in the
target variable are explained by descriptive features, i.e., it is a perfect (ideal/not real) model.
Similarly, I checked the accuracy of the same model, but this time, on the testing set, I found an
R2 score of 0.93, which implies that 93% of variations of the target variable can be explained by
descriptive features.

![image](https://github.com/user-attachments/assets/efcdf6a8-a542-4cf2-932f-fbcb1ed2c504)

This unusual observation suggests the presence of Overfitting in training data. This issue
can be resolved by performing tuning of various hyperparameters to find the best values of those
parameters, which results in a more generalized model.

**Tuning of Hyperparameters to Find the Best Model**

I used the technique called K-fold Cross Validation, using Grid Search. I defined the
following most crucial hyperparameters and their values in the grid:

![image](https://github.com/user-attachments/assets/1ebcb960-f468-48e8-8fba-d99ffd9009bb)

I performed 10-fold cross-validation to ensure an optimum number of iterations to find the
best possible model. After tuning of hyperparameters, I found the following list of best values of
hyperparameters:

![image](https://github.com/user-attachments/assets/9210f974-b233-4530-ae2c-cb768a853d0e)

So, the best model, which uses the above list of values of hyperparameters, attains the
following scores of performance metrics:

![image](https://github.com/user-attachments/assets/708df8f1-0fa6-409e-b377-f81957fc9c0a)

**Final Accuracy Assessment of the Best Decision Tree Model**

After executing this best model on the training and testing set, I found the following
performance metrics:

![image](https://github.com/user-attachments/assets/7d60c950-277c-4b8e-afe3-ea4dae996010)

Now, the difference between training and testing accuracy is reasonable. It is a general
norm that the model always performs a bit less accurately on unseen data.

**(b) Visualizations and Interpretations of the Decision Tree Model**

**Visualization of the Decision Tree Model**

![image](https://github.com/user-attachments/assets/460e1619-f3d1-44f9-b5fb-8307063338dd)

**Root Node and the First Internal Node**

![image](https://github.com/user-attachments/assets/d9370467-411c-4522-960b-bfb0f48492ef)

The best decision tree model, obtained after tuning of hyperparameters, has used “Business
Type Pizza Store” as the root node and the “Drive-through Sales” as the first internal node,
highlighting that these features split the data most purely, and hence, are most influential features
in contributing to the fabrication of this Decision Tree. This fact is further supported by the below
Feature Importance visualization.

**Feature Importance**

![image](https://github.com/user-attachments/assets/40c0a826-b2c3-46f9-a3f7-d1b4075af3aa)

The top 2 most influential features in the Decision Tree Model are: “Drive-through Sales”
and “Business Type_Pizza Store”. These contribute the maximum in making predictions and are
chosen as the first internal and root nodes, respectively.

**(c) Development of Random Forest (RF) Model**

Just like the foundational steps of developing the Decision Tree (DT) Model, I defined the same
descriptive features and the target feature (Net Profit). Also, I split the whole dataset again into
70% for training and 30% for testing. Then, I used “Random Forest Regressor” to fit and
implement the model on training data. Afterward, I evaluated the model's performance using
metrics, R2 score, and MSE (Mean Squared Error).

![image](https://github.com/user-attachments/assets/31433f2a-a9ce-4aa6-b301-585bfcf69187)

As the model’s R-squared score is over **95%**, there is a scope for tuning to prevent overfitting if it
exists.

**(d) Rationalization of Selected Structure of Random Forest (RF) Model**

I performed hyperparameter tuning on my trained RF Model, using the Randomized Search
Cross-Validation technique, as it is faster than Grid Search and is recommended for RF Models.
However, the trade-off exists so that it may find a good but not a great model. For this purpose, I
defined the following grid of hyperparameters to tune:

![image](https://github.com/user-attachments/assets/620382e4-919d-4994-9ccc-ef0445821fe9)

The best hyperparameters obtained are:

![image](https://github.com/user-attachments/assets/17396341-0af0-474a-90d9-3f194e9b80ac)

Now, I retrained my model using these best hyperparameter values and found out the best
possible R-squared and MSE for this model:

![image](https://github.com/user-attachments/assets/9869267c-4cc5-40d0-bb7e-629aaecd8311)

The accuracy provided by the RF Model is a little over **97%**.

**(e) Simulations, Visualizations, and Interpretations of Random Forest (RF) Model**

**Simulation of Model Parameters**

For this part, I have narrowed down the scope by choosing two parameters: “n_estimators”
and “max_depth”, and have enumerated a list of values in them to see their respective impact on
MSE (Mean Squared Error) and R-Squared score visually. Here is the enumerated list:

![image](https://github.com/user-attachments/assets/e7dfbcb3-64c4-41b2-a775-71b6c560543d)

**Here are the graphs simulating the impact of “n_estimators” on performance metrics:**
![image](https://github.com/user-attachments/assets/d7390d6f-121c-49aa-8afd-ac7ada7e2d53)

It can be seen that the above two graphs supplement each other. They evidently show the
value of best “n_estimators” lie somewhere around 400, because that is the only interval where R-
squared is highest and MSE is lowest. This information provided by graphs is further supported
by our hyper tuning results, which indicate the best value of “n_estimators” as 393, which is
exactly around 400.

**Below are the graphs simulating the impact of “max_depth” on performance metrics:**

![image](https://github.com/user-attachments/assets/e92d3a44-a1d4-473e-9813-b8b17fdabe23)

Both the above graphs again supplement the result of tuning of hyperparameters. As seen,
the best value of “max_depth” is 20, where R-squared is maximum and MSE is minimum. After
20, both the performance metrics are stable (plateau), irrespective of the value of “max_depth”.
This time, the graph result and hypertuning result is exactly identical (max_depth=20).

**Impact of Descriptive Features: Feature Importance**

Feature Importance quantifies the contribution of each feature to the model's predictive
performance. It provides insights into which features are most influential in making predictions,
helping to understand the underlying patterns and dependencies in the data.

![image](https://github.com/user-attachments/assets/6138f6b9-ec38-40ef-99d8-0ffe61e104cc)

The top 2 most influential features in the Random Forest model are: “Drive-through Sales”
and “Business Type_Burger store.” Like the DT Model, Locations do not contribute to making
predictions. This can be attributed to one of the limitations of given dataset, which is imbalanced data distribution. Since it had only 4% data for Richmond, the model had reduced the importance of “Location” feature. Moreover, the Rest of the descriptive features soundly influence predictions, too.

**Impact of Descriptive Features: Correlation Matrix**

![image](https://github.com/user-attachments/assets/794c6ef7-17d1-4541-887c-363367bd0c46)

This correlation matrix also reveals the same trend: high correlation with “Drive-through
Sales”, “Counter Sales” and “Business Type_Burger Store”. However, Pizza Store and Café are
negatively correlated. Furthermore, Locations and Number of Customers are insignificantly
correlated with Net Profit.

**(f)Particular Problem of Net Profit Forecasting**

As the Counter and drive-through sales have units- million $, I took them as $0.5 million
and $0.7 million, respectively. As the number of customers visiting the given franchise daily was
not provided, I imputed it as the mean of that column.

![image](https://github.com/user-attachments/assets/8322452e-a4b1-4e1b-8673-450aef6308a1)

The prediction of the target feature (Net Profit), which I obtained from both of the models,
is as follows:

![image](https://github.com/user-attachments/assets/66339864-1a81-4a83-9afe-b4dfebe27fbe)

According to the Decision Tree Model, the given franchise will make a net profit of $0.6
million ($600,000), while according to the Random Forest Model, It is $0.38 million ($380,000).

**Comparison and Commentary**
The Decision Tree’s higher prediction may reflect overfitting to specific data patterns in
the training set, leading to optimistic profit estimates. In contrast, the Random Forest's
lower prediction balances bias and variance, likely providing a more accurate reflection of
expected profitability.

By averaging the results of multiple trees, Random Forests tend to provide more robust and
generalizable predictions. The more conservative forecast by the RF model might be more
realistic and dependable for business decision-making.

Conclusively, relying on the Random Forest forecast might be wiser for strategic planning due
to its robustness and lower risk of overestimation. The decision-makers should consider the RF
model’s prediction as it potentially offers a more cautious and realistic estimate of the net profit.

**(g)Roles of “max_features” and “n_estimators”**

**“max_features”: Trade-off between Variance and Bias**

This hyperparameter in a Random Forest Regressor is crucial in determining the number
of features to consider when looking for the best split at each node. By randomly selecting a subset
of features, it introduces diversity among the trees in the forest, which helps to improve the overall
performance and robustness of the model. The choice of “max_features” can significantly impact
the bias-variance tradeoff and the computational efficiency of the model. It can take four values:
“auto”, “sqrt”, “log2”, and “None.”

![image](https://github.com/user-attachments/assets/8597fba6-fee7-40d6-8194-812e5d98b2b6)

Here, “n_features” refers to the total number of features in the dataset. Also, “auto” and
“sqrt” imply the same operation. Moreover, the default value of this parameter in Scikit-learn is
1.0.

**“n_estimators”: Trade-off between Accuracy/Generalization and Computational Cost/Time**

This hyperparameter in a Random Forest Regressor specifies the number of individual
decision trees in the forest. The random forest algorithm works by building multiple decision trees
and aggregating their results to improve the predictive performance and robustness of the model.
Each tree in the ensemble contributes to the final prediction by voting or averaging, which helps
to reduce the overall variance and prevent overfitting.

Increasing the number of trees generally enhances generalization by capturing diverse
patterns and reduces variance without the risk of overfitting. However, more trees also increase
training and prediction time. The default of 100 trees often provides a good balance.

**Interesting Finding:** Unlike other models, increasing the number of trees in a random forest
does not cause overfitting. This is because the trees in a random forest are trained on different
subsets of the data and are combined in a way that averages out the errors.

**(h)Assumptions and Limitations**

**Assumptions of the Decision Tree Model**

• Observations are independently drawn from the population.

• The training and test data come from the same distribution.

• The chosen features are relevant and sufficient for predicting the target variable.

**Limitations of the Decision Tree Model**

• Prone to overfitting, especially with deep trees, capturing noise and specific patterns in
the training data that do not generalize well.

• Makes locally optimal splits, which might not lead to the globally optimal tree.
Performance can degrade on imbalanced datasets where one class dominates.

• Deep trees can result in leaves with very few data points, leading to unreliable
predictions.

**Assumptions of the Random Forest Model**

• Individual trees are somewhat independent due to bootstrap sampling and random feature
selection.

• Feature importance remains relatively stable across different bootstrap samples.

**Limitations of the Random Forest Model**

• Training and prediction can be computationally intensive with a large number of trees.
Also, storing multiple trees requires significant memory.

• The ensemble nature makes the model less interpretable than a single decision tree.

• It can be biased towards features with more categories.

• Feature importance measures can be misleading in the presence of correlated features.

**Conclusion**

My Decision Tree and Random Forest Models provided an accuracy of 97% and 95%,
respectively. In conclusion, Decision Trees excel in interpretability and computational
efficiency but may suffer from overfitting. Random Forests overcome this constraint by improving
prediction accuracy and robustness through ensemble learning, although with diminished
interpretability.

**References**

MLK. (2023, February 17). Decision Tree Regression in Python Sklearn with Example. MLK -
Machine Learning Knowledge. https://machinelearningknowledge.ai/decision-tree-regression-in-python-sklearn-with-example/#google_vignette

scikit-learn. (n.d.). sklearn.tree.DecisionTreeRegressor — scikit-learn 0.23.2 documentation.
Scikit-Learn.org. https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html

scikit-learn. (2018). 3.2.4.3.2. sklearn.ensemble.RandomForestRegressor — scikit-learn 0.20.3
documentation. Scikit-Learn.org. https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html

Yse, D. L. (2020, August 31). Modelling Regression Trees. Medium.
https://towardsdatascience.com/modelling-regression-trees-b376e959d02e
