# Who to call?

<img src="/Images/audience-targeting.png" alt="Audience Targeting" >

*By Nadine Amersi-Belton*

## Problem statement

Bank XYZ has onboarded 2000 new customers through acquiring a smaller bank. Bank XYZ launches a telemarketing campaign to incite these customers to subscribe to a term deposit. Due to budget and ressource constraints, only 500 of these new customers can be contacted in this initial stage.

Using data from bank XYZ's customer base and results of the campaign for its existing customers, the goal is to identify which of the 500 customers to contact to maximise revenue and provide recommendations for future campaigns.

## Components

* **Jupyter Notebook**

The [Jupyter Notebook](https://nbviewer.jupyter.org/github/nadinezab/bank-marketing/blob/master/bank-marketing.ipynb) is our key deliverable and contains details of our approach and methodology, data cleaning, exploratory data analysis, model training and tuning and recommendations.

* **Presentation**

The [presentation](https://github.com/nadinezab/bank-marketing/blob/master/presentation.pdf) gives a high-level overview of our approach, findings and recommendations for non-technical stakeholders. It is aimed to be between 5 and 10 minutes long.

* **Data**

The dataset can be found in the file *bank-additional-full.csv* in the Data folder, in this repository. It was originally from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing). 

* **Blog Post**

A blog post was created as part of this project.

## Results and recommendations

A term deposit subscription was valued at USD 100 and through domain expertise, we know that the average uptake is around 10-15%. As such choosing the 500 customers to target randomly would have resulted in expected revenue between USD 5,000 and USD 7,500.

However by gathering data and using an XGBoost classifier (accuracy 85%) , we were able to **increase revenue to USD 14,500**.

A summary of the key steps which led us to this result are as follows:

1. We gathered data relating to customers' personal profile (job, age, education etc), financial (housing loan, personal loan, default) and economic indices (consumer price index). We had around 40,000 entries on which to train our algorithm.
2. We cleaned the data, replacing unknown values using K-Nearest Neighbours imputation
3. We adjusted for the class imbalance by using SMOTE
4. We established a custom profit metric using valuations for revenue and call cost to evaluate our models, together with recall and F1 score.
5. We trained the following classifiers and tuned them using RandomizedGridSearchCV: Logistic Regression, Decision Tree, K-Nearest Neighbours, Naive Bayes, Support Vector Classification, Adaboost, Gradient Boosting and XGBoost. Based on our profit metric, we selected XGBoost as our best performing model.
6. We further fine tuned the paramaters of XGBoost using GridSearchCV.
7. We applied our final model to the test set and chose the 500 customers with highest probability of being classified as subscribers.

<img src="/Images/ROC.png" alt="Train and Test ROC curves" >

The 5 features which had the highest impact on the classification are:
1. number of employees (a quarterly metric which represents economic state)
2. consumer confidence index
3. whether the customer has a university degree
4. whether the customer's job falls into the admin category
5. whether the customer is divorced.

<img src="/Images/feats.png" alt="Top 5 Features" >

Recommendations:
* Focus on students and retired customers
* Focus on repeat customers as clients who previously subscribed to a term deposit are more likely to do so again (65%)
* Conduct calls on cellular where possible

<img src="/Images/job.png" alt="Rate of subscription amongst job categories" >

## Contact

* If you have any questions, you can contact me at nzamersi@gmail.com