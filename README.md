# **Internet Food Analysis**

## **Dataset**
The dataset 'onlinefoods.csv' we are using for this study is being fetched from Kaggle. This dataset contains 388 rows and 13 features including the unnecessary columns, which are of type float64 (2), int64 (3), and object (8).

You can get the Dataset here: [Online Foods](https://www.kaggle.com/code/yinn94/food-visualization-classification-acc-0-91/input)

## **Methodology**
### Data Acquisition and Preprocessing
- Data Source: We have collected the data from Kaggle, "onlinefoods.csv", containing the data relevant to food orders, such as customer info details and their feedback.
- Import required libraries: In this step we import the required libraries like numpy, pandas, sea born, matplotlib time, label encoder and some other metrics libraries like accuracy score, precision score, f1 score etc.
- Load the dataset: In this step we load the dataset called onlinefood.csv file by using pandas.
- Basic Checks: Here we do basic checks like info of the data, if the data has any null values, describe the data to know mean, standard deviation to each column.
- We dropped a column which is named 'Unnamed: 12’ in the data.
- Exploratory Data Analysis: In this stage we plot graphs by using seaborn and matplotlib libraries and we compared columns how they behave.
- Label Encoder: Label encoder is used to convert the objective type data to numerical data. In this stage we change the objective data to numerical data by using replace function or label encoder library.
- Splitting the data for training and testing data: In this step we split the data for training and testing purpose with test size 0.25.
- Model implementation:  Here we write the code for Random Forest and XG Boost, SVM Classifiers, and Gradient Boosting Classifier from scratch for the training and testing of the data. Provide arguments like n estimators, max depth, min sample split, learning rate etc.
- Training and testing the data: In this step we train the data in both the models by using fit() and also we test the data in all models. We have also found the time taken for training the data in models by importing time library.
- Evaluating the models: In this step we found accuracy score, precision score, recall score, f1 score for the testing data in XG Boost and Random Forest, SVM Classifiers, Gradient Boosting Classifier, Polynomial kernel function SVM and Radial Basis kernel function SVM models and we have done confusion matrix visualization for all models.
- We plotted ROC Curve and Precision Recall curves.

## **Models used**
- Random Forest Classifier
- XGBoost
- Gradient Boosting Classifer

![Confusion Matrices of all theree models](https://github.com/neehanthreddym/internet-food-analysis/assets/167118432/033e9a63-2d48-4bd0-9c7c-8126baccbad9)
- Because the Random Forest model has the highest values off the diagonal, it seems to have the most overall errors.
- Given that it has the lowest values off the diagonal, the XGBoost model seems to work the best.
- When it comes to class 1 prediction, the Gradient Boosting model seems to make some mistakes, but it's difficult to compare these to the Random Forest model's errors.

## **Precision-Recall Curve**
![image](https://github.com/neehanthreddym/internet-food-analysis/assets/167118432/430e6c9c-9e91-4dfd-b6b2-b208e26854c9)

## **Evaluation metrics**

|Model              | Accuarcy           | Precision          | Recall             | F1-score           |
| ----------------- | ------------------ | ------------------ | ------------------ | ------------------ |
| Random Forest     | 0.905511811023622  | 0.8615384615384616 | 0.9491525423728814 | 0.903225806451613  |
| XGBoost           | 0.937007874015748  | 0.9180327868852459 | 0.9491525423728814 | 0.9333333333333333 |
| Gradient Boosting | 0.9291338582677166 | 0.9464285714285714 | 0.8983050847457628 | 0.9217391304347826 |

- `XGBoost`: The evaluation metrics indicate that XGBoost is the top performer. As proven by the F1-score, it gains the maximum accuracy and strikes a good balance between recall and precision.
- `Gradient Boosting Classifier`: Although it has the highest precision, it may miss a larger percentage of real positive cases due to its lower recall and F1-score when compared to XGBoost.
- `Random Forest`: Out of the three, it has the lowest accuracy and F1-score but is the fastest to train.

## **Limitations**
- Predictions in real-world scenarios may be biased or inaccurate due to limitations or biases in the data used.
- The list excludes additional variables like user demographics, the time of day, promotions, and weather that may affect food ordering behavior. Increasing the number of features in the model may help it perform better.
- It is possible that the models require retraining or fine-tuning for varying customer demographics or geographic locations, and that the dataset utilized does not accurately represent the entire population.
- Gradient Boosting Classifier's performance metrics were slightly lower than XGBoost's, despite providing improved interpretability.

# **Conclusion**

While all three models performed well, XGBoost showed a tiny advantage in terms of accuracy and F1-score. XGBoost performed exceptionally well in testing speed, while Random Forest provided the fastest training times. Even though it was slower, the gradient boosting classifier offered a model that might be easier to understand. All three of the models may be useful resources for meal delivery services, according to our evaluation metrics and confusion matrices. By reviewing customer feedback and personalizing recommendations based on their location, they can optimize delivery decisions and ultimately improve the dining experience for their customers.
