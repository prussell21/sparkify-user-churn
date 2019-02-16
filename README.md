# Sparkify User Churn Prediction

## Project Overview

The goal of this project was to perform exploratory analysis, data processing, and machine learning on a sample of sparkify user data to predict user Churn.

With the sparkify dataset reaching 12GB in data, frameworks such as Apache Spark are needed to process and pipeline the information for modeling. In this example, the entire dataset is not used. The goal of this repo was to protoype the process before deplying to cloud services where it can scale to the entire dataset.

## Predicting Churn

Knowing when a user is likely to churn can help a company take pro-active actions to help retain that customer. Using features aggregated from user's activities or experiences on the platform can help build promising models for solving these problems.

I expect that by testing out several machine learning models and optimizing using the multiple metrics, there will be a model that returns a successful accuracy rate at predicing which users will churn.

## Understanding the Data

As said before the data used for this project was supplied by Udacity's Data Scientist Nanodgree program. The data contains a sample of user data records.

<img src="https://github.com/prussell21/sparkify-user-churn/blob/master/docs/images/dataset-schema.png?raw=true">

I decided to investigate the difference between page interactions such as Advertisements and Thumb Ups per song listened by each churned and non churned user groups.

In addition to the spotable difference in advertisemetns watched I decided to include Thumbs Up, Thumbs Down, and Add a friend to potential features for modelling.

### Cleaning

This dataset is relatively clean, with little to no missing values and errors that can be easily spotted. However, there were some missing user ID's in the orignal records. The first step was to remove these rows.

### Page Activity for Churned Users

<img src="https://github.com/prussell21/sparkify-user-churn/blob/master/docs/images/churned-page-count-table.png?raw=true">

### Page Activity for Non Churned Users

<img src="https://github.com/prussell21/sparkify-user-churn/blob/master/docs/images/non-churned-page-count-table.png?raw=true">

## Feature Engineering

In additiion to using page interaction as a feature for modeling, I decided to normalize this activity by the number of songs each user had listened to.

### Per Song Features

Adverts, Thumbs Up, Thumbs Down, Add a friend.

### Additional Features

Level (users are either on a paid of free subscription base)

### Per Song Dataset
After a bit of data manipulation, the per song per feature dataset looked like this.

```processed_data_pd.head()```

<img src="https://github.com/prussell21/sparkify-user-churn/blob/master/docs/images/processed-data-head.png?raw=true">

```processed_data_pd.describe()```

<img src="https://github.com/prussell21/sparkify-user-churn/blob/master/docs/images/processed_data-describe.png?raw=true">

### Vector Assembly

```    
ml_data = per_song_df.select(['Thumb Ups',
                                  'Thumb Downs',
                                  'Adverts',
                                  'Add Friend',
                                  'Level',
                                  'label'])
    
assembler = VectorAssembler(inputCols=['Thumb Ups',
                                           'Thumb Downs',
                                           'Adverts',
                                           'Add Friend',
                                           'Level'], outputCol="features")
    
features_assembled = assembler.transform(ml_data)
```

### Splitting the Data

```
label_and_features = assembled_data.select(['label', 'features'])
    
train, test, validation = label_and_features.randomSplit([0.60, 0.20, 0.20], seed=12345)
```

### Labels and Vector Assembled Features

```train.show()```

<img src="https://github.com/prussell21/sparkify-user-churn/blob/master/docs/images/vector-assembled-feats-labels.png?raw=true">

## Modeling

The labels in this dataset are not evenly weighted. In fact the target variable is a relatively small number in this dataset (~23%). Because of this, F1 Score and accuracy was used to optimize and validate several models as to which was superior.

### Pre Optimization Accuracy of Selected Models

#### Logistic Regression

Accuracy: 0.66

Area Under ROC: 0.71

#### Random Forest

Accuracy: 0.63

Area Under ROC: 0.64

#### Gradient Boosted Trees

Accuracy: 0.638

Area Under ROC: 0.65

### Refinement

Results of using Cross Validation and F1 Score to optimize

#### Logistic Regression

```
Validation accuracy: 0.6388

Cross Validaiton Best F1 Score: 0.7526

Best Parameters: maxIter: 10, regParam: 0.01

Test set accuracy: 0.7962
```

The validation accuracy improved for the logistic regression model after performing cross validation. According to the F1 scores accross each paramMap for this model, the model performs better the lower the regParam hyper parameter.

#### Random Forest

```
Validation accuracy: 0.6388

Cross Validation Best F1 Score: 0.7299

Best Parameters: maxDepth: 3, numTrees: 10

Test accuracy: 0.8148
```
Validation set accuracy remained the same after cross validation in this case. Due to the low number of users in the sparkify sample set it appears that the random forest classifier was unable to improve. The difference in maxDepth and numTrees did not effect its accuracy.

#### Gradient Boosted Trees

```
Validation accuracy: 0.6388

Cross Validation Best F1 Score: 0.7040

Best Parameters: maxDepth: 5, maxIter: 10

Test accuracy: 0.7037
```
Like the random forest classifier, the GBT model also did not improve its validation accuracy after optimizing with cross validation. I would attribute this to the low amount of sample data.

## Conclusion

In this case the superior model would be Random Forest. The Random Forest model boosted the best test set accuracy as well as the second best F1 score.

### Issues

Due to the extensive time it takes to train all of these models with cross validation, it was unfeesible to create a more robust hyper paramGrid for each individual model. Moving forward, it is highly recommened this process be refactored to be performed using cloud services such as AWS. 

In addition to the issues with time, the mini sparkiy datafile does not contain a sufficient amount of users (225) for successfully build a model. The entire datset of 12GB is needed in this case. It appears that the random forest and GBT models suffered from little training data, as they did not improve after cross validation.

### More Features?

More features such as including the rest of the pages that users can visit would most likely increase the accuracy of these models as well. However, due to the contrainst of computing power, many features would increase the time to train significantly.

## Requirements

Pyspark

Pandas

Numpy

Matplotlib

Seaborn

## Files in Repo

Sparify.ipynb : Notebook containing data exploration, feature extraction and machine learning on Sparkify sample data
README.md
Docs/Images: Table and data visualizations from Sparkify.ipynb
Index.md: Markdown for blog post

## Acknowledgements
