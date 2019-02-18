# Sparkify User Churn Prediction

<img src="https://github.com/prussell21/sparkify-user-churn/blob/master/docs/images/user-churn-header.png?raw=true">

## Project Overview

The goal of this project was to perform exploratory analysis, data processing, and machine learning on a sample of sparkify user data to predict user Churn.

With the sparkify dataset reaching 12GB in data, frameworks such as Apache Spark are needed to process and pipeline the information for modeling. In this example, the entire dataset is not used. The goal of this repo was to protoype the process before deplying to cloud services where it can scale to the entire dataset.

## Predicting Churn

Knowing when a user is likely to churn can help a company take pro-active actions to help retain that customer. Using features aggregated from user's activities or experiences on the platform can help build promising models for solving these problems.

I expect that by testing out several machine learning models and optimizing using the multiple metrics, there will be a model that returns a successful accuracy rate at predicing which users will churn.

## Understanding the Data

Data used for this project was supplied by Udacity's Data Scientist Nanodgree program. The data contains a sample of user data records.

<img src="https://github.com/prussell21/sparkify-user-churn/blob/master/docs/images/dataset-schema.png?raw=true">

```
data.describe('sessionId').show()
print (data.count(), ' total records in dataset.')
```
<img src="https://github.com/prussell21/sparkify-user-churn/blob/master/docs/images/total-records.png?raw=true">

In it's raw form this dataset contains the records for each page(interaction) for each user. 

#### Cleaning and Preprocessing

This dataset is relatively clean, with little to no missing values and errors that can be easily spotted. However, there were some missing user ID's in the orignal records. The first step was to remove these rows.

Additionally, houra and day timestamp columns were created to help with data exploration and potential features. Most importantly of all, Churn is defined as a user confirming their cancellation.

```
data = data.dropna(how='any', subset= ['userId', 'sessionId'])
data = data.filter(data['userId'] != "")
    
#Create new hour column from timestamp
get_hour = udf(lambda x: datetime.datetime.fromtimestamp(x / 1000.0). hour)
data = data.withColumn("hour", get_hour(data.ts))

#Create new day column from timestamp
get_day = udf(lambda x: datetime.datetime.fromtimestamp(x / 1000.0). day)
data = data.withColumn("day", get_day(data.ts))
    
#Defining Churn
cancellation_confirmation = udf(lambda x: 1 if x == "Cancellation Confirmation" else 0, IntegerType())
```
#### Data Exploration

I decided to investigate the difference between page interactions such as Advertisements and Thumb Ups per song listened by each churned and non churned user groups.

In addition to the spotable difference in advertisements watched I decided to include Thumbs Up, Thumbs Down, and Add a friend to potential features for modelling.

<img src="https://github.com/prussell21/sparkify-user-churn/blob/master/docs/images/page-breakdown-image.png?raw=true">

#### Page Activity for Churned Users

<img src="https://github.com/prussell21/sparkify-user-churn/blob/master/docs/images/churned-page-count-table.png?raw=true">

#### Page Activity for Non Churned Users

<img src="https://github.com/prussell21/sparkify-user-churn/blob/master/docs/images/non-churned-page-count-table.png?raw=true">

There is clearly a difference in page activity between users who churned and those who stayed. I decided to examin if there were any differences in the 'level' of a user and the likelihood of churning.

<img src="https://github.com/prussell21/sparkify-user-churn/blob/master/docs/images/free-vs-paid-breakdown.png?raw=true">

## Feature Engineering

In additiion to using page interaction as a feature for modeling, I decided to normalize this activity by the number of songs each user had listened to.


#### Selected Features

'Adverts', 'Thumbs Up', 'Thumbs Down', and 'Add a friend' per song listened to. I also included the user's level (free or paid). 

#### Per Song Dataset
After a bit of data manipulation, the per song per feature dataset looked like this.

```
##Creating features and creating 'one hot encoded' dataset for selected page variables
thumbs_up = udf(lambda x: 1 if x == "Thumbs Up" else 0, IntegerType())
data = data.withColumn("Thumbs Up", thumbs_up("page"))

thumbs_down = udf(lambda x: 1 if x == "Thumbs Down" else 0, IntegerType())
data = data.withColumn("Thumbs Down", thumbs_down("page"))

songs = udf(lambda x: 1 if x == "NextSong" else 0, IntegerType())
data = data.withColumn("Song", songs("page"))

adverts = udf(lambda x: 1 if x == "Roll Advert" else 0, IntegerType())
data = data.withColumn("Advert", adverts("page"))

add_friend = udf(lambda x: 1 if x == "Add Friend" else 0, IntegerType())
data = data.withColumn("Add Friend", add_friend("page"))

level = udf(lambda x: 1 if x == "paid" else 0, IntegerType())
data = data.withColumn("Level", level("level"))

userId = udf(lambda x: int(x), IntegerType())
data = data.withColumn("User ID", userId("userId"))

#Selecting feature and target from larger dataset
model_data = data.select(['User ID','Thumbs up', 'Thumbs down', 'Song', 'Advert', 'Add Friend', 'Level', 'Churn'])

#Grouping by users and counting total page interactions
sums = model_data.groupBy('User ID').agg(_sum('Song').alias('Songs'),
                                             _sum('Thumbs Up').alias('Thumb Ups'),
                                             _sum('Thumbs Down').alias('Thumb Downs'),
                                             _sum('Advert').alias('Adverts'),
                                             _sum('Add Friend').alias('Add Friend'),
                                             _max('Level').alias('Level'),
                                             _max('Churn').alias('label'))
                                             
```

```processed_data_pd.head()```

<img src="https://github.com/prussell21/sparkify-user-churn/blob/master/docs/images/processed-data-head.png?raw=true">

```processed_data_pd.describe()```

<img src="https://github.com/prussell21/sparkify-user-churn/blob/master/docs/images/processed_data-describe.png?raw=true">

#### Vector Assembly

Spark MLlib requires the feature inputs to be vectorized for training and prediction. This is performed using the VectorAssembler class as shown below. 

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

#### Splitting the Data

```
label_and_features = assembled_data.select(['label', 'features'])
    
train, test, validation = label_and_features.randomSplit([0.60, 0.20, 0.20], seed=12345)
```

#### Labels and Vector Assembled Features

```train.show()```

<img src="https://github.com/prussell21/sparkify-user-churn/blob/master/docs/images/vector-assembled-feats-labels.png?raw=true">

Training dataset size: 125

Validation dataset size: 50

Test dataset size: 50

## Modeling

The labels in this dataset are not evenly weighted. In fact the target variable is a relatively small number in this dataset (~23%). Because of this, F1 Score and accuracy was used to optimize and validate several models as to which was superior.

#### Metrics and Evaluation

```
def evaluate_model(preds):
    '''
    Evaluates models accruacy and area under roc
    
    INPUT
    preds: models set of predicitons and labels
    
    OUTPUT
    Accuracy
    Area Under ROC
    '''

    #instantiate multiclass evaluator
    evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
    print("Accuracy: " + str(evaluator.evaluate(preds, {evaluator.metricName: "accuracy"})))
    
    #instantiate binaryclassification evaluator
    evaluator2 = BinaryClassificationEvaluator()
    print("Area Under ROC: " + str(evaluator2.evaluate(preds, {evaluator2.metricName: "areaUnderROC"})))
```

As a metric for evaluation I decided to use both accuracy and area under ROC to validate each model. For refinement, and optmization I have included cross validation with F1 score as the evaluator.

```
def cross_validation(model, paramGrid, folds=3):
    
    '''
    Cross_validation and optimization with f1 score
    
    INPUT
    model: instantiated model
    pramGrid: specific prameter grid for given model
    folds(int): number of folds to use for cross validaiton
    
    OUTPUT
    optimal cross_validated predictions dataset
    '''
    
    #evaluate cross validation best model using f1 score
    evaluator = MulticlassClassificationEvaluator(metricName="f1")
    
    cv = CrossValidator(estimator=model, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=folds, seed=42)
    
    #train cvModel
    cvModel = cv.fit(train)
    
    #user best model to predict on the validation set
    best_model = cvModel.bestModel
    
    cvpredictions = best_model.transform(validation)
    
    return cvpredictions, cvModel
 ```

#### Training

##### Logistic Regression

```
Accuracy: 0.66

Area Under ROC: 0.71
```

##### Random Forest

```
Accuracy: 0.63

Area Under ROC: 0.64
```

##### Gradient Boosted Trees

```
Accuracy: 0.638

Area Under ROC: 0.65
```

#### Refinement

Results of using Cross Validation and F1 Score to optimize

##### Logistic Regression

```
Validation accuracy: 0.6388

Cross Validaiton Best F1 Score: 0.7526

Best Parameters: maxIter: 10, regParam: 0.01

Test set accuracy: 0.7962
```

The validation accuracy improved for the logistic regression model after performing cross validation. According to the F1 scores accross each paramMap for this model, the model performs better the lower the regParam hyper parameter.

##### Random Forest

```
Validation accuracy: 0.6388

Cross Validation Best F1 Score: 0.7299

Best Parameters: maxDepth: 3, numTrees: 10

Test accuracy: 0.8148
```
Validation set accuracy remained the same after cross validation in this case. Due to the low number of users in the sparkify sample set it appears that the random forest classifier was unable to improve. The difference in maxDepth and numTrees did not effect its accuracy.

##### Gradient Boosted Trees

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

