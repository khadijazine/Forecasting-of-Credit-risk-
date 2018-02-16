import sys
from os import environ, path

environ.update(
    {'SPARK_HOME': '/sparkdirectory'})
spark_home = environ.get('SPARK_HOME')
sys.path.insert(0, spark_home + "/python")
sys.path.insert(0, path.join(spark_home, 'python/lib/py4j-0.10.3-src.zip'))

from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, NaiveBayes


# Random Forest
def random_forest(train_data, test_data):
    rf = RandomForestClassifier()
    pipeline = Pipeline(stages=[rf])
    # Create ParamGrid for Cross Validation
    paramGrid = ParamGridBuilder().addGrid(rf.maxDepth, [5, 10]) \
        .addGrid(rf.maxBins, [25, 31]) \
        .addGrid(rf.minInfoGain, [0.01, 0.001]) \
        .addGrid(rf.numTrees, [20, 60]) \
        .addGrid(rf.impurity, ["gini", "entropy"]) \
        .build()

    evaluator = BinaryClassificationEvaluator(labelCol="label")
    crossValidator = CrossValidator(estimator=pipeline,
                                    evaluator=evaluator,
                                    estimatorParamMaps=paramGrid,
                                    numFolds=10)

    # use the Random Forest Classifier to train (fit) the model
    cv = crossValidator.fit(train_data)
    # and Get the best Random Forest model
    best_model = cv.bestModel.stages[0]
    best_model.save("model/LR1")
    prediction = cv.transform(test_data)
    metric = evaluator.evaluate(prediction)
    print "The metric of test's accuracy= %g" % metric


# Logistic Regression
def logistic_regression(train_data, test_data):
    # Create initial Logistic Regression model
    lr = LogisticRegression(maxIter=10)
    pipeline = Pipeline(stages=[lr])
    # Create ParamGrid for Cross Validation
    paramGrid = (ParamGridBuilder()
                 .addGrid(lr.regParam, [0.01, 0.5, 2.0])
                 .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])
                 .addGrid(lr.maxIter, [1, 5, 10])
                 .build())

    evaluator = BinaryClassificationEvaluator(labelCol="label")
    crossValidator = CrossValidator(estimator=pipeline,
                                    evaluator=evaluator,
                                    estimatorParamMaps=paramGrid,
                                    numFolds=10)

    # use the Logistic Regression Classifier to train (fit) the model
    # and Get the best Logistic Regression model
    cv = crossValidator.fit(train_data)
    best_model = cv.bestModel.stages[0]
    best_model.save("model/RFM1")
    prediction = cv.transform(test_data)
    metric = evaluator.evaluate(prediction)
    print "The metric of test's accuracy= %g" % metric


def naive_Bayes(train_data, test_data):
    nb = NaiveBayes(smoothing=1.0, modelType="multinomial")
    pipeline = Pipeline(stages=[nb])
    paramGrid = ParamGridBuilder().addGrid(nb.smoothing, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5]).build()
    evaluator = BinaryClassificationEvaluator(labelCol="label")
    crossValidator = CrossValidator(estimator=pipeline,
                                    evaluator=evaluator,
                                    estimatorParamMaps=paramGrid,
                                    numFolds=10)
    model = crossValidator.fit(train_data)
    # Fetch best model
    best_model = model.bestModel.stages[0]
    best_model.save("model/NB1")
    predictions = model.transform(test_data)
    accuracy = evaluator.evaluate(predictions)
    print "Model Accuracy with Cross Validation: ", accuracy
