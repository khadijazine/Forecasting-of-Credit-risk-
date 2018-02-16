import sys
from os import environ, path

environ.update(
    {'SPARK_HOME': '/sparkdirectory'})
spark_home = environ.get('SPARK_HOME')
sys.path.insert(0, spark_home + "/python")
sys.path.insert(0, path.join(spark_home, 'python/lib/py4j-0.10.3-src.zip'))

from pyspark.ml.classification import RandomForestClassifier, LogisticRegression, NaiveBayes
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline


def random_Forest(train_data, test_data):
    # Create initial Random Forest model
    print ("Accuracy of Random Forest Classifier :")
    rf = RandomForestClassifier()
    model = rf.fit(train_data)
    predictions = model.transform(test_data)
    evaluator = BinaryClassificationEvaluator(labelCol="label")
    accuracy = evaluator.evaluate(predictions)
    print "The  accuracy = %g" % accuracy


def logistic_reg(train_data, test_data):
    # Create initial Logistic regression model
    print ("Accuracy of Logistic Regression Classifier :")
    rf = LogisticRegression()
    model = rf.fit(train_data)
    predictions = model.transform(test_data)
    evaluator = BinaryClassificationEvaluator(labelCol="label")
    accuracy = evaluator.evaluate(predictions)
    print "The  accuracy = %g" % accuracy


def naive_Bayes(train_data, test_data):
    # Create initial Naive Bayes model
    nb = NaiveBayes(smoothing=1.0, modelType="multinomial")
    pipeline = Pipeline(stages=[nb])
    model = pipeline.fit(train_data)
    predictions = model.transform(test_data)
    evaluator = BinaryClassificationEvaluator(labelCol="label")
    accuracy = evaluator.evaluate(predictions)
    print "Accuracy NB Model without Cross Validation: ", accuracy
