import os
import sys
from os import environ, path

environ.update(
    {'SPARK_HOME': '/sparkdirectory'})
spark_home = environ.get('SPARK_HOME')
sys.path.insert(0, spark_home + "/python")
sys.path.insert(0, path.join(spark_home, 'python/lib/py4j-0.10.3-src.zip'))

from pyspark.ml.classification import RandomForestClassificationModel, LogisticRegressionModel, NaiveBayesModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator

pathtoproject = os.getcwd()
pathtodataset = pathtoproject + '/dataset/' + 'german_credit.csv'


def load_Random_Model(dataset):
    print ("Accuracy of best RFC Model with CrossValidation:")
    evaluator = BinaryClassificationEvaluator()
    best_RFModel = RandomForestClassificationModel.load("model/RFM1/")
    predictions = best_RFModel.transform(dataset)
    accuracy = evaluator.evaluate(predictions)
    print "The  accuracy = %g" % accuracy


def load_LogisticReg_Model(dataset):
    print ("Accuracy of best LRC Model with CrossValidation:")
    evaluator = BinaryClassificationEvaluator()
    best_LRModel = LogisticRegressionModel.load("model/LR1/")
    predictions = best_LRModel.transform(dataset)
    accuracy = evaluator.evaluate(predictions)
    print "The  accuracy = %g" % accuracy


def load_NaiveB_Model(dataset):
    print ("Accuracy of best NB Model with CrossValidation:")
    evaluator = BinaryClassificationEvaluator()
    best_NBModel = NaiveBayesModel.load("model/NB1/")
    predictions = best_NBModel.transform(dataset)
    accuracy = evaluator.evaluate(predictions)
    print "The  accuracy = %g" % accuracy
