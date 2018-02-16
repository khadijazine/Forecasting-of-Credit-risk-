import sys
from os import environ, path

from sklearn import svm

from pyspark.conf import SparkConf
from pyspark.context import SparkContext
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# find spark directory
def init_spark_context():
    environ.update(
        {'SPARK_HOME': '/sparkdirectory'})
    spark_home = environ.get('SPARK_HOME')
    sys.path.insert(0, spark_home + "/python")
    sys.path.insert(0, path.join(spark_home, 'python/lib/py4j-0.10.3-src.zip'))
    # load spark context
    conf = SparkConf().setAppName("SparkProject")
    sc = SparkContext(conf=conf, pyFiles=['main.py', 'modules.py'])

    return sc


# Split data using k_folds cross_validation
# This snippet shows how to shuffle the data when using cross_val_score() function
def k_fold_cross_validation(Set_instances, K, shuffle=True):

    if shuffle: from random import shuffle; Set_instances = list(Set_instances); shuffle(Set_instances)
    for k in xrange(K):
        training = [x for i, x in enumerate(Set_instances) if i % K != k]
        validation = [x for i, x in enumerate(Set_instances) if i % K == k]
        yield training, validation


# Classifiers
# SVM
def s_v_m(Input, target):
    svm_model = svm.SVC()
    print("Support vector machine Classifier: ")
    print(cross_val_score(svm_model, Input, target, scoring='accuracy', cv=10))
    accuracy = cross_val_score(svm_model, Input, target, scoring='accuracy', cv=10).mean() * 100
    print("Accuracy of SVMC is: ", accuracy)


# Random Forest
def random_forest(Input, target):
    random_forest_model = RandomForestClassifier(n_estimators=10)
    print("Random Forest Classifier: ")
    print(cross_val_score(random_forest_model, Input, target, scoring='accuracy', cv=10))
    accuracy = cross_val_score(random_forest_model, Input, target, scoring='accuracy', cv=10).mean() * 100
    print("Accuracy of RFC is: ", accuracy)


# Logistic Regression
def logistic_regression(Input, target):
    logistic_Regression_model = LogisticRegression()
    print("Logistic Regression Classifier: ")
    print(cross_val_score(logistic_Regression_model, Input, target, scoring='accuracy', cv=10))
    accuracy = cross_val_score(logistic_Regression_model, Input, target, scoring='accuracy', cv=10).mean() * 100
    print("Accuracy of LRC is: ", accuracy)
