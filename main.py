# Import Libraries
import os
import NormalClassification
import testBestModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.types import *
from pyspark.shell import sqlContext
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.sql.functions import col

if __name__ == "__main__":

    pathtoproject = os.getcwd()
    pathtodataset = pathtoproject + '/dataset/' + 'german_credit.csv'
    # The input CSV file is loaded as Spark DataFrame format
    # Load it through RDD, construct an RDD, define the schema and then convert it into a data frame
    # Changing the header as the default headers are lengthy
    schema = StructType([

        StructField('creditability', DoubleType(), True),

        StructField('balance', DoubleType(), True),

        StructField('duration', DoubleType(), True),

        StructField('history', DoubleType(), True),

        StructField('purpose', DoubleType(), True),

        StructField('amount', DoubleType(), True),

        StructField('savings', DoubleType(), True),

        StructField('employment', DoubleType(), True),

        StructField('instPercent', DoubleType(), True),

        StructField('sexMarried', DoubleType(), True),

        StructField('guarantors', DoubleType(), True),

        StructField('residenceDuration', DoubleType(), True),

        StructField('assets', DoubleType(), True),

        StructField('age', DoubleType(), True),

        StructField('concCredit', DoubleType(), True),

        StructField('apartment', DoubleType(), True),

        StructField('credits', DoubleType(), True),

        StructField('occupation', DoubleType(), True),

        StructField('dependents', DoubleType(), True),

        StructField('hasPhone', DoubleType(), True),

        StructField('foreign', DoubleType(), True)])
    # We use the sqlContext.read method to read the data and set a few options:
    #  'format': specifies the Spark CSV data source
    #  'header': set to true to indicate that the first line of the CSV data file is a header
    # The file is called 'german_credit.csv'.
    dataF = sqlContext.read.format("com.databricks.spark.csv") \
        .option("header", "true") \
        .option("inferschema", "true") \
        .option("mode", "DROPMALFORMED") \
        .load(pathtodataset, schema=schema)
    # Calling cache on the DataFrame will make sure we persist it in memory the first time it is used.
    # The following uses will be able to read from memory, instead of re-reading the data from disk.
    dataF.cache()
    # The DataFrame is currently using strings, but we know all columns are numeric. Let's cast them.
    # The following call takes all columns (df.columns) and casts them using Spark SQL
    # to a numeric type (DoubleType).
    df = dataF.select([col(c).cast("double").alias(c) for c in dataF.columns])

    featuresCols = df.columns
    featuresCols.remove('creditability')
    # Spark dataframes are not used like that in Spark ML;
    # This concatenates all feature columns into a single feature vector in a new column "features"
    assembler = VectorAssembler(inputCols=featuresCols, outputCol="features")
    df1 = assembler.transform(df)
    # This identifies categorical features and indexes them.
    labelIndexer = StringIndexer(inputCol="creditability", outputCol="label")
    df2 = labelIndexer.fit(df1).transform(df1)

train_data, test_data = df2.randomSplit([0.9, 0.1])
evaluator = BinaryClassificationEvaluator()

# Random Forest classifier
print NormalClassification.random_Forest(train_data, test_data)
# test the best random forest model with cross validation
print testBestModel.load_Random_Model(test_data)

# Logistic Regression classifier
print NormalClassification.logistic_reg(train_data, test_data)
# test the best logistic regression model with cross validation
print testBestModel.load_LogisticReg_Model(test_data)

# Naive Bayes classifier
print NormalClassification.naive_Bayes(train_data, test_data)
# test the best Naive Bayes model with cross validation
print testBestModel.load_NaiveB_Model(test_data)
