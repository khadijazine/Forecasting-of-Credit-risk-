# Import Libraries
import os
import pandas as pd
import modules

if __name__ == "__main__":

    pathtoproject = os.getcwd()
    pathtodataset = pathtoproject + '/dataset/' + 'german_credit.csv'
    # Extract the header as a list from the rest of the data file
    sc = modules.init_spark_context()
    df = pd.read_csv(pathtodataset)
    feature_cols = ['Account Balance', 'Duration of Credit (month)',
                    'Payment Status of Previous Credit', 'Purpose', 'Credit Amount',
                    'Value Savings/Stocks', 'Length of current employment', 'Instalment per cent',
                    'Sex & Marital Status', 'Guarantors', 'Duration in Current address',
                    'Most valuable available asset', 'Age (years)', 'Concurrent Credits',
                    'Type of apartment', 'No of Credits at this Bank', 'Occupation',
                    'No of dependents', 'Telephone', 'Foreign Worker']
    target_data = df['Creditability']
    input_data = df[feature_cols]
    # Split dataset into training and validation(test) dataset using 10-folds cross validation
    input_dataFolds = [i for i in xrange(100)]
    print("Training       Validation")
    for training, validation in modules.k_fold_cross_validation(input_dataFolds, K=10, shuffle=True):
        for x in input_dataFolds: assert (x in training) ^ (x in validation), x
        print(training, validation)

# Create SVM, Random Forest and Logistic Regression classifiers
print modules.s_v_m(input_data, target_data)
print modules.random_forest(input_data, target_data)
print modules.logistic_regression(input_data, target_data)
