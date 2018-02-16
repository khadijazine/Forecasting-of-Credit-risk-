                            Analyzing Credit Risk with Spark Machine Learning

Objective: Minimization of risk and maximization of profit on behalf of the bank

The bank needs a decision rule regarding whether a person will pay back a loan or not,
who to give approval of the loan and who not to. An applicant’s profile is considered by loan managers before a decision is taken regarding a loan application.
If the applicant is a good credit risk, is likely to repay the loan, then not approving the loan to the person results in a loss of business to the bank. 
If the applicant is a bad credit risk, is not likely to repay the loan, then approving the loan to the person results in a financial loss to the bank. 

Dataset description :

Our data is from The German Credit Data Set, it contains data on 21 attributes and the classification is about whether an applicant is considered creditable or not creditable  for 1000 loan applicants. 
A predictive model developed on this data is expected to provide a bank manager guidance for making a decision.
For each bank loan application we have the following informations:
Creditability : label takes the value 1 if the applicant is a good credit risk and 0 if is bad,
Balance : Account balance, status of existing checking account 
Duration : duration of credit in months.
History : history of previous credits.
Purpose : purpose of credit
Credit amount
Saving : savings account/bonds
Employment : Present employment since 
InstPercent : installment rate in percentage of disposable income
Personal status and sex : single, married, divorced/male, female
Guarantors : other debtors / guarantors
Residence durationPresent residence since….

We worked with the numerical attributes of the dataset, this is the format of the CSV file :

1,1,18,4,2,1049,1,2,4,2,1,4,2,21,3,1,1,3,1,1,1

1,1,9,4,0,2799,1,3,2,3,1,2,1,36,3,1,2,3,2,1,1

Software :

Why predicting loan credit risk with Apache Spark Machine Learning ?
Banks wanted to automate the loan eligibility process based on customer provided in their online application form. As the number of transactions in banking sector is rapidly growing and huge data volumes are available, the customers’ behavior can be easily analyzed and the risks around loan can be reduced. So, it is very important to predict the loan type and loan amount based on the banks’ data using a robust framework as Apache Spark.
In this project, we discuss about how Random Forest, Logistic Regression and Naive Bayes Classification models using Python can be used to predict the loans.

Input variables :

All the attributes that contribute to the classification.

Used techniques :

The global goal of this work is the use of cross validation process to optimize the classification and these are the main steps :

	-Load data from the CSV file into spark’s dataframe.

	-To build a classifier model, we extract the features that most contribute to the classification.

	-We train the model using a pipeline, which gives better results, it provides a simple way to try out all the combinations of parameters and tries each one making 10 folds cross validated model for each combination.

	-Grid search method returns the best set of predictions from the best model that it tried.

	-To train a machine learning Pipeline, we used :
		VectorAssembler : assemble the feature columns into a feature vector with known label .
		VectorIndexer : identify column which should be treated as categorical label.
		Classifiers : Random Forest, Logistic Regression and Naive Bayes.
		CrossValidator : each classifier has several hyperparameters and tuning them to our data to improve accuracy.
		
it tests a grid of parameters, select the best model and save it to "model/" directory.

Project package:

The package contains main.py, NormalClassification.py, cross_validation.py and testBestModel.py files.
NormalClassification.py: in which we implemented 3 functions, each one apply a classifier, train and get the accuracy of a model without using cross validation.
cross_validation.py: in which we classify the data using cross validation process for Random Forest, Logistic Regression and Naive Bayes models.
testBestModel.py: after we saved the best trained model, we tested it on the test_data.
main.py: this is the main function where we call the NormalClassification and testBestModel functions then get the accuracy before and after the cross validation, obviously we can remark that cross validation improved the accuracy.
N.B: When you want to retrain a new CrossValidator model, you have to delete the trained models from the /model/ directory. 
To run the project, open a shell into the project directory, type the following command: python main.py and get the output results bellow.

Output results :

Accuracy of Random Forest Classifier without cross validation:
The  accuracy = 0.788852
Accuracy of best RFC Model with CrossValidation:
The  accuracy = 0.994568

Accuracy of Logistic Regression Classifier without cross validation:
The  accuracy = 0.771847
Accuracy of best LRC Model with CrossValidation:
The  accuracy = 0.794993

Accuracy of Naive Bayes Model without Cross Validation:
The  accuracy = 0.388757675957
Accuracy of best NB Model with Cross Validation:
The  accuracy = 0.388285

The accuracy allows us to understand how these models are behaving for the test dataset and determine the best model.
From these results, we conclude that the best model that is very suitable with our dataset is Random Forest, Logistic Regression is slightly better and Naive Bayes is not suitable for our dataset, which might explain the low accuracy that we are getting.
