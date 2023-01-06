# imports
import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# set seed
seed =42

def prep_model(train,validate,test):
    '''Prepare train, validate, and test data for modeling'''
    
    # create a list with encoded attributes
    encoded = ['tenure', 'monthly_charges', 'total_charges','online_security_no',  'online_backup_no', 'tech_support_no', 'internet_service_type_fiber optic', 'contract_type_month-to-month', 'contract_type_two year','payment_type_electronic check']


    # create labels 
    X_train = train[encoded]
    y_train = train.churn_yes

    X_validate = validate[encoded]
    y_validate = validate.churn_yes

    X_test = test[encoded]
    y_test = test.churn_yes

    return X_train, X_validate, X_test, y_train, y_validate, y_test


def get_baseline_accuracy(X_train, y_train):
    '''get baseline accuracy score'''
    
    # assign most common class to baseline
    baseline = y_train.mode()
    
    # compare baseline with y_train class to get most common class
    matches_baseline_prediction = (y_train == 0)
    
    # get mean
    baseline_accuracy = matches_baseline_prediction.mean()
    
    # print baseline accuracy
    print(f"Baseline accuracy: {(baseline_accuracy)}") 
    
    
def get_decision_tree(X_train, X_validate, y_train, y_validate):
    '''get decision tree accuracy score on train and validate data'''
    
    # create model
    clf = DecisionTreeClassifier(max_depth = 6, random_state=42)

    # fit the model to train data
    clf.fit(X_train, y_train)

    # compute accuracy
    train_acc = clf.score(X_train, y_train)
    validate_acc = clf.score(X_validate, y_validate)
    
    # print accuracy score on train
    print(f'Decision Tree Accuracy score on train set: {train_acc}')
    
    # print accuracy score on validate
    print(f'Decsion Tee Accuracy score on validate set: {validate_acc}')
    
    
def get_random_forest(X_train, X_validate, y_train, y_validate):
    '''get random forest accuracy score on train and validate data'''
    
    # create model
    rf= RandomForestClassifier(min_samples_leaf = 10, random_state=42) 

    # fit the model to train data
    rf.fit(X_train, y_train)

    # compute accuracy
    train_acc = rf.score(X_train, y_train)
    validate_acc = rf.score(X_validate, y_validate)
    
    # print accuracy score on train
    print(f'Random Forest Accuracy score on train set: {train_acc}')
    
    # print accuracy score on validate
    print(f'Random Forest score on validate set: {validate_acc}')



def get_knn(X_train, X_validate, y_train, y_validate):
    ''' get KNN accuracy score on train and validate data'''
    
    # create model
    knn= KNeighborsClassifier(n_neighbors = 10) 

    # fit the model to train data
    knn.fit(X_train, y_train)

    # compute accuracy
    train_acc = knn.score(X_train, y_train)
    validate_acc = knn.score(X_validate, y_validate)
    
    # print accuracy score on train
    print(f'KNN Accuracy score on train set: {train_acc}')
    
    # print accuracy score on validate
    print(f'KNN Accuracy score on validate set: {validate_acc}')
    
        
def get_logistic_regression(X_train, X_validate, y_train, y_validate):
    '''get logistic regression accuracy score on train and validate data'''
    
    # create model
    logit = LogisticRegression(C = 1, random_state=seed, solver='liblinear')

    # fit the model to train data
    logit.fit(X_train, y_train)

    # compute accuracy
    train_acc = logit.score(X_train, y_train)
    validate_acc = logit.score(X_validate, y_validate)
    
    # print accuracy score on train
    print(f'Logistic Regression Accuracy score on train set: {train_acc}')
    
    # print accuracy score on validate
    print(f'Logistic Regression Accuracy score on validate set: {validate_acc}')
    

def get_logistic_regression_test(X_train, X_validate, y_train, y_validate, X_test, y_test, test): 
    '''get logistic regression accuracy on train data,validate data, test data
    return a dataframe with predictions
    '''

    # create model
    logit = LogisticRegression(C = 1, random_state=seed, solver='liblinear')

    # fit the model to train data
    logit.fit(X_train, y_train)
    
    # evaluate the model's performance on train
    y_train_pred =logit.predict(X_train)

    # evaluate the model's performance on validate
    y_validate_pred = logit.predict(X_validate)

    # use Logistic Regression model to make predictions 
    y_test_pred = logit.predict(X_test)
   
    # compute accuracy
    train_acc = logit.score(X_train, y_train)
    validate_acc = logit.score(X_validate, y_validate)
    test_acc = logit.score(X_test, y_test)
    
    # print accuracy score on test
    print(f'Logistic Regression Accuracy score on test set: {test_acc}')
    
    # estimate probability
    y_pred_proba = logit.predict_proba(X_test)

    # select the 2nd item in the array to get prob of survival (1)
    y_pred_proba = np.array([i[1] for i in y_pred_proba])

    # create a dataframe
    predictions = pd.DataFrame({'customer_id': test.customer_id,'probability': y_pred_proba,'prediction': y_test_pred})
    
    # return dataframe
    return predictions

