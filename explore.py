import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt


from scipy import stats

# custom imports
import acquire
import prepare
import explore as e

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

seed =42

# acquire telco data using function from acquire module
telco_original = acquire.get_telco_data()

# acquire clean data using function from prepare module
telco = prepare.prep_telco(telco_original)

# split data using function from prepare module
train, validate, test = prepare.train_validate_test_split(telco, 'churn')

# create a list with encoded attributes
encoded = ['tenure', 'monthly_charges', 'total_charges','online_security_no',  'online_backup_no', 'tech_support_no', 'internet_service_type_fiber optic', 'contract_type_month-to-month', 'contract_type_two year','payment_type_electronic check']


# create labels 
X_train = train[encoded]
y_train = train.churn_yes

X_validate = validate[encoded]
y_validate = validate.churn_yes

X_test = test[encoded]
y_test = test.churn_yes


def vis_pie_churn(train):
    '''takes in a dataframe train and show pie chart of churn'''
    
    # set values and labels for chart
    values = [(train.churn == 'Yes').sum(), (train.churn == 'No').sum()] 
    labels = ['Churn','No churn', ] 

    # generate and show chart
    plt.pie(values, labels=labels, autopct='%.0f%%')
    plt.title('churn percentage of the train data')
    plt.show()


def vis_countplot(col, train):
    ''' takes in a column name and a dataframe and show countplot graph'''
    
    #plot countplot graph
    sns.countplot(x=col, hue='churn', data=train)
    plt.xticks(rotation=45)
    plt.show()

    
def vis_distplot(col, train):
    '''takes in a column name and a dataframe and show distplot graph'''
    
    # plot distplot graph
    sns.displot( x=col, hue='churn', data=train, multiple="dodge")
    plt.show()
    
    
def vis_countplot_cat(cols, train):
    '''takes in a list and a dataframe and show countplot graph'''
    
    # plot countplot graph
    for col in cols:
        if col != 'churn':
            sns.countplot(x=col, hue='churn', data=train)
            plt.xticks(rotation=45)
            plt.show()
            
            
def vis_distplot_num(cols, train):
    '''takes in a list and a dataframe and show distplot graph'''

    # plot distplot graph
    for col in cols:
        sns.displot(x=col, hue='churn', data=train, multiple="dodge")
        plt.show()

        
def cat_cols(train):
    '''takes in a dataframe and return a list'''
    
    #create a dataframe which contains columns with data type object and drop 'customer_id'
    df1 = train.select_dtypes(include='object').drop(columns='customer_id')
    
    #create a list from a dataframe
    cat_cols= df1.columns.tolist()
    
    # add a 'senior_citizen' to a list
    cat_cols.append('senior_citizen')
    
    return cat_cols


def num_cols(train):
    '''takes in a dataframe and return a list'''
    
    # create a list
    num_cols = ['tenure', 'monthly_charges', 'total_charges']
    
    return num_cols
    
                
def chi_test(cols, train):
    '''takes in a list and a dataframe and runs chi-square test to compare relationship of churn 
    with a datframe attributes 
    '''
    
    # set alpha value to 0.05
    alpha = 0.05
    
    for col in cols:
        if col != 'churn':
            
            # set null and alternative hypothesis 
            null_hypothesis = col + ' and churn are independent'
            alternative_hypothesis = col + ' and churn are dependent'
            
            # create an observed crosstab, or contingency table from a dataframe's two columns
            observed = pd.crosstab(train[col], train.churn)
            
            # run chi-square test
            chi2, p, degf, expected = stats.chi2_contingency(observed)
            
            # print column name
            print(f'{col}: ')
            
            # print Null Hypothesis followed by a new line
            print(f'Null Hypothesis: {null_hypothesis}\n')
            
            # print Alternative Hypothesis followed by a new line
            print(f'Alternative Hypothesis: {alternative_hypothesis}\n')
            
            # print the chi2 value
            print(f'chi^2 = {chi2}') 
            
            # print the p-value followed by a new line
            print(f'p     = {p}\n')
            
            if p < alpha:
                print(f'We reject null hypothesis')
                print(f'There exists some relationship between {col} and churn.')
            else:
                print(f'We fail to reject null hypothesis')
                print(f'There appears to be no significant relationship between {col} and churn.')
            print('--------------------------------------------------------------------------------------------\n')
            
            
def ind_t_test_greater(cols, train):
    
    '''takes in a list and a dataframe and runs independent t-test(1_tail,greater than) to compare mean between dataframe attributes'''
    
    # set alpha value to 0.05
    alpha = 0.05
    
    for col in cols:
       
        col_with_churn = train[train.churn == 'Yes'][col]
        col_without_churn = train[train.churn == 'No'][col]
        
         # set null and alternative hypothesis
        null_hypothesis = 'mean of ' + col +' of customers who churned is less or equal to mean of ' + col + ' of customers who haven\'t churned'
        alternative_hypothesis = 'mean of ' + col +' of customers who churn is greater than mean of ' + col + ' of customers who haven\'t churned'
        
        # print Null Hypothesis followed by a new line
        print(f'Null Hypothesis: {null_hypothesis}\n')
            
        # print Alternative Hypothesis followed by a new line
        print(f'Alternative Hypothesis: {alternative_hypothesis}\n')

        # verify assumptions:
            # - independent samples
            # - more than 30 observation
            # -equal Variances

        if col_with_churn.var() != col_without_churn.var(): 
            
            # run independent t-test without equl variances
            t, p = stats.ttest_ind(col_with_churn, col_without_churn, equal_var = False)
            
            # print t-statistic value
            print(f't: {t}')
            
            # print p-value followed by a new line
            print(f'p: {p}\n')

            if t > 0 and p/2 < alpha:
                print('we reject null hypothesis\n')
                print(alternative_hypothesis)
            else:
                print('We fail to reject null hypothesis\n')
                print(f'It appears that {null_hypothesis}')
        else: 
            # run independent t-test with equal variances
            t, p = stats.ttest_ind(col_with_churn, col_without_churn)
            
            # print t-statistic value
            print(f't: {t}')
            
            # print p-value followed by a new line
            print(f'p: {p}\n')

            if t > 0 and p/2 < alpha:
                print('we reject Null Hypothesis\n')
                print(alternative_hypothesis)
            else:
                print('We fail to reject Null Hypothesis\n')
                print(f'It appears that {null_hypothesis}')
        print('--------------------------------------------------------------------------------------------\n')

            
def baseline_accuracy():
    '''get baseline accuracy score'''
    
    # assign most common class to baseline
    baseline = y_train.mode()
    
    # compare baseline with y_train class to get most common class
    matches_baseline_prediction = (y_train == 0)
    
    # get mean
    baseline_accuracy = matches_baseline_prediction.mean()
    
    # print baseline accuracy
    print(f"Baseline accuracy: {(baseline_accuracy)}") 
    
    
def decision_tree():
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
    
    
def random_forest():
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



def knn():
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
    
        
def logistic_regression():
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
    

def logit_test(): 
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





