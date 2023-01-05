# Telco Churn

# Project Description

Telco is a telecommunication company that is seeing 27%  of customer churn. The goal of the project is to analyze the data, find driving features of customer churn, predict cusotmer churn, and recommend actions to reduce customer churn.  

# Project Goals
* Discover driver of customer churn at Telco
* Use drivers to develop a model to predict customer churn
* Offer recommendation to reduce customer churn

# Initial Questions
* Find percentage of churn
* Find important categorical variables driving customer churn
* Do customers who churn have a higher average monthly charge than customers who do not churn?
* Does contract type play a role in higher customer churn?
* Do customers with tech support likely to churn less?

# The Plan

* Acquire data
    * Acquire data from Telco_churn database from Codeup database using SQl queryy using function from acquire.py file

* Prepare data
    * Use functions from prepare.py to clean data. 
    * Drop unnecessary columns. 
    * Replace white space and convert data types.
    * Encode attributes to fit in ML format.
    * split data into train, validate and test (approximatley 56/24/20)

* Explore Data
    * Use graph and hypothesis testing to find churn, driving factors of churn, and answer the following initial questions
        * Do customers who churn have a higher average monthly charge than customers who do not churn?
        * Does contract type play a role in higher customer churn?
        * Do customers with tech support likely to churn less?

* Develop Model
    * Use driving attributes to create labels
    * Set up baseline prediction
    * Evaluate models on train data and validate data
    * Select the best model based on the highest accuracy 
    * Evaluate the best model on test data to make predictions

* Draw Conclusions

# Data Dictionary
| Feature | Definition |
|:--------|:-----------|
Payment_type_id|customer’s payment type id| 1, 2, 3, 4|
| Contract_type_id| customer’s contract type id| 1, 2, 3|
| Internet_service_type_id| customer’s internet service type id| 1, 2, ,3, 4|
|  Customer_id| customer’s id|
|  Gender| customer’s gender| male, female|
|  Senior_citizen| is the customer a senior citizen? 0, 1|
|  Partner| Does the customer have a partner? yes, no|
|  Dependents| Does the customer have dependents? yes, no|
|  Tenure| Duration in month customer have been with the company|
|  Phone_sevice| Does the customer have phone service? yes, no|
|  Multiple_lines | Does the customer have multiple phone lines? yes, no, no phone service|
|  Online_security| Does the customer have online security? yes, no, no internet service|
|  Online_backup| Does the customer have online backup? yes, no, no internet service|
|  Device_protection| Does the customer have device protection? yes, no, no internet service|
|  Tech_support| Does the customer have tech support?yes, no, no internet service|
|  Streaming _tv| Does the customer have streaming tv| yes, no, no internet service|
|  Streaming _movies| Does the customer have streaming movies?| yes, no, no internet service|
|  Paperless_billing| Does the customer use paperless billing? yes, no|
|  Monthly_charges| the customer’s monthly charges|
|  Total_charges| the customer’s total charges|
|  Churn| Did the customer churn? yes, no|
|  Internet_service_type| Type of internet service| fiber, DSL, none|
|  Contract_type| Type of contract| two year, one year, month-to-month|
|  Payement_type| Type of payment| electronic, mailed check, bank transfer (automatic), credit card (automatic)|

# Steps to Reproduce
1. Clone this repo 
2. To acquire data, need to have access to to MYSQL database of codeup. 
3. Data can be also be acquired from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), save a file as 'telco.csv' and put the file into the cloned repo 
5. Run notebook

# Takeaways and Conclusions
* Customer churn is about 27%.
* Gender and Phone service are not driving customer churn.
* online security, online backup, device protection, tech support, internet service type, tenure, monthly charges, and total charges are main drivers of customer churn.
* Customers with high monthly charges churn in higher ratio.
* Contract type is one of main driver of customer churn. Customers with contract type of month-to-month churn in higher ratio than customers with other contract type
* Customers with add on : online security, online backup, device protection, and tech support churn less than customers who do do not have those add on.

# Recommendations
* Run programs or offers that increase customer to sign one year or two year contract. Customers with contract of monht-to-month churn in higher ration than customers with contract of one year or two year.
* Maintian constant monthly charges or lower monthly charges to retain customer who churn when monthly charge increases.
* Customer with additional package like online security, online backup, device protection, and tech support tends to less churn so run programs that increase customer to add those packages.







