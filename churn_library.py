'''
Import all the necessary libraries
'''
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_roc_curve, classification_report

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

PATH=r"./data/bank_data.csv"
def import_data(pth):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    df = pd.read_csv(pth)
    df['Churn'] = df['Attrition_Flag'].apply\
    (lambda val: 0 if val == "Existing Customer" else 1)
    return df
def perform_eda(df):
    '''
    Save All the Images In the local Directory
    '''
    plt.figure(figsize=(20,10))
    df['Churn'].hist()
    plt.savefig('images/eda/Churn.png')
    plt.figure(figsize=(20,10))
    df['Customer_Age'].hist()
    plt.savefig('images/eda/Customer_Age.png')
    plt.figure(figsize=(20,10))
    df.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig('images/eda/Marital_Status.png')
    plt.figure(figsize=(20,10))
    sns.distplot(df['Total_Trans_Ct'])
    plt.savefig('images/eda/Total_Trans_Ct.png')
    plt.figure(figsize=(20,10))
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths = 2)
    plt.savefig('images/eda/heatmap.png')

cat_columns = ['Gender','Education_Level', 'Marital_Status', 'Income_Category','Card_Category']


def encoder_helper(df, category_lst):
    '''
    This is the function where I am converting the categorical data into a neumeric data
     And finaly return the X value.
    '''
    category_churn = []
#     y = df['Churn']
    X = pd.DataFrame()
    keep_cols = ['Customer_Age', 'Churn',
                 'Dependent_count', 'Months_on_book','Total_Relationship_Count',
                 'Months_Inactive_12_mon','Contacts_Count_12_mon', 'Credit_Limit',
                 'Total_Revolving_Bal','Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1',
                 'Total_Trans_Amt','Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1',
                 'Avg_Utilization_Ratio', 'Gender_Churn', 'Education_Level_Churn',
                 'Marital_Status_Churn',
                 'Income_Category_Churn', 'Card_Category_Churn']
    for category in category_lst:
        cat_lst=[]
        cat_group = df.groupby(category).mean()['Churn']
        for val in df[category]:
            cat_lst.append(cat_group.loc[val])
        churn_col =  category +"_Churn"
        df[churn_col] =  cat_lst
        category_churn.append(churn_col)
    X[keep_cols] = df[keep_cols]
    return X

data = import_data(PATH)
fun = encoder_helper(data, cat_columns)
fun.head()
data.head()
fun.columns

response=[]
def perform_feature_engineering(df, response):
    '''
    Split the df & return X_train_data, X_test_data, y_train_data, y_test_data
    '''
    y = df[response]
    X = df.drop(response, axis=1)
    X_train_data, X_test_data,y_train_data, y_test_data = train_test_split(X, y, test_size= 0.3,
                                                 random_state=42)
    return X_train_data, X_test_data, y_train_data, y_test_data

X_train, X_test, y_train, y_test = perform_feature_engineering(fun, 'Churn')

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# grid search
rfc = RandomForestClassifier(random_state=42)
lrc = LogisticRegression()

param_grid = {'n_estimators': [200, 500],
              'max_features': ['auto', 'sqrt'], 'max_depth' : [4,5,100],
              'criterion' :['gini', 'entropy']
              }
cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
cv_rfc.fit(X_train, y_train)

lrc.fit(X_train, y_train)

y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

y_train_preds_lr = lrc.predict(X_train)
y_test_preds_lr = lrc.predict(X_test)

def classification_report_image(y_train_data,
                                y_test_data,
                                y_train_preds_lr_data,
                                y_train_preds_rf_data,
                                y_test_preds_lr_data,
                                y_test_preds_rf_data):
    '''
    Print all the classifiers results & save the ROC curve, AUPR curve in the project directory
    '''
    print('random forest results')
    print('test results')
    print(classification_report(y_test, y_test_preds_rf))
    print('train results')
    print(classification_report(y_train, y_train_preds_rf))
    print('logistic regression results')
    print('test results')
    print(classification_report(y_test, y_test_preds_lr))
    print('train results')
    print(classification_report(y_train, y_train_preds_lr))
    lrc_plot = plot_roc_curve(lrc, X_test, y_test)
    # plt.savefig('images/results/LR.png')
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    rfc_disp = plot_roc_curve(cv_rfc.best_estimator_, X_test, y_test, ax=ax, alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    # plt.savefig('images/results/LR-RF.png')
    joblib.dump(cv_rfc.best_estimator_, 'models/rfc_model.pkl')
    joblib.dump(lrc, 'models/logistic_model.pkl')
    rfc_model = joblib.load('models/rfc_model.pkl')
    lr_model = joblib.load('models/logistic_model.pkl')
    lrc_plot = plot_roc_curve(lr_model, X_test, y_test)
    # plt.savefig('images/results/roc_LR.png')
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    rfc_disp = plot_roc_curve(rfc_model, X_test, y_test, ax=ax, alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    # plt.savefig('images/results/roc_LR-RF.png')

classification_report_image(y_train, y_test,
                            y_train_preds_lr, y_train_preds_rf,
                            y_test_preds_lr, y_test_preds_rf)
# Calculate feature importances
importances = cv_rfc.best_estimator_.feature_importances_
 # # Sort feature importances in descending order
indices = np.argsort(importances)[::-1]

func = encoder_helper(data, cat_columns)
X_data = func.drop('Churn', axis=1)

#     # Add bars
# plt.bar(range(X_data.shape[1]), model[indices])
# output_pth = plt.savefig('images/results/feature_importance.png')

model = 'models'
output_pth = 'images/results'
def feature_importance_plot(model, X_data, output_pth):
    '''
    Rearrange feature names so they match the sorted feature importances
    '''
    names = [X_data.columns[i] for i in indices]
    # Create plot
    plt.figure(figsize=(20,18))
    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')
    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])
    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    #save image
    # plt.savefig('images/results/feature_importance.png')

feature_importance_plot(model, X_data, output_pth)

def train_models(X_train_data, X_test_data, y_train_data, y_test_data):
    '''
    Train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Random Forest Train'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)),
             {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.6, str('Random Forest Test'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)),
             {'fontsize': 10}, fontproperties = 'monospace')
    plt.axis('off')
    print("\n")
    plt.rc('figure', figsize=(5, 5))
    plt.text(1.95, 1.25, str('Logistic Regression Train'),
    {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(1.95, 0.05, str(classification_report(y_train, y_train_preds_lr)),
    {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(1.95, 0.6, str('Logistic Regression Test'),
    {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(1.95, 0.7, str(classification_report(y_test, y_test_preds_lr)),
    {'fontsize': 10}, fontproperties = 'monospace')
    plt.axis('off')
train_models(X_train, X_test, y_train, y_test)
