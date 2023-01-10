#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 11:42:49 2022

@author: sijingyu
"""
#********Project: credit card approval*******
#******* This .py is only for the model training and anaysls



#import
#*************************************************
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import (roc_auc_score,plot_roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.inspection import partial_dependence
from sklearn.inspection import  plot_partial_dependence
from mlxtend.evaluate import feature_importance_permutation
from sklearn.model_selection import KFold



####################################################
#############Part 2: Model part.

### over sampling function###
from imblearn.over_sampling import RandomOverSampler
def over_sampling_fun(X_train,y_train):
    ros = RandomOverSampler(sampling_strategy="auto",random_state=29)
    X_res,y_res=ros.fit_resample(X_train,y_train)
    return X_res,y_res

### SMOTE NC sampling function ###
from imblearn.over_sampling import SMOTENC
def smotenc_fun(X_train,y_train,ind,k):
    smnc=SMOTENC(sampling_strategy="auto",random_state=20,k_neighbors=k,n_jobs=-1,categorical_features=ind)
    # ind: indeces of the columns of categorical variables
    x_smnc,y_smnc=smnc.fit_resample(X_train, y_train)
    return x_smnc,y_smnc


#### Normalize function ####
def scaler_fun(X,test=0):
    ##Normalize the feature varibale(not include the categorical variable)
    ###MinmaxScaler
    scaler=MinMaxScaler()
    scaler.fit(X)
    X=scaler.transform(X)
    return X

####(1) transform some categorical varibale as dummy varibale 
#   (2) normilize the numerical variable
def transform_x_train(df,y_exist=1):
    ###normilize the numerical variable in X and do one hot encoding for categorical varibale in X, and concat them
    ##There are only 2 categorical variables:ind_XYZ and rep_education
    if y_exist==1:
        X_train_numerical=df.drop(['rep_education','Def_ind'],axis=1) # all numerical variable and ind_XYZ(ind_XYZ will keep the same before and after the nomilization)
    else:
        X_train_numerical=df.drop(['rep_education'],axis=1) # all numerical variable and ind_XYZ(ind_XYZ will keep the same before and after the nomilization)
    name=X_train_numerical.columns
    X_train_numerical = pd.DataFrame(scaler_fun(X_train_numerical))# Normalize the numerical variable
    X_train_numerical.columns=name
    dummies=pd.DataFrame(pd.get_dummies(df['rep_education'],drop_first=True))#create dummy variable for rep_education
    dummies.index=X_train_numerical.index
    X_train=pd.concat([X_train_numerical,dummies],axis=1)
    return X_train


#####Only need transform some categorical varibale as dummy varibale 
def transform_x_train_RandomForest(df,y_exist=1):
    ###normilize the numerical variable in X and do one hot encoding for categorical varibale in X, and concat them
    ##There are only 2 categorical variables:ind_XYZ and rep_education
    if y_exist==1:
        X_train_numerical=df.drop(['rep_education','Def_ind'],axis=1) # all numerical variable and ind_XYZ(ind_XYZ will keep the same before and after the nomilization)
    else:
        X_train_numerical=df.drop(['rep_education'],axis=1) # all numerical variable and ind_XYZ(ind_XYZ will keep the same before and after the nomilization)
    dummies=pd.DataFrame(pd.get_dummies(df['rep_education'],drop_first=True))#create dummy variable for rep_education
    dummies.index=X_train_numerical.index
    X_train=pd.concat([X_train_numerical,dummies],axis=1)
    return X_train
#
###deleting some data
def data_pre_processing(train,test):
    ### Training data: Drop some variable and some samples
    train['ratio open credit']=ratio_open_credit(train.num_card_12_month,train.num_inq_12_month)
    train=train.drop('pct_card_over_50_uti',axis=1)
    train=train.drop('credit_card_age',axis=1)
    df=train.dropna(axis=0,how="any")
    y_train=pd.DataFrame(df.Def_ind)
    ### Test data: drop some variable and impute missing data 
    test=test.drop('credit_card_age',axis=1)
    test=test.drop('pct_card_over_50_uti',axis=1)
    test['ratio open credit']=ratio_open_credit(test.num_card_12_month,test.num_inq_12_month)
    test.info()
    test["rep_income"]=test["rep_income"].fillna(df.rep_income.median())
    test["rep_education"]=test["rep_education"].fillna(df.rep_education.mode()[0])
    y_test=test['Def_ind']
    return df,y_train,test,y_test

# find best patameters of Logistic Regression
def Logistic_search_param(X_train,y_train):
    X_train=transform_x_train(df)
    param_grid={#'C':[0.001,0.005,0.01,0.05,0.1,1,5,10,50,100,500,1000],
                'class_weight':["balanced",{0:1,1:10},{0:1,1:25},{0:1,1:50},{0:1,1:75},{0:1,1:100}]
                }
    search= GridSearchCV(estimator=LogisticRegression(penalty='l1',solver='saga'),#class_weight="balanced"),
                         scoring='roc_auc',
                         param_grid=param_grid,
                         cv=10).fit(X_train,y_train)
    print(search.best_params_)
    print(search.best_estimator_)
    return search.best_estimator_

## Train the best Logistic regression model
def Logistic_train_with_best_param():
    logmodel =LogisticRegression(penalty='l1',solver='saga',C=0.1,class_weight="balanced")
    clf=logmodel.fit(X_train,y_train)
    #coef=pd.DataFrame(list(clf.coef_,)[0])
    #print(pd.concat([pd.DataFrame(X_train.columns),coef],axis=1))
    return logmodel

## evaluate the model
def evla(model,X_train,y_train,X_test,y_test):
    predictions=logmodel.predict(X_test)
    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test, predictions))
    print(roc_auc_score(y_test, predictions))
          
    train_pre=logmodel.predict(X_train)
    print(classification_report(y_train, train_pre))
    print(confusion_matrix(y_train, train_pre))
    print(roc_auc_score(y_train, train_pre))
    return 

# find best patameters of Random Forest
def Random_Forest_turn_para(X_train,y_train):
    rf=RandomForestClassifier(n_estimators=100,random_state=39,max_depth=2,n_jobs=-1,max_features='sqrt')
    param_grid={
        #'n_estimators':[10,50,100,200],
        #'max_depth':[10,20,30,50,75,100],
        'min_samples_leaf':[20,30,50]
        }
    search= RandomizedSearchCV(estimator=RandomForestClassifier(class_weight="balanced"),#,max_features='sqrt',)
                         scoring='roc_auc',
                         param_distributions=param_grid,
                         cv=10,n_iter=10).fit(X_train,y_train)
    print(search.best_params_)
    print(search.best_estimator_)
    print(search.score(X_test,y_test))
    return search.best_params_

# Train the best model of RF
def Random_Forest_with_best_param():
    rf=RandomForestClassifier(n_estimators=200,max_depth=10,min_samples_leaf=20,random_state=39,n_jobs=-1,class_weight="balanced")#,max_features='sqrt')#
    rf.fit(X_train,y_train)
    return rf

# plot feature_mportance
def feature_mportance(X_train,rf):
    feat_imp_df=pd.DataFrame({'features':X_train.columns,'importance':rf.feature_importances_})
    ind=feat_imp_df.index
    df_new=feat_imp_df.sort_values("importance",ascending=False)
    plt.figure()
    plt.title("Random Forest impurity-based feature importance")
    feture_import_output=pd.concat([pd.DataFrame(df_new.features),pd.DataFrame(df_new.importance)],axis=1)
    feture_import_output.index=ind
    return feture_import_output

# plot partial_depend_plot
def partial_depend_plot(X_train,X_test,rf):
    features=list(X_test.columns)
    features=['ind_XYZ']
    plot_partial_dependence(rf,X_train,features,n_jobs=-1,grid_resolution=20)
    fig=plt.gcf()
    fig.set_size_inches(14,5)
    fig.suptitle("Partial dependence of default risk on ind_XYZ")
    return

#  drop some variables depending on original RF result
def drop_X_by_primary_RF(X_train,X_test):
    X_train=X_train.drop('tot_balance',axis=1)
    X_train=X_train.drop('credit_age_good_account',axis=1)
    X_train=X_train.drop(['num_acc_30d_past_due_6_months','num_acc_30d_past_due_12_months','num_mortgage_currently_past_due'],axis=1)
    X_train=X_train.drop('num_card_inq_24_month',axis=1)
    X_train=X_train.drop(['uti_max_credit_line','pct_over_50_uti'],axis=1)

    X_test=X_test.drop('tot_balance',axis=1)
    X_test=X_test.drop('credit_age_good_account',axis=1)
    X_test=X_test.drop(['num_acc_30d_past_due_6_months','num_acc_30d_past_due_12_months','num_mortgage_currently_past_due'],axis=1)
    X_test=X_test.drop('num_card_inq_24_month',axis=1)
    X_test=X_test.drop(['uti_max_credit_line','pct_over_50_uti'],axis=1)
    return X_train,X_test

# feature_permutation
def feature_permutation(X_train,y_train,rf):
    X_train, x_validation, y_train, y_validation=train_test_split(X_train,y_train,test_size=0.2)
    
    imp_vals, imp_all = feature_importance_permutation(
        predict_method=rf.predict,
        X=x_validation.values,
        y=y_validation,
        metric='accuracy',
        num_rounds=50,
        seed=0
        )
    std=pd.DataFrame(np.std(imp_all,axis=1))
    std.columns=["std"]
    feat_imp_df=pd.concat([pd.DataFrame(X_train.columns),pd.DataFrame(imp_vals),std],axis=1)
    feat_imp_df.columns=['features','importance','std']
    #feat_imp_df=pd.DataFrame({'features':pd.DataFrame(X_train.columns),'importance':pd.DataFrame(imp_vals),'std':std})
    df_new=feat_imp_df.sort_values("importance",ascending=False)[:6]

    plt.figure()
    plt.title("Random Forest permutaion feature importance top 6")
    plt.bar(df_new.features,df_new.importance.values,yerr=df_new['std'])
    plt.bar
    return 

# trainsform X for SMOTNC
def transform_x_train_SMOTENC(df):
    ###normilize the numerical variable in X,keep the categorical variables as their original form
    ###There are only 2 categorical variables:ind_XYZ and rep_education
    X_train_numerical=df.drop(['rep_education','Def_ind','ind_XYZ'],axis=1) # all numerical variable
    name=X_train_numerical.columns
    X_train_numerical = pd.DataFrame(scaler_fun(X_train_numerical))# Normalize the numerical variable
    X_train_numerical.columns=name
    df_categorical=df[['ind_XYZ','rep_education']] # keep the categorical variables as their original form
    df_categorical.index=X_train_numerical.index
    X_train=pd.concat([X_train_numerical,df_categorical],axis=1) # Conbine the data
    return X_train

# RandomForst_SMOTNC
def RandomForest_SMOTNE(df,y_train):
    ####### Banlance data with SMOTENC and do RandomForst#####
    ########1. set some candidate paratemers
    ########2. do 10-flod Crossvalidation for the Training data set
    ########3. for each step of crossvalidation, we will get one training data and one validation date
    ########4. use SMOTENC for each steps training data, do not use it for validation data
    ########5. compute K-fold auc and average them 
    ########6. choose the best params, use this params to fit whole of the original training data set
    Depth =[10]#,25,50,75,100]
    N_estimators=[50]#,100,200]
    Min_samples_leaf=[20],#30,50]

    X_train=transform_x_train_SMOTENC(df)
    mean_acu=[]
    D_record=[]
    N_record=[]
    Min_leaf_record=[]
    for D in Depth:
        for N in N_estimators:
            for n_leaf in Min_samples_leaf:
                kf = KFold(n_splits=10)
                auc=0
                for train_index, test_index in kf.split(X_train):
                    ## Do 10 fold crossvalidation
                    X_train_now_fold, X_validation = X_train.iloc[train_index,:], X_train.iloc[test_index,:]
                    y_train_now_fold, y_validation = y_train.iloc[train_index], y_train.iloc[test_index]
                    # For each step, we will do SMOTENC for the current trainging dataset, not do it for validation dataset
                    [X_train_now_fold,y_train_now_fold]=smotenc_fun(X_train_now_fold,y_train_now_fold,[17,18],5)   
                    # transform the dataset as the imput of RandomForest
                    X_train_now_fold=transform_x_train_RandomForest(X_train_now_fold,y_exist=0)
                    # fit model and predict
                    rf=RandomForestClassifier(n_estimators=N,max_depth=D,min_samples_leaf=n_leaf,random_state=39,n_jobs=-1)#,max_features='sqrt')
                    rf.fit(X_train_now_fold,y_train_now_fold)
                    X_validation=transform_x_train(X_validation,y_exist=0)
                    # caculate auc
                    predictions = rf.predict(X_validation)
                    auc+=roc_auc_score(y_validation, predictions)
                mean_acu.append(auc/10)
                D_record.append(D)
                N_record.append(N)
                Min_leaf_record.append(n_leaf)
    # select best params
    param_sel=pd.concat([pd.DataFrame(mean_acu),pd.DataFrame(D_record),pd.DataFrame(N_record),pd.DataFrame(Min_leaf_record)],axis=1)
    param_sel.columns=['mean_acu','depth','n_estimator','min_samples_leaf']
    param_sel=param_sel[param_sel.mean_acu==max(param_sel.mean_acu)]
    print(param_sel)

    # retrain the model for the original whole dataset with best params
    [X_train_now,y_train_now]=smotenc_fun(X_train,y_train,[17,18],5) 
    X_train_now=transform_x_train_RandomForest(X_train_now,y_exist=0)
    rf=RandomForestClassifier(n_estimators=100,max_depth=25,min_samples_leaf=20,random_state=39,n_jobs=-1)#,max_features='sqrt')#,class_weight="balanced")
    rf.fit(X_train_now,y_train_now)
    return rt
    
 
###load data   
df = pd.read_csv("Training_R-197135_Candidate Attach #1_JDSE_SRF #456.csv.csv")
test=pd.read_csv("Test_R-197135_Candidate Attach #2_JDSE_SRF #456.csv")
df_before_balance=df.copy()

###data pre_processing
df,y_train,test,y_test=data_pre_processing(df,test)


##Logistical regession
X_train=transform_x_train(df)
X_test=transform_x_train(test)
best_parme=Logistic_search_param(X_train,y_train)
logmodel=Logistic_train_with_best_param()
evla(logmodel,X_train,y_train,X_test,y_test)

##Random Forest
X_train=transform_x_train_RandomForest(df)
X_test=transform_x_train_RandomForest(test)
best_parme=Random_Forest_turn_para(X_train,y_train)
rf=Random_Forest_with_best_param()
evla(rf,X_train,y_train,X_test,y_test)
feture_import_output=feature_mportance(X_train,rf)
partial_depend_plot(X_train,X_test,rf)

#### Random Forest with SMOTENC
#rt_SMOTE_NC=RandomForest_SMOTNE(df,y_train)


####Adjust Random Forest
X_train=transform_x_train_RandomForest(df)
X_test=transform_x_train_RandomForest(test)
X_train,X_test=drop_X_by_primary_RF(X_train,X_test)
best_parme=Random_Forest_turn_para(X_train,y_train)
rf=Random_Forest_with_best_param()
feture_import_output=feature_mportance(X_train,rf)
partial_depend_plot(X_train,X_test,rf)
#feature_permutation(X_train,y_train,rf)



    