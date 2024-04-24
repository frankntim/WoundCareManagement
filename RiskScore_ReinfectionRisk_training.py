# Project
# RiskScore - Reinfection Risk model for retraining datasets with logging activities
# Descriptions:

#        Make prediction of test dataset
#        Log key important activities into LOG table
#        Log scores into SCORE table
#        Log errors into ERROR table



import pandas as pd
import os
from datetime import datetime
import pyodbc
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,roc_auc_score,roc_curve,auc,ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.metrics import  classification_report,accuracy_score, recall_score,precision_score, f1_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate, GridSearchCV, KFold, RandomizedSearchCV, train_test_split


from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

import joblib

import warnings
warnings.filterwarnings('ignore')




def load_data(connection_string):
  '''
  This function loads data from a specific table through a dedicated connection string
  '''
    WC = pyodbc.connect(connection_string)

    sql_query = pd.read_sql_query('''
                                SELECT
                                 *
                                FROM WC.dbo.MemberAnalysisMemberSummary12OCT22
                                Where Episode = 1
                                ''' ,WC)
    member_summary_df = pd.DataFrame(sql_query)

    sql_query = pd.read_sql_query('''
                                SELECT
                                TPSClaimID,
                                TPSMemberID,
                                MemberYearId,
                                StartDateofService,
                                EndDateofService,
                                AdmitDtm,
                                DischargeDtm,
                                PlaceOfServiceCode,
                                PrimaryMSDRG,
                                TypeofBill,
                                AdmissionType,
                                DischargeStatus,
                                PrimaryDx,
                                RevenueCode,
                                CPTHCPCSCode,
                                Adjustment,
                                AllowedAmount,
                                AddedDTM,
                                Top_Level_Category,
                                Category,
                                TurningPoint_Relevant_Comorbidity_Category,
                                SubCategory,
                                Label,
                                Hierarchy,
                                Billing_InNetwork,
                                Billing_Classification,
                                Gender,
                                MemberName,
                                MemberAge,
                                DRG_Final_Category,
                                HCPCS_Final_Category,
                                ICD10DX_Final_Category,
                                ICD10PCS_Final_Category,
                                CPT_Final_Category,
                                ServiceYear,
                                Episode

                                FROM WC.dbo.MemberAnalysisClaimsDetail
                                Where Episode = 1
                                ''' ,WC)
    member_claim_df = pd.DataFrame(sql_query)
    
    return member_claim_df, member_summary_df



def age_category(age):
  '''
  This function bins  the ages into 4 categories
  '''
    if age < 18:
        return 1
    if 18 <= age < 48:
        return 2
    if 48 <= age < 64:
        return 3
    else:
        return 4


def to_1D(series):
  return pd.Series([x for _list in series for x in _list])


def get_item_frequency_in_counts(item_lists, unique_items, prefix = ''):
  '''
  Generates counts of categorical list items in dataframe cell as one-hot encoded columns
  '''
# Create empty dict
    count_dict = {}
    # Make sure there are no Nones in item_list
    
    # Loop through all the tags
    for i, item in enumerate(unique_items):
        
        # Apply boolean mask
        count_dict[item] = item_lists.apply(lambda x: x.count(item))
            
    # Return the results as a dataframe
    df = pd.DataFrame(count_dict)
    df.replace({False: 0, True: 1}, inplace=True)
    
    df.columns = [prefix + x  for x in df.columns]
    
    return df


def get_item_frequency_in_bool(item_lists, unique_items, prefix = ''):
  '''
  Generates unique counts of categorical list items in dataframe cell as one-hot encoded columns
  '''
# Create empty dict
    bool_dict = {}
    # Make sure there are no Nones in item list
    
    # Loop through all the tags
    for i, item in enumerate(unique_items):
        
        # Apply boolean mask
        bool_dict[item] = item_lists.apply(lambda x: item in x)
        
    # Return the results as a dataframe
    df = pd.DataFrame(bool_dict)
    df.replace({False: 0, True: 1}, inplace=True)
    
    df.columns = [prefix + x  for x in df.columns]        
    # Return the results as a dataframe
    return df

def drop_out_adjustment_claims(member_analytic_dataset_df):
  '''
  This function drops certain claims whose allowed amount are negative
  '''
    # Select all claims where adjustment  = 1
    adjustment_one_df = member_analytic_dataset_df[member_analytic_dataset_df['Adjustment']==1][['TPSClaimID','AllowedAmount','MemberName','AdmissionType','AddedDTM']]
    adjustment_one_df['AllowedAmount'] = abs(adjustment_one_df['AllowedAmount'])
    adjustment_one_df = adjustment_one_df.astype(str)

    # Prepare the master dataset
    member_analytic_dataset_df = member_analytic_dataset_df[member_analytic_dataset_df['Adjustment']==0]
    member_analytic_dataset_df[['TPSClaimID',
                                'AllowedAmount',
                                'MemberName',
                                'AdmissionType',
                                'AddedDTM']] = member_analytic_dataset_df[['TPSClaimID',
                                                                           'AllowedAmount',
                                                                           'MemberName',
                                                                           'AdmissionType',
                                                                           'AddedDTM']].astype(str)


    # Merge the fraud dataset with the master to get duplicates
    member_analytic_merged_df = (
        member_analytic_dataset_df.merge(adjustment_one_df, 
                  on=['TPSClaimID','AllowedAmount','MemberName','AdmissionType','AddedDTM'],
                  how='left', 
                  indicator=True)
    )
    # Select only one of the duplicates
    member_analytic_merged_df = member_analytic_merged_df.drop_duplicates(['TPSClaimID','AllowedAmount','MemberName','AdmissionType','AddedDTM'])
    member_analytic_merged_df['AllowedAmount'] = pd.to_numeric(member_analytic_merged_df['AllowedAmount'])
    
    return member_analytic_merged_df



def dq_training_dataset(training_dataset_df):
  '''
  This function performs data quality check 
  '''
    # DQ Check 1: Replace all null dates in Admission with NaN and fill it with ServiceDates

    print(f'initial dimension of dataset: {training_dataset_df.shape}')
    training_dataset_df[['AdmitDtm','DischargeDtm']] = training_dataset_df[['AdmitDtm','DischargeDtm']].astype(str).replace({'NaT': np.NaN})
    training_dataset_df[['AdmitDtm','DischargeDtm']] = training_dataset_df[['AdmitDtm','DischargeDtm']].replace({'1753-01-01': np.NaN})
    training_dataset_df['AdmitDtm'] = training_dataset_df['AdmitDtm'].mask(training_dataset_df['AdmitDtm'].isna(), 
                                                                         training_dataset_df['StartDateofService'])
    training_dataset_df['DischargeDtm'] = training_dataset_df['DischargeDtm'].mask(training_dataset_df['DischargeDtm'].isna(), 
                                                                                 training_dataset_df['EndDateofService'])

    # DQ Check 2:
    training_dataset_df[['AdmitDtm','DischargeDtm']] = training_dataset_df[['AdmitDtm','DischargeDtm']].apply(pd.to_datetime)
    training_dataset_df['Length_of_Stay'] = (training_dataset_df['DischargeDtm'] - training_dataset_df['AdmitDtm']).dt.days
    training_dataset_df['AdmitYear'] = pd.DatetimeIndex(training_dataset_df['AdmitDtm']).year
    training_dataset_df['AdmitMonth'] = pd.DatetimeIndex(training_dataset_df['AdmitDtm']).month


    # DQ Check 5:AllowedAmount  > 0
    training_dataset_df = training_dataset_df[training_dataset_df['AllowedAmount'] > 0]


    # DQ Check 6: Drop out Adjustments  =1
    training_dataset_df = drop_out_adjustment_claims(training_dataset_df)
    print(f'final dimension of dataset: {training_dataset_df.shape}')
    
    return training_dataset_df


def select_cutoff_variation(dataset_df, column_name ,list_size = 3000):
    
    df = pd.DataFrame(dataset_df[column_name].value_counts())
    df = df.reset_index()
    df.columns = ['data','count']
    df = df.head(list_size)
    value_list = list(df['data'].values)
    dataset_df[column_name] = dataset_df[column_name].apply(lambda i: i if i in value_list else 'OTHER')
    return dataset_df



def get_aggregated_member_from_claim_dataset(dataset_df):
    '''
    Since a member can have multiple claims, this function aggregate all the member's claims
    '''
    
    
    categorical_columns = ['PlaceOfServiceCode',
                    'PrimaryMSDRG',
                    'TypeofBill',
                    'AdmissionType',
                    'DischargeStatus',
                    'PrimaryDx',
                    'RevenueCode',
                    'CPTHCPCSCode',
                    'Top_Level_Category',                
                    'Category',
                    'TurningPoint_Relevant_Comorbidity_Category',
                    'SubCategory',
                    'Label',
                    'Hierarchy',
                    'Billing_InNetwork',
                    'Billing_Classification',
                    'Gender',
                    'DRG_Final_Category',
                    'HCPCS_Final_Category',
                    'ICD10DX_Final_Category',
                    'ICD10PCS_Final_Category',
                    'CPT_Final_Category']

    for cat in categorical_columns:

        dataset_df[cat]= dataset_df[cat].apply(lambda x: x if x is not None else 'OTHER')
        
    for cat in categorical_columns:
        dataset_df[cat]= dataset_df[cat].astype(str)
        
    
    
    year_list = list(dataset_df['ServiceYear'].unique())
    year_list = sorted(year_list)
    
    member_summary_list = []

    for year in year_list:

        member_per_year = dataset_df.query('ServiceYear == @year')
        member_per_year = member_per_year.drop_duplicates()


        agg_members_df = member_per_year.groupby(['TPSMemberID','Gender','ServiceYear'], as_index=False).agg({
                                                                                'TPSClaimID' : lambda x: x.count(),
                                                                                 'MemberAge': 'mean',
                                                                                'PrimaryDx': lambda x:x.tolist(),
                                                                                'PrimaryMSDRG': lambda x:x.tolist(),
                                                                                 'CPTHCPCSCode': lambda x:x.tolist(),
                                                                                 'Category': lambda x:x.tolist(),
                                                                                 'Label':lambda x:x.tolist(),
                                                                                 'PlaceOfServiceCode': lambda x:x.tolist(),
                                                                                 'AdmissionType':lambda x:x.tolist(),
                                                                                 'DischargeStatus':lambda x:x.tolist(),
                                                                                 'Billing_InNetwork': lambda x:x.tolist(),
                                                                                'Billing_Classification':lambda x:x.tolist(),
                                                                                'TypeofBill':lambda x:x.tolist(),
                                                                                 'RevenueCode':lambda x:x.tolist() ,
                                                                                'SubCategory':lambda x:x.tolist() ,
                                                                                'Hierarchy': lambda x:x.tolist(),
                                                                                'DRG_Final_Category':lambda x:x.tolist(),
                                                                                'HCPCS_Final_Category':lambda x:x.tolist(),
                                                                                 'ICD10DX_Final_Category':lambda x:x.tolist() ,
                                                                                'CPT_Final_Category':lambda x:x.tolist() ,
                                                                                'AllowedAmount': 'sum'

                                                                                })
        member_summary_list.append(agg_members_df)
        
    member_summary_list_df  = pd.concat(member_summary_list)
    
    # Final DQ
    member_summary_list_df = member_summary_list_df.dropna()
    member_summary_list_df['TPSMemberID'] = member_summary_list_df['TPSMemberID'].astype(str)
    member_summary_list_df['ServiceYear'] = member_summary_list_df['ServiceYear'].astype(str)


    member_summary_list_df['Gender'] = member_summary_list_df['Gender'].astype(str)
    member_summary_list_df['Gender'] = member_summary_list_df['Gender'].apply({'M':0, 'F':1}.get)
    member_summary_list_df['Gender'] = member_summary_list_df['Gender'].astype(int).astype(str)
    member_summary_list_df = member_summary_list_df.drop('MemberAge', axis = 1)
    
    return member_summary_list_df



                                                     

def generate_model_training_set_from_encodings(dataset_df, 
                                              standard_encoding_dataset_df, 
                                              categorical_cols,
                                              categorical_col_names,
                                              standardized_encoding_columns,
                                              target_variable = "ReinfectionEpisode",
                                              additional_columns = ['EpisodeLOS','AvgEpisodeLength']):
    '''
    One hot encoding dataset is generated for each member year
    '''
    
    dataset_df['ServiceYear'] = dataset_df['ServiceYear'].astype(int)
    year_list = list(dataset_df['ServiceYear'].unique())
    year_list = sorted(year_list)
    
    fixed_columns = ['MemberYearId','ServiceYear','Gender','Age']
    
    
    if additional_columns is not None:
        key_columns = fixed_columns + additional_columns + [target_variable]
    else:
        key_columns = fixed_columns + [target_variable]

            
    member_modeling_list = []   
    
    for year in year_list:

        member_per_year = dataset_df.query('ServiceYear == @year')
        member_modeling_df = pd.DataFrame()

        for cat_col, cat_name in zip(categorical_cols,categorical_col_names):

            unique_codes = list(standard_encoding_dataset_df[cat_col].values[0])

            codes_df = get_item_frequency_in_counts(member_per_year[cat_col], 
                                                                         unique_codes, 
                                                                         prefix=cat_name)

            member_modeling_df = pd.concat([member_modeling_df,codes_df ], axis = 1)

        index_df = member_per_year[key_columns]
        
        index_df = index_df.reset_index(drop=True)
        # Standardize the columns
        member_modeling_df = member_modeling_df[standardized_encoding_columns]
        member_modeling_df = member_modeling_df.reset_index(drop=True)
        
        member_modeling_df = pd.concat([index_df,member_modeling_df], axis=1)
        member_modeling_list.append(member_modeling_df)

    member_modeling_list_df = pd.concat(member_modeling_list, axis = 0)
    
    member_modeling_list_df = member_modeling_list_df.drop_duplicates()
    
    #member_modeling_list_df = member_modeling_list_df.set_index('MemberYearId')
    
    member_modeling_list_df['Gender'] = member_modeling_list_df['Gender'].astype(int)
    
    # 
    
    target = member_modeling_list_df[target_variable]
    #member_modeling_list_df = member_modeling_list_df.drop(target_variable, axis = 1)
    
    col_to_drop = ['TPSMemberID','ServiceYear','AllowedAmount']
    
    for col in col_to_drop:
        if col in member_modeling_list_df.columns:
            member_modeling_list_df = member_modeling_list_df.drop(col,axis = 1)
    
    return member_modeling_list_df






def get_aggregated_member_year_from_claim_dataset(dataset_df):
    #Make sure these columns exist in the input dataset
    
    categorical_columns = ['PlaceOfServiceCode',
                    'PrimaryMSDRG',
                    'TypeofBill',
                    'AdmissionType',
                    'DischargeStatus',
                    'PrimaryDx',
                    'RevenueCode',
                    'CPTHCPCSCode',
                    'Top_Level_Category',                
                    'Category',
                    'SubCategory',
                    'Label',
                    'Hierarchy',
                    'Billing_InNetwork',
                    'Billing_Classification',
                    'Gender',
                    'DRG_Final_Category',
                    'HCPCS_Final_Category',
                    'ICD10DX_Final_Category',
                    'ICD10PCS_Final_Category',
                    'CPT_Final_Category']

    for cat in categorical_columns:

        dataset_df[cat]= dataset_df[cat].apply(lambda x: x if x is not None else 'OTHER')
        
    for cat in categorical_columns:
        dataset_df[cat]= dataset_df[cat].astype(str)
        
    
    
    year_list = list(dataset_df['ServiceYear'].unique())
    year_list = sorted(year_list)
    
    claim_aggregation_list = []

    for year in year_list:

        claims_per_year = dataset_df.query('ServiceYear == @year')
        claims_per_year = claims_per_year.drop_duplicates()


        agg_members_df = claims_per_year.groupby(['TPSMemberID','MemberYearId','ServiceYear'], as_index=False).agg({
                                                                                'PrimaryDx': lambda x:x.tolist(),
                                                                                'PrimaryMSDRG': lambda x:x.tolist(),
                                                                                 'CPTHCPCSCode': lambda x:x.tolist(),
                                                                                 'Category': lambda x:x.tolist(),
                                                                                 'Label':lambda x:x.tolist(),
                                                                                 'PlaceOfServiceCode': lambda x:x.tolist(),
                                                                                 'AdmissionType':lambda x:x.tolist(),
                                                                                 'DischargeStatus':lambda x:x.tolist(),
                                                                                 'Billing_InNetwork': lambda x:x.tolist(),
                                                                                'Billing_Classification':lambda x:x.tolist(),
                                                                                'TypeofBill':lambda x:x.tolist(),
                                                                                 'RevenueCode':lambda x:x.tolist() ,
                                                                                'SubCategory':lambda x:x.tolist() ,
                                                                                'Hierarchy': lambda x:x.tolist(),
                                                                                'DRG_Final_Category':lambda x:x.tolist(),
                                                                                'HCPCS_Final_Category':lambda x:x.tolist(),
                                                                                 'ICD10DX_Final_Category':lambda x:x.tolist() ,
                                                                                'CPT_Final_Category':lambda x:x.tolist(),
                                                                                'AllowedAmount':'sum'

                                                                                })
        claim_aggregation_list.append(agg_members_df)
        
    claim_aggregation_list_df  = pd.concat(claim_aggregation_list)
    
    # Final DQ
    claim_aggregation_list_df = claim_aggregation_list_df.dropna()
    claim_aggregation_list_df['MemberYearId'] = claim_aggregation_list_df['MemberYearId'].astype(str)
    
    return claim_aggregation_list_df


def parse_encoding_into_standardized_column(dataset_df):
    
    cat_cols = [
                'PrimaryDx', 
                'PrimaryMSDRG',
                'RevenueCode',
                'CPTHCPCSCode',
                'PlaceOfServiceCode',
                'Billing_InNetwork',
               'Billing_Classification',
               'TypeofBill',
                 'AdmissionType', 
                 'DischargeStatus', 
                 'Category',
                 'SubCategory',
                'Hierarchy',
                  'Label',
                'DRG_Final_Category',
                'HCPCS_Final_Category',
                'ICD10DX_Final_Category',
                'CPT_Final_Category'
                 ]

    cat_names = ['DX_', 
                 'DRG_', 
                 'REV_',
                 'CPT_',
                 'POS_',
                 'BI_',
                 'BC',
                 'TB_', 
                 'AT_', 
                 'DS_', 
                 'Cat_', 
                 'SubCat_', 
                 'HRCY_',
                 'LAB_',
                 'DCat_',
                 'HCCat_',
                 'ICDCat_',
                 'CPTCat_']
    
    

    dummy_dataset_df = pd.DataFrame()

    for cat_col, cat_name in zip(cat_cols, cat_names):

        unique_code = list(dataset_df[cat_col].values[0])

        codes_df = get_item_frequency_in_counts(dataset_df[cat_col], unique_code, 
                                                                     prefix=cat_name)

        dummy_dataset_df = pd.concat([dummy_dataset_df,codes_df ], axis = 1)
        
        standardized_columns = list(dummy_dataset_df.columns)
        
    
    return cat_cols, cat_names, standardized_columns







def transform_data_into_Reinfection_predictors(claim_dataset, 
                                               member_summary_dataset,
                                                summary_columns,
                                               count_encoding = True,
                                               standardized_file_location = 'standardized_dummy_data.pkl'):
    
    aggregated_claim_df = get_aggregated_member_year_from_claim_dataset(claim_dataset)

    
    
    standard_encoding_data_df = pd.read_pickle(standardized_file_location)
    cat_cols, cat_names, standardized_columns = parse_encoding_into_standardized_column(standard_encoding_data_df)

    base_encoded_dataset_df = generate_model_training_set_from_encodings(aggregated_claim_df, 
                                                      standard_encoding_data_df, 
                                                      cat_cols,
                                                      cat_names,
                                                      standardized_columns,
                                                      count_encoding = count_encoding,
                                                       )

    member_summary_df = member_summary_dataset[summary_columns]
    member_summary_df['Age_Cat'] = member_summary_df['Age'].apply(age_category)
    #member_summary_df = member_summary_df.drop('Age', axis = 1)

    predictor_dataset_df = member_summary_df.merge(base_encoded_dataset_df,
                                                             on = ['MemberYearId'],
                                                             how = 'inner')
    
    # Final clean out
    predictor_dataset_df = predictor_dataset_df.fillna(0)
    
    return predictor_dataset_df


def cross_validation_on_models(X_data, y, classifier, fold = 5):
    
    # Change the columns names
    X = X_data.copy()
    #X.columns = [ f'col_{i}' for i in range(X.shape[1])]
    
    cv = StratifiedKFold(n_splits = fold)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots()
    for i, (train, test) in enumerate(cv.split(X, y)):
        classifier.fit(X.loc[train], y.loc[train])
        viz = RocCurveDisplay.from_estimator(
            classifier,
            X.loc[test],
            y.loc[test],
            name="ROC fold {}".format(i),
            alpha=0.3,
            lw=1,
            ax=ax,
        )
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        title="Receiver operating characteristic example",
    )
    ax.legend(loc="lower right")
    plt.show()
    
def evaluate_on_validation(classifier, validation_data, validation_target):
    
    validationX = validation_data.copy()
    
    validationX.columns = [ f'col_{i}' for i in range(validationX.shape[1])]

    predict_target = classifier.predict(validationX)

    print(classification_report(validation_target, predict_target))

    #plot_roc_curve(validation_target, predict_target)
    #fpr, tpr, _ = roc_curve(validation_target,  predict_target)
    #auc = roc_auc_score(validation_target, predict_target)

    #create ROC curve
    #plt.plot(fpr,tpr,label="AUC="+str(auc))
    #plt.ylabel('True Positive Rate')
    #plt.xlabel('False Positive Rate')
    #plt.legend(loc=4)
    #plt.show()
    
    
    #c_matrix = confusion_matrix(validation_target, predict_target)
    #ConfusionMatrixDisplay(c_matrix).plot()
    
    print(f"Accuracy: {accuracy_score(validation_target, predict_target)}")
    print(f"Recall Score: {recall_score(validation_target, predict_target)}")
    print(f"Precision Score: {precision_score(validation_target, predict_target)}")
    print(f"F1 Score: {f1_score(validation_target, predict_target)}")
    
    return

    
def train_evaluate_lightgbm(training_data_df, train_target, validation_data_df, validation_target):
    
    scoring = ['precision_macro', 'recall_macro','f1_macro']

    training_X = training_data_df.copy()
    training_X.columns =  [ f'col_{i}' for i in range(training_X.shape[1])]
    
    print('============Cross-validation with Light GBM============')

    params = {'boosting_type': 'gbdt', 'max_depth': 6, 'objective': 'binary', 
                  'num_leaves': 64, 'learning_rate': 0.1, 'max_bin': 512, 
                  'subsample_for_bin': 200, 'subsample': 1, 'subsample_freq': 1,
                  'colsample_bytree': 0.8, 'reg_alpha': 5, 'reg_lambda': 10, 
                  'min_split_gain': 0.5, 'min_child_weight': 1, 
                  'min_child_samples': 5, 'scale_pos_weight': 1, 'num_class': 1, 
                  'metric': 'binary_error'}
    
    lgb_classifier = lgb.LGBMClassifier()

    cv_results = cross_validate(lgb_classifier,  training_X, train_target, cv=5, scoring=scoring)

    f1_score_df = pd.DataFrame(cv_results['test_f1_macro'], columns=['F1-score'])
    f1_score_df.plot.barh(rot=0, color='black')
    display(f1_score_df)

    lgb_classifier.fit(training_X, train_target)
    feature_importance(lgb_classifier, 
                       training_data_df)
    
    y_predict_lgb = evaluate_on_validation(lgb_classifier, validation_data_df, validation_target)
    
    plt.figure(figsize=(10,8))
    explainer = shap.TreeExplainer(lgb_classifier)
    importance = training_X.copy()
    importance_with_col_names = training_data_df.copy()
    shap_values = explainer.shap_values(importance)
    shap.summary_plot(shap_values, importance_with_col_names,plot_type='bar')
    
    return 


def train_evaluate_catboost(training_data_df, train_target, validation_data_df, validation_target):
    
    print('============Cross-validation with CatBoost============')
    scoring = ['precision_macro', 'recall_macro','f1_macro']

    training_X = training_data_df.copy()
    training_X.columns =  [ f'col_{i}' for i in range(training_X.shape[1])]
    
    
    cat_classifier = CatBoostClassifier(
                            iterations=100,
                            #     verbose=5,
                            )

    cv_results = cross_validate(cat_classifier,  training_X, train_target, cv=5, scoring=scoring)

    f1_score_df = pd.DataFrame(cv_results['test_f1_macro'], columns=['F1-score'])
    f1_score_df.plot.barh(rot=0, color='black')
    display(f1_score_df)

    cat_classifier.fit(training_X, train_target)
    
    importance_df = feature_importance(cat_classifier, 
                                       training_data_df)
    y_predict_cat = evaluate_on_validation(cat_classifier, validation_data_df, validation_target)
    
    plt.figure(figsize=(10,8))
    explainer = shap.TreeExplainer(cat_classifier)
    importance = training_X.copy()
    importance_with_col_names = training_data_df.copy()
    shap_values = explainer.shap_values(importance)
    shap.summary_plot(shap_values, importance_with_col_names,plot_type='bar')
    
    return 
 
    
def feature_importance(model, training_data):
    
    feature_names = np.array(training_data.columns)
    sorted_idx = model.feature_importances_.argsort()

    importance_df = pd.DataFrame([feature_names[sorted_idx], model.feature_importances_[sorted_idx]])
    
    importance_df = importance_df.T
    importance_df.columns = ['features','importance']
    importance_df = importance_df.sort_values(by=['importance'], ascending = False)
    importance_df = importance_df.head(15)
    
    plt.figure(figsize=(6,5))
    sns.barplot(x = 'importance',
            y = 'features',
            data = importance_df,
                color = 'black')
    plt.show()
    
    return 


def performance_cut_off(prediction_result_df, 
                        label = 'AmputationRisk',
                       label_proba = 'AmputationRisk_Prediction_Proba',
                       threshold_list = [0.875, 0.90, 0.925, 0.95, 0.975]):
    
    cut_off_performance = pd.DataFrame(columns=['Cut-off', 'Accuracy','Recall','Precision','F1-Score'])


    
    
    for threshold in threshold_list:
        
        column_name = f'{label}_Predict_{str(threshold)}'
        prediction_result_df[column_name] = (prediction_result_df[label_proba].                                                                    map(lambda x: 1 if x >= threshold else 0))
        accuracy = accuracy_score(prediction_result_df[label],
                                  prediction_result_df[column_name])
        
        recall = recall_score(prediction_result_df[label],
                              prediction_result_df[column_name])
        precision = precision_score(prediction_result_df[label],
                                    prediction_result_df[column_name])
        
        f1_value = f1_score(prediction_result_df[label],
                            prediction_result_df[column_name])
        
        cut_off_performance.loc[len(cut_off_performance.index)] = [f'{threshold}',accuracy, recall, precision, f1_value] 
        
    return cut_off_performance


def plot_roc_curves(data_list,
                    label='ReinfectionEpisode', 
                    threshold = 0.875):
    
    label_predict = f'{label}_Predict_{threshold}'
    
    
    model_names = ['XGB', 'LightGBM','CatBoost', 'LogReg']
    plt.figure(0).clf()
    i = 0
    for data in data_list:
        
        data_df  = data[[label,label_predict]]
        model_name = model_names[i]
        fpr, tpr, thresh = roc_curve(data_df[label], 
                                         data_df[label_predict])
        auc = roc_auc_score(data_df[label], 
                                             data_df[label_predict])
        f1_value = f1_score(data_df[label], 
                                             data_df[label_predict])
        label_ = f"{model_name}: AUC= {str(round(auc,3))}: F1={round(f1_value,3)}"
        plt.plot(fpr,tpr,label=label_)
        i = i + 1
        
    
    plt.legend(loc=0)
    plt.title(f"{label} Performance Metrics at {threshold} threshold")
    
    return


### Parameters
ID = 10
Step = 0
Name = "RiskScore-ReinfectionRisk-Train"

ERROR_TYPE_LIST = ['Start',
                  'Macros Definition Failed',
                  'Config-Parameters Failed to Load',
                  'Connection to Database Failed',
                  'Failed to load Train/Validation Data',
                   'Failed to transform Train/Validation Data',
                  'Failed to Generate Train/Validation Features',
                  'Failed to Train w/ Tree-Based Models:',
                  'Failed to Train w/ Linear Models : Logistic Regression',
                   'Table Insertion Failed'
                 ]
# Start Logging
log = pd.DataFrame(columns=['Name','ID','Type','Value','Step','TimeStamp'])
current_dateTime = datetime.now()
print('Start ', current_dateTime)

Step = Step + 1
Activity = 'Start'
log.loc[len(log.index)] = [Name, ID, Activity,datetime.now(), Step,datetime.now()] 




REINFECTION_PARAMS = {
    'summary_columns' : ['TPSMemberID', 
                     'ServiceYear', 
                     'MemberYearId', 
                     'Gender', 
                     'Age',
                     'EpisodeLOS',
                     'ReinfectionEpisode',
                    'ChronicEpisode',
                    'DistClms',
                    'Clms',
                    'DebridementEpisodes',
                    'WCDistClms', 
                    'WCDistClmsPerEpisode',
                   'AmputationDistClms', 
                    'HBODistClms',
                    'SkinSubDistClms'
                        ],
          'columns_to_remove': ['TPSMemberID',
                                'ServiceYear',
                                'MemberYearId',
                                'ReinfectionEpisode',
                                'AllowedAmount',
                                'EpisodeDuration',
                                'Age_Cat',
                                'ICDCat_Infection',
                                'ICDCat_Infections',
                                'ICDCat_Severe Infection',
                                'CPTCat_Infection Treatment'
                               ],
                "encoding_dimension": 300,
                'is_count_encoding' : True,
                "standard_dummy_loc": "./utilities/standardized_dummy_data.pkl",
                "logistic_reg_model_loc": "./utilities/LogisticRegression_reinfection_risk_model.pickle",
                "lgbm_classification_model_loc": "./utilities/lgbm_reinfection_risk_model.pickle",
                "xgb_classification_model_loc" : "./utilities/xgb_reinfection_risk_model.pickle",
                "results_loc": "./utilities/RiskScore_ReinfectionRisk_Results.csv",
                "log_table_insert_statement": "INSERT INTO [Log].[dbo].[log] VALUES (?,?,?,?,?,?)",
                "scoring_table_insert_statement": "INSERT INTO [Log].[dbo].[scoring] VALUES (?,?,?,?,?)",
                "error_table_insert_statement": "INSERT INTO [Log].[dbo].[error] VALUES (?,?,?,?,?,?)",
                "connection_string" : ""
                }





try:
    # Configuration Parameters
    Step += 1
    Activity = 'Config-Parameters Loaded'
    
    SUMMARY_COLUMNS = REINFECTION_PARAMS["summary_columns"]
    STANDARD_DUMMY_LOCATION = REINFECTION_PARAMS["standard_dummy_loc"]
    IS_COUNT_ENCODING = REINFECTION_PARAMS["is_count_encoding"]

    XGB_MODEL_LOC = REINFECTION_PARAMS["xgb_classification_model_loc"]
    COLUMS_TO_REMOVE = REINFECTION_PARAMS['columns_to_remove']
    
    LOG_TABLE_INSERT_STATEMENT = REINFECTION_PARAMS['log_table_insert_statement']
    SCORING_TABLE_INSERT_STATEMENT = REINFECTION_PARAMS['scoring_table_insert_statement']
    ERROR_TABLE_INSERT_STATEMENT = REINFECTION_PARAMS['error_table_insert_statement']
    CONNECTION_STRING = REINFECTION_PARAMS['connection_string']
    
    print(Activity, current_dateTime)
    log.loc[len(log.index)] = [Name, ID, Activity,datetime.now(), Step,datetime.now()] 

    ############################### Establish the global connection here
    
    current_dateTime = datetime.now()
    Activity = 'Connection Established'
    Step = Step + 1

    cnxn = pyodbc.connect(CONNECTION_STRING
                         )
    print(Activity, current_dateTime)
    log.loc[len(log.index)] = [Name, ID, Activity,datetime.now(), Step, datetime.now()]
    
    ###############################################
    current_dateTime = datetime.now()
    Step = Step +  1
    member_analysis_claim_df,member_analysis_summary_df = load_data(CONNECTION_STRING)
    
    
    TRAIN_DATA = member_analysis_claim_df[member_analysis_claim_df['ServiceYear'].isin([2017,2018, 2019, 2020])]
    training_data_df = dq_training_dataset(TRAIN_DATA)
    
    TEST_DATA  = member_analysis_claim_df[member_analysis_claim_df['ServiceYear'].isin([2021])]
    validation_data_df = dq_training_dataset(TEST_DATA)

    
    Activity = 'Train-Validation Data Loaded'
    print(Activity, current_dateTime)
    log.loc[len(log.index)] = [Name, ID, Activity,datetime.now(), Step ,datetime.now()]
    ############################################################
    current_dateTime = datetime.now()
    Step = Step + 1
    Activity = 'Train-Validation Data Transformed'
    
    
    
    train_df  = transform_data_into_Reinfection_predictors(training_data_df, 
                                               member_analysis_summary_df,
                                                summary_columns = SUMMARY_COLUMNS,
                                                count_encoding = IS_COUNT_ENCODING,
                                               standardized_file_location = STANDARD_DUMMY_LOCATION)
    
    test_df  = transform_data_into_Reinfection_predictors(validation_data_df, 
                                               member_analysis_summary_df,
                                                summary_columns = SUMMARY_COLUMNS,
                                                count_encoding = IS_COUNT_ENCODING,
                                               standardized_file_location = STANDARD_DUMMY_LOCATION)
    print(Activity, current_dateTime)
    log.loc[len(log.index)] = [Name, ID, Activity,datetime.now(), Step ,datetime.now()] 
    ######################################################################
    current_dateTime = datetime.now()
    Step = Step + 1
    Activity = 'Train/Val Features Generated'
    
    trainingX = train_df.copy()
    trainY = train_df['ReinfectionEpisode']

    for col in COLUMS_TO_REMOVE:
        if col in trainingX:
            trainingX = trainingX.drop(col, axis = 1)

    training_X = trainingX.copy()
    training_X.columns =  [ f'col_{i}' for i in range(training_X.shape[1])]
    
    # trainingX contains true column names
    training_cols = trainingX.columns
    
    testX = test_df.copy()
    testY = test_df['ReinfectionEpisode']

    for col in COLUMS_TO_REMOVE:
        if col in testX:
            testX = testX.drop(col, axis = 1)

    testX = testX[training_cols]
    
    test_X = testX.copy()
    test_X.columns =  [ f'col_{i}' for i in range(test_X.shape[1])]
    print(Activity, current_dateTime)
    log.loc[len(log.index)] = [Name, ID, Activity,datetime.now(), Step ,datetime.now()] 
    #################################################################
    ################ Train with Tree-Based Models
    Step = Step + 1
    current_dateTime = datetime.now()
    Activity = 'Train w/ Tree: XGBoost'
    
    xgb = XGBClassifier(n_estimators=100,
        objective= "binary:logistic",
        booster = "gbtree",
        eval_metric= "auc",
        eta= 0.1,
        tree_method= 'exact',
        max_depth= 6,
        subsample= 1,
        colsample_bytree = 1,
        min_child_weight= 1,
        scale_pos_weight=19
       )
    #cross_validation_on_models(training_X, trainY, xgb, fold = 5)
    print('===========Train with XGBoost====== ')
    xgb.fit(training_X, trainY)
    feature_importance(xgb, trainingX)
    evaluate_on_validation(xgb, test_X, testY)
    
    xgb_prediction = xgb.predict(test_X)
    
    accuracy = accuracy_score(testY, xgb_prediction)
    recall = recall_score(testY, xgb_prediction)
    precision = precision_score(testY, xgb_prediction)
    f1_value = f1_score(testY, xgb_prediction)
    
    reinfection_result_xgb_df = test_df[['ReinfectionEpisode']]


    reinfection_result_xgb_df['ReinfectionEpisode_Prediction'] = xgb.predict(test_X)
    reinfection_result_xgb_df['ReinfectionEpisode_Prediction_Proba'] = xgb.predict_proba(test_X)[:,1]
    xgb_cutoff_df = performance_cut_off(reinfection_result_xgb_df, 
                        label = 'ReinfectionEpisode',
                       label_proba = 'ReinfectionEpisode_Prediction_Proba',
                       threshold_list = [0.875, 0.90, 0.925, 0.95, 0.975])
    
    
    joblib.dump(xgb, XGB_MODEL_LOC) 
    
    ############### LGBoost
    print('===========Train with LightGBM====== ')
    lgb_classifier = lgb.LGBMClassifier()
    lgb_classifier.fit(training_X, trainY)
    #feature_importance(lgb_classifier, trainingX)
    evaluate_on_validation(lgb_classifier, test_X, testY)
    
    
    reinfection_lgbm_result_df = test_df[['ReinfectionEpisode']]


    reinfection_lgbm_result_df['ReinfectionEpisode_Prediction'] = lgb_classifier.predict(test_X)
    reinfection_lgbm_result_df['ReinfectionEpisode_Prediction_Proba'] = lgb_classifier.predict_proba(test_X)[:,1]
    lgbm_cutoff_df = performance_cut_off(reinfection_lgbm_result_df, 
                        label = 'ReinfectionEpisode',
                       label_proba = 'ReinfectionEpisode_Prediction_Proba',
                       threshold_list = [0.875, 0.90, 0.925, 0.95, 0.975])
    
    ############## CatBoost
    print('===========Train with CatBoost====== ')
    cat_classifier = CatBoostClassifier(
                            iterations=10,
                            )
    cat_classifier.fit(training_X, trainY)
    feature_importance(cat_classifier, trainingX)
    evaluate_on_validation(cat_classifier, test_X, testY)
    
    
    reinfection_cat_result_df = test_df[['ReinfectionEpisode']]


    reinfection_cat_result_df['ReinfectionEpisode_Prediction'] = cat_classifier.predict(test_X)
    reinfection_cat_result_df['ReinfectionEpisode_Prediction_Proba'] = cat_classifier.predict_proba(test_X)[:,1]
    cat_cutoff_df = performance_cut_off(reinfection_cat_result_df, 
                        label = 'ReinfectionEpisode',
                       label_proba = 'ReinfectionEpisode_Prediction_Proba',
                       threshold_list = [0.875, 0.90, 0.925, 0.95, 0.975])
    print(Activity, current_dateTime)
    log.loc[len(log.index)] = [Name, ID, Activity,f1_value, Step ,datetime.now()]
    
    ################ Train with Logistic Regression
    Step = Step + 1
    current_dateTime = datetime.now()
    Activity = 'Train w/ LogisticRegression'
    
    print('===========Train with Logistic Regression====== ')
    model = LogisticRegression()
    solvers = ['liblinear','lbfgs']#,'newton-cg','sag','saga','newton-cholesky']
    penalty = ['l1', 'l2']#,'elasticnet']
    c_values = [100, 10, 1.0, 0.1, 0.01]

    grid = dict(solver=solvers,penalty=penalty,C=c_values)
    #cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=3, scoring='roc_auc',error_score=0)
    grid_result = grid_search.fit(training_X, trainY)



    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
        
    clf = LogisticRegression(solver='liblinear',C = 0.01, penalty='l1').fit(training_X, trainY)


    reinfection_lg_result_df = test_df[['ReinfectionEpisode']]


    reinfection_lg_result_df['ReinfectionEpisode_Prediction'] = clf.predict(test_X)
    reinfection_lg_result_df['ReinfectionEpisode_Prediction_Proba'] = clf.predict_proba(test_X)[:,1]
    #reinfection_result_df['ReinfectionEpisode_Prediction'] = reinfection_result_df['ReinfectionEpisode_Prediction_Proba'].map(lambda x: 1 if x > 0.925 else 0)
    confusion_matrix(reinfection_lg_result_df['ReinfectionEpisode'], reinfection_lg_result_df['ReinfectionEpisode_Prediction'])
    
    accuracy = accuracy_score(reinfection_lg_result_df['ReinfectionEpisode'], reinfection_lg_result_df['ReinfectionEpisode_Prediction'])
    recall = recall_score(reinfection_lg_result_df['ReinfectionEpisode'], reinfection_lg_result_df['ReinfectionEpisode_Prediction'])
    precision = precision_score(reinfection_lg_result_df['ReinfectionEpisode'], reinfection_lg_result_df['ReinfectionEpisode_Prediction'])
    f1_value = f1_score(reinfection_lg_result_df['ReinfectionEpisode'], reinfection_lg_result_df['ReinfectionEpisode_Prediction'])

    evaluate_on_validation(clf, test_X, testY)
    
    logistic_cutoff_df = performance_cut_off(reinfection_lg_result_df, 
                        label = 'ReinfectionEpisode',
                       label_proba = 'ReinfectionEpisode_Prediction_Proba',
                       threshold_list = [0.875, 0.90, 0.925, 0.95, 0.975])
    print(Activity, current_dateTime)
    log.loc[len(log.index)] = [Name, ID, Activity,f1_value, Step ,datetime.now()]
    
    #plot_roc_curves(data_list=[reinfection_result_xgb_df,
    #                       reinfection_lgbm_result_df,
    #                       reinfection_cat_result_df,
    #                       reinfection_lg_result_df],
    #                label='ReinfectionEpisode', 
    #                threshold = 0.875)
    
    # Display the cut-offs from all models
    print('====XGB Performance For Each Thresholds====')
    print(xgb_cutoff_df)
    print('====LGBM Performance For Each Thresholds===')
    print(lgbm_cutoff_df)
    print('====CAT Performance For Each Thresholds===')
    print(cat_cutoff_df)
    print('====Logistic Regression Performance For Each Thresholds====')
    print(logistic_cutoff_df)
    
    ################################################################
    Activity = 'End'
    Step = Step + 1
    current_dateTime = datetime.now()
    log.loc[len(log.index)] = [Name, ID, Activity,datetime.now(), Step,datetime.now()] 
    print(Activity, current_dateTime)
    log = log.drop_duplicates(subset=['Name','ID','Type','Step'])
    # We temporarily elimate the Error/Messages; the Log table does not have those columns
    log_df = log[['Name','ID','Type','Value','Step','TimeStamp']]

    ####################################################################


    log_cursor = cnxn.cursor()
    log_cursor.fast_executemany = True
    log_cursor.executemany(LOG_TABLE_INSERT_STATEMENT, log_df.values.tolist())
    log_cursor.commit()
    log_cursor.close()

except Exception as err:
    
    ErrorPointer = Step -2
    MessageType = ERROR_TYPE_LIST[ErrorPointer]
    JobID = 0
    error_message = str(err)
    print(err)
    current_dateTime = datetime.now()


    log.loc[len(log.index)] = [Name, ID, MessageType,datetime.now(), Step,datetime.now()] 
    log_with_error_df = log[['Name','ID','Type','Value','Step','TimeStamp']]

    # We temporarily elimate the Error/Messages; the Log table does not have those columns
    log_with_error_cursor = cnxn.cursor()
    log_with_error_cursor.fast_executemany = True
    log_with_error_cursor.executemany(LOG_TABLE_INSERT_STATEMENT, log_with_error_df.values.tolist())
    log_with_error_cursor.commit()
    log_with_error_cursor.close()


    print(MessageType)
    # Create the error dataframe
    error_df = pd.DataFrame(columns=['ID','JobID','Name','MessageType','Message','TimeStamp'])
    error_df.loc[len(error_df.index)] = [ID, JobID,Name, MessageType, error_message, current_dateTime] 


    error_cursor = cnxn.cursor()
    error_cursor.fast_executemany = True
    error_cursor.executemany(ERROR_TABLE_INSERT_STATEMENT, error_df.values.tolist())
    error_cursor.commit()
    error_cursor.close()
    
finally:
    cnxn.close()
    print(log)


