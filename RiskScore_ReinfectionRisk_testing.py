# Project
# RiskScore - Reinfection Risk model for Predicting test datasets with logging activities
# Descriptions:

#        Make prediction of test dataset
#        Log key important activities into LOG table
#        Log scores into SCORE table
#        Log errors into ERROR table



ID = 10
Step = 0
Name = "RiskScore-ReinfectionRisk"

ERROR_TYPE_LIST = [
    "Start",
    "Macros Definition Failed",
    "Config-Parameters Failed to Load",
    "Connection to Database Failed",
    "Data Ingestion Failed",
    "Data Transformation Failed",
    "Model Failed to Load",
    "Model Prediction Failed",
    "Prediction Output Failed to Save",
    "Model Performance Failed to Compute",
    "Table Insertion Failed",
]


# In[47]:


import pandas as pd
from datetime import datetime
import pyodbc
import numpy as np
import json
import pickle
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import (
    roc_curve,
    auc,
    recall_score,
    precision_score,
    f1_score,
    accuracy_score,
)
import joblib
import random
import warnings

warnings.filterwarnings("ignore")


Step = +1
current_dateTime = datetime.now()
Activity = "Start"
print(Activity, current_dateTime)
log = pd.DataFrame(
    columns=["Name", "ID", "Type", "Value", "Step", "TimeStamp", "Status", "Msg"]
)
log.loc[len(log.index)] = [
    Name,
    ID,
    Activity,
    datetime.now(),
    Step,
    datetime.now(),
    "Success",
    "Success",
]





def age_category(age):
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




def select_cutoff_variation(dataset_df, column_name ,list_size = 3000):
    
    df = pd.DataFrame(dataset_df[column_name].value_counts())
    df = df.reset_index()
    df.columns = ['data','count']
    df = df.head(list_size)
    value_list = list(df['data'].values)
    dataset_df[column_name] = dataset_df[column_name].apply(lambda i: i if i in value_list else 'OTHER')
    return dataset_df


                                                   

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



def generate_model_training_set_from_encodings(dataset_df, 
                                              standard_encoding_dataset_df, 
                                              categorical_cols,
                                              categorical_col_names,
                                              standardized_encoding_columns,
                                              count_encoding = True,
                                               ):
    
    dataset_df['ServiceYear'] = dataset_df['ServiceYear'].astype(int)
    year_list = list(dataset_df['ServiceYear'].unique())
    year_list = sorted(year_list)
    
    
    key_columns = ['TPSMemberID','MemberYearId','AllowedAmount','ServiceYear']
       

            
    member_modeling_list = []   
    
    for year in year_list:

        member_per_year = dataset_df.query('ServiceYear == @year')
        member_modeling_df = pd.DataFrame()

        if count_encoding:
             
            for cat_col, cat_name in zip(categorical_cols,categorical_col_names):

                unique_codes = list(standard_encoding_dataset_df[cat_col].values[0])

                codes_df = get_item_frequency_in_counts(member_per_year[cat_col], 
                                                                                 unique_codes, 
                                                                                 prefix=cat_name) 
                member_modeling_df = pd.concat([member_modeling_df,codes_df], axis = 1)
                
        else:
            
            # Encode in presence with bool
            for cat_col, cat_name in zip(categorical_cols,categorical_col_names):

                unique_codes = list(standard_encoding_dataset_df[cat_col].values[0])

                codes_df = get_item_frequency_in_bool(member_per_year[cat_col], 
                                                                             unique_codes, 
                                                                             prefix=cat_name)
                member_modeling_df = pd.concat([member_modeling_df,codes_df], axis = 1)

            

        index_df = member_per_year[key_columns]
        
        index_df = index_df.reset_index(drop=True)
        # Standardize the columns
        member_modeling_df = member_modeling_df[standardized_encoding_columns]
        member_modeling_df = member_modeling_df.reset_index(drop=True)
        
        member_modeling_df = pd.concat([index_df,member_modeling_df], axis=1)
        member_modeling_list.append(member_modeling_df)

    member_modeling_list_df = pd.concat(member_modeling_list, axis = 0)
    
    member_modeling_list_df = member_modeling_list_df.drop_duplicates()
    
    # drop these columns
    member_modeling_list_df = member_modeling_list_df.drop(['TPSMemberID','ServiceYear'], axis = 1)
    
    
    return member_modeling_list_df


def load_data():
    WC = pyodbc.connect('Trusted_Connection=yes', 
                     driver = '{ODBC Driver 17 for SQL Server}',
                     server = 'TPS-PRD-WCARE1')

    sql_member_query = pd.read_sql_query('''
                                SELECT
                                 *
                                FROM WC.dbo.MemberAnalysisMemberSummary12OCT22
                                Where Episode = 1
                                ''' ,WC)
    
    
    sql_claim_query = pd.read_sql_query('''
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
    claim_df = pd.DataFrame(sql_claim_query)
    member_summary_df = pd.DataFrame(sql_member_query)
    return claim_df, member_summary_df



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
    
    cols_to_drop = ['TPSMemberID','ServiceYear','MemberYearId','ReinfectionEpisode','AllowedAmount','EpisodeDuration','Age_Cat']
    cols_to_drop  = cols_to_drop + ['ICDCat_Infection',
                                    'ICDCat_Infections',
                                    'ICDCat_Severe Infection',
                                    'CPTCat_Infection Treatment']

    reinfection_result_df = predictor_dataset_df[['MemberYearId','ReinfectionEpisode','AllowedAmount']]
    
    


    for col in cols_to_drop:
            if col in predictor_dataset_df.columns:
                predictor_dataset_df = predictor_dataset_df.drop(col, axis = 1)

    predictor_df = predictor_dataset_df.copy()
    predictor_df.columns =  [ f'col_{i}' for i in range(predictor_df.shape[1])]
    
    return predictor_df, reinfection_result_df


def prediction_performance(y_actual, y_prediction):

    fpr, tpr, thresholds = roc_curve(y_actual, y_prediction)
    roc_auc = auc(fpr, tpr)
    accuracy = accuracy_score(y_actual, y_prediction)
    recall = recall_score(y_actual, y_prediction)
    precision = precision_score(y_actual, y_prediction)
    f1_value = f1_score(y_actual, y_prediction)

    return accuracy, recall, precision, f1_value, roc_auc


def performance_per_member(prediction_result_dataset, dfu_actual, dfu_prediction):

    prediction_member_dataset_df = prediction_result_dataset.groupby(
        ["MemberYearId"], as_index=True
    ).agg({dfu_actual: lambda x: x.max(), dfu_prediction: lambda x: x.max()})
    prediction_member_dataset_df.columns = ["Score", "Prediction"]
    prediction_member_dataset_df = prediction_member_dataset_df.reset_index()

    return prediction_member_dataset_df


Step += 1
Activity = "Macros Defined"
print(Activity, current_dateTime)
log.loc[len(log.index)] = [
    Name,
    ID,
    Activity,
    datetime.now(),
    Step,
    datetime.now(),
    "Success",
    "Success",
]





def dq_training_dataset(training_dataset_df):
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
                "encoding_dimension": 300,
                'is_count_encoding' : True,
                "standard_dummy_loc": "./utilities/standardized_dummy_data.pkl",
                "logistic_reg_model_loc": "./utilities/LogisticRegression_reinfection_risk_model.pickle",
                "lgbm_classification_model_loc": "./utilities/lgbm_reinfection_risk_model.pickle",
                "xgb_classification_model_loc" : "./utilities/xgb_reinfection_risk_model.pickle",
                "results_loc": "./utilities/RiskScore_ReinfectionRisk_Results.csv",
                "log_table_insert_statement": "INSERT INTO [Log].[dbo].[log] VALUES (?,?,?,?,?,?)",
                "scoring_table_insert_statement": "INSERT INTO [Log].[dbo].[scoring] VALUES (?,?,?,?,?)",
                "error_table_insert_statement": "INSERT INTO [Log].[dbo].[error] VALUES (?,?,?,?,?,?)"
                }





try:
    # Put the parameters here

    
    SUMMARY_COLUMNS = REINFECTION_PARAMS["summary_columns"]
   
    STANDARD_DUMMY_LOCATION = REINFECTION_PARAMS["standard_dummy_loc"]
    LR_MODEL_LOCATION = REINFECTION_PARAMS["logistic_reg_model_loc"]
    LGBM_MODEL_LOCATION = REINFECTION_PARAMS["lgbm_classification_model_loc"]
    XGB_MODEL_LOCATION = REINFECTION_PARAMS["xgb_classification_model_loc"]
    
    IS_COUNT_ENCODING = REINFECTION_PARAMS["is_count_encoding"]
    
    RESULTS_LOCATION = REINFECTION_PARAMS["results_loc"]

    LOG_TABLE_INSERT_STATEMENT = REINFECTION_PARAMS["log_table_insert_statement"]
    SCORING_TABLE_INSERT_STATEMENT = REINFECTION_PARAMS[
        "scoring_table_insert_statement"
    ]
    ERROR_TABLE_INSERT_STATEMENT = REINFECTION_PARAMS["error_table_insert_statement"]

    Step += 1
    Activity = "Config-Parameters Loaded"
    print(Activity, current_dateTime)
    log.loc[len(log.index)] = [
        Name,
        ID,
        Activity,
        datetime.now(),
        Step,
        datetime.now(),
        "Success",
        "Success",
    ]

    ############################### Establish the global connection here
    current_dateTime = datetime.now()
    Activity = "Connection Established"
    Step = Step + 1

    cnxn = pyodbc.connect(
        "Trusted_Connection=yes",
        driver="{ODBC Driver 17 for SQL Server}",
        server="TPS-PRD-DS06",
    )
    print(Activity, current_dateTime)
    log.loc[len(log.index)] = [
        Name,
        ID,
        Activity,
        datetime.now(),
        Step,
        datetime.now(),
        "Success",
        "Success",
    ]

    ##################### Load data
    # We read sample testing dataset for demonstrations purposes
    fidelis_claim_dataset, fidelis_member_summary_dataset = load_data()
    #fidelis_claim_dataset = pd.read_pickle('./utilities/member_analysis_claim_df.pickle')
    #fidelis_member_summary_dataset = pd.read_pickle('./utilities/member_analysis_summary_df.pickle')
    
    PREDICTION_DATA = fidelis_claim_dataset[fidelis_claim_dataset['ServiceYear'].isin([2021])]
    PREDICT_DATA = dq_training_dataset(PREDICTION_DATA)
    
    Step = Step + 1
    current_dateTime = datetime.now()
    Activity = "Data Ingested"
    print(Activity, current_dateTime)
    log.loc[len(log.index)] = [
        Name,
        ID,
        Activity,
        datetime.now(),
        Step,
        datetime.now(),
        "Success",
        "Success",
    ]
    ######################### Transform Data
    Activity = "Data Transformed"
    Step = Step + 1
    
    predictor_df, prediction_result_df = transform_data_into_Reinfection_predictors(PREDICT_DATA, 
                                               fidelis_member_summary_dataset,
                                                summary_columns  = SUMMARY_COLUMNS,
                                               count_encoding = IS_COUNT_ENCODING,
                                               standardized_file_location = STANDARD_DUMMY_LOCATION)
   
    
    print(Activity, current_dateTime)
    log.loc[len(log.index)] = [
        Name,
        ID,
        Activity,
        datetime.now(),
        Step,
        datetime.now(),
        "Success",
        "Success",
    ]
    Activity = "Model Ingestion"
    Step = Step + 1
    
    classifier = joblib.load(XGB_MODEL_LOCATION)

    current_dateTime = datetime.now()
    print(Activity, current_dateTime)
    log.loc[len(log.index)] = [
        Name,
        ID,
        Activity,
        datetime.now(),
        Step,
        datetime.now(),
        "Success",
        "Success",
    ]
    ########################## Model Prediction
    Activity = "Model Prediction"
    Step = Step + 1
   
    prediction_result_df["ReinfectionEpisode_Prediction"] = classifier.predict(predictor_df)
    prediction_result_df["PredictionDate"] = datetime.now()
    prediction_result_df["ID"] = ID
    print(Activity, current_dateTime)
    log.loc[len(log.index)] = [
        Name,
        ID,
        Activity,
        datetime.now(),
        Step,
        datetime.now(),
        "Success",
        "Success",
    ]
    
    ################################ Prediction Output
    # Lets create error
    Activity = "Prediction Output"
    Step = Step + 1
    prediction_result_df.to_csv(RESULTS_LOCATION, index=False)

    current_dateTime = datetime.now()

    print(Activity, current_dateTime)
    log.loc[len(log.index)] = [
        Name,
        ID,
        Activity,
        datetime.now(),
        Step,
        datetime.now(),
        "Success",
        "Success",
    ]
    ############################# Analysis
    Activity = "Model Performance"
    current_dateTime = datetime.now()
    Step = Step + 1
    accuracy, recall, precision, f1_value, roc_auc = prediction_performance(
        prediction_result_df['ReinfectionEpisode'],prediction_result_df['ReinfectionEpisode_Prediction']
    )

    member_level_scoring_df = performance_per_member(
        prediction_result_df,
        dfu_actual="ReinfectionEpisode",
        dfu_prediction="ReinfectionEpisode_Prediction",
    )

    member_level_scoring_df["DateTimeStamp"] = current_dateTime
    member_level_scoring_df["ID"] = ID

    print(Activity, current_dateTime)
    log.loc[len(log.index)] = [
        Name,
        ID,
        Activity,
        f1_value,
        Step,
        datetime.now(),
        "Success",
        "Success",
    ]
     ########################### End
    Activity = "End"
    Step = Step + 1
    current_dateTime = datetime.now()

    print(Activity, current_dateTime)
    log.loc[len(log.index)] = [
        Name,
        ID,
        Activity,
        datetime.now(),
        Step,
        datetime.now(),
        "Success",
        "Success",
    ]
    log = log.drop_duplicates(subset=["Name", "ID", "Type", "Step"])
    # We temporarily elimate the Error/Messages; the Log table does not have those columns
    log_df = log[["Name", "ID", "Type", "Value", "Step", "TimeStamp"]]

    log_cursor = cnxn.cursor()
    log_cursor.fast_executemany = True
    log_cursor.executemany(LOG_TABLE_INSERT_STATEMENT, log_df.values.tolist())
    log_cursor.commit()
    log_cursor.close()

    # Open a new connection, Insert into scoring
    
    member_level_scoring_df = member_level_scoring_df[
        ["ID", "MemberYearId", "Score", "Prediction", "DateTimeStamp"]
    ]
    member_level_scoring_df.columns = ["ID", "TPSMemberID", "Score", "Prediction", "DateTimeStamp"]

    scoring_cursor = cnxn.cursor()
    scoring_cursor.fast_executemany = True
    scoring_cursor.executemany(
        SCORING_TABLE_INSERT_STATEMENT, member_level_scoring_df.values.tolist()
    )
    scoring_cursor.commit()
    scoring_cursor.close()

    print("RiskScore-ReinfectionRisk prediction scores successfully persisted in SCORES table")
    
except Exception as err:
    print(err)
    
finally:
    cnxn.close()
    print(log)


