###########################
# Utilities for getting training feature libraries
# Includes
#1. get_training_feature_library_top_level_category
#2. get_training_feature_library_category
#3. get_training_feature_library_subcategory
#4. get_training_feature_library_placeofservice
#5. get_training_feature_library_cpt3
#6. get_training_feature_library_chronic_disease
#7. get_training_feature_library_totalpaid
#8. get_training_feature_library_HOSP_visits
#9. get_training_feature_library_EM_visits
#10. get_training_feature_library_age_gender
#11. get_training_feature_library_ER_visits
#12. get_training_feature_library_SNF_visits
#13. get_training_feature_library_HomeCare_visits
#14. get_training_feature_library_OfficeVisit
#15. get_training_feature_library_OutPatientHospital
#16. get_training_feature_library_ER_Related
#17. get_training_feature_library_Behavioral_Health


########################################
#Training (SQL): year(hdr.StartDateofService) = year(getdate()-545)
#	and datepart(q,hdr.StartDateofService) = datepart(q,(getdate()-545))

########################################


import pyodbc
import pandas as pd
import numpy as np
import json
import pickle
import argparse
from scipy.sparse import csr_matrix 

from datetime import datetime 

 


import warnings

warnings.filterwarnings("ignore")

############################################################Logging


#current_dateTime = datetime.now()
#print("Start ", current_dateTime)

#Step = Step + 1
#Activity = "Start"
log_df = pd.DataFrame(
    columns=["Name","Type","Step", "Activity", "TimeStamp","Duration" ,"Status", "Msg"]
)



###################### Common utilities ################


def to_1D(series):
 return pd.Series([x for _list in series for x in _list])

def split_dataframe(df, chunk_size = 10000): 
    chunks = list()
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        chunks.append(df[i*chunk_size:(i+1)*chunk_size])
    return chunks


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


def get_categorical_levels(cxcn, categorical_feature):
    '''
    !Dont modify server client and table here
    '''
    
    sql_cat_feature_levels = pd.read_sql_query(f"""
                   select distinct {categorical_feature} from centene.map.comorbidities
                    
                    """,cxcn)
    sql_cat_feature_levels_df = pd.DataFrame(sql_cat_feature_levels)
    
    col = f"normalizing_column"
    
    sql_cat_feature_levels_df.columns = [col]
    
    return sql_cat_feature_levels_df


def generate_member_combinations_with_categorical_levels(cat_encoding_df, member_tuples):
    
    member_df = pd.DataFrame(member_tuples)
    member_df.columns = ['TPSMemberID']
    
    combination_df = pd.merge(
                        member_df.assign(key=1),
                        cat_encoding_df.assign(key=1),
                                        on='key'
                    ).drop('key', axis=1)
    
    return combination_df


def get_unique_memberid_from_table(connection, client_name, table_name):
    
    sql_members = pd.read_sql_query(f"""
                       select distinct TPSMemberID from {client_name}.dbo.{table_name}

                        """,connection)

    sql_members = tuple(list(sql_members['TPSMemberID'].values))
    
    return sql_members


def generate_feature_encodings(dataset_df, combination_df, column_name, prefix_string):

    
    feature_dict = {'CPT3':'CPT3_',
                    'CPT4':'CPT4_',
                    'CPT':'CPT_',
                    'Category':'CAT_',
                    'SubCategory':'SUBCAT_',
                    'Top_Level_Category':'TOP_',
                    'PlaceOfServiceCode':'POS_',
                    'POS':'POS_',
                    'BillType': 'BT_',
                    'PDX3':'PDX3_',
                    'PDX4':'PDX4_',
                    'ChronicDisease':'CHRONIC_'
                   }
    
    
    visits_dict = {'CPT3':'CPT3_Visits',
                    'CPT4':'CPT4_Visits',
                    'CPT':'CPT_Visits',
                    'Category':'DXCAT_Visits',
                    'SubCategory':'DXSUBCAT_Visits',
                    'Top_Level_Category':'DXTOP_Visits',
                    'PlaceOfServiceCode':'POS_Visits',
                    'POS':'POS_Visits',
                   'BillType': 'BillType_Visits',
                   'PDX3': 'PrimaryDx_Visits',
                   'PDX4': 'PrimaryDx_Visits',
                    'ChronicDisease':'CHRONIC_Visits'
                   }
    
    paidamount_dict = {'CPT3':'CPT3_Paid',
                    'CPT4':'CPT4_Paid',
                    'CPT':'CPT_Paid',
                    'Category':'DXCAT_Paid',
                    'SubCategory':'DXSUBCAT_Paid',
                    'Top_Level_Category':'DXTOP_Paid',
                    'PlaceOfServiceCode':'POS_Paid',
                    'POS':'POS_Paid',
                   'BillType': 'BillType_Paid',
                   'PDX3': 'PrimaryDx_Paid',
                   'PDX4': 'PrimaryDx_Paid',
                    'ChronicDisease':'CHRONIC_Paid'
                   }
    
    distinctclaims_dict = {'CPT3':'CPT3_DistinctClaims',
                    'CPT4':'CPT4_DistinctClaims',
                    'CPT':'CPT_DistinctClaims',
                    'Category':'DXCAT_DistinctClaims',
                    'SubCategory':'DXSUBCAT_DistinctClaims',
                    'Top_Level_Category':'DXTOP_DistinctClaims',
                    'PlaceOfServiceCode':'POS_DistinctClaims',
                    'POS':'POS_DistinctClaims',
                   'BillType': 'BillType_DistinctClaims',
                   'PDX3': 'PrimaryDx_DistinctClaims',
                   'PDX4': 'PrimaryDx_DistinctClaims',
                    'ChronicDisease':'CHRONIC_DistinctClaims'
                   }
    
    
    prefix = feature_dict[column_name]
    feature_visits = visits_dict[column_name]
    feature_paid = paidamount_dict[column_name]
    feature_claims = distinctclaims_dict[column_name]
    
    col = "normalizing_column"
    
    dataset_df['TPSMemberID'] = dataset_df['TPSMemberID'].astype(str)
    combination_df['TPSMemberID'] = combination_df['TPSMemberID'].astype(str)
    
    df = combination_df.merge(dataset_df, how='left', 
                         left_on=['TPSMemberID',col], 
                        right_on=['TPSMemberID',column_name])
    # At this point we can set the missing to OTHER
    df[column_name] = df[column_name].fillna('OTHER')
    
    df = df.fillna(0)

    dataset_agg_df = (df.groupby(['TPSMemberID'], as_index=False)
                                                   .agg({column_name: lambda x:x.to_list(),
                                                         feature_visits: lambda x:x.sum(),
                                                     feature_claims: lambda x:x.sum(),
                                                     feature_paid: lambda x:x.sum()
                                                                                    }))

    unique_encoding_code = list(set(list(df[column_name].values)))

    unique_normalizing_code = list(set(list(df[col].values)))



    codes_df = get_item_frequency_in_counts(dataset_agg_df[column_name], unique_encoding_code, prefix='')
    index_df = dataset_agg_df[['TPSMemberID',feature_visits,feature_claims,feature_paid]]

    
    
        
    if 'OTHER' in codes_df.columns:
        
        codes_df = codes_df.drop('OTHER', axis = 1)

    # Get a normalizing 
    dummy_df = pd.DataFrame(columns=unique_normalizing_code)
    codes_dff = pd.concat([codes_df, dummy_df])

    codes_dff = codes_dff[unique_normalizing_code]
    codes_dff = codes_dff.fillna(0)
    codes_dff = codes_dff.astype(int)
    
    # Add the prefix
    codes_dff.columns = [prefix_string + str(col)  for col in codes_dff.columns]

    final_df = pd.concat([index_df,codes_dff], axis = 1)
    
    
    return final_df

#################################### Feature Libraries
########################################1. Top level Categories

def get_top_level_category_sql_dataset(cxcn, client_name, table_name, start_date, end_date):

    
    
    sql_query = pd.read_sql_query(f"""
    
    select
        hdr.TPSMemberID
        ,base.Top_Level_category as Top_Level_Category
        ,sum(hdr.TotalPaidAmount) as DXTOP_Paid
        ,count(distinct(hdr.TPSClaimID)) as DXTOP_DistinctClaims
        ,count(distinct(concat(hdr.tpsmemberid,hdr.startdateofservice))) as DXTOP_Visits
        
    
    from (
        select
            com.Top_Level_Category
            ,dx.DxCode
            ,dx.tpsclaimid
            ,dx.DxSeq
        from CENTENE.map.[Comorbidities] com
        inner join {client_name}.dbo.ClaimDx dx on 
            (dx.DxCode in (select distinct code from CENTENE.map.[Comorbidities])
            and com.code = dx.DxCode)
            ) base
        inner join centene.dbo.Claims hdr on 
        (
        hdr.TPSMemberID in (select  
                            TPSMemberID 
                        from {client_name}.dbo.{table_name})
        and base.tpsclaimid = hdr.TPSClaimID
        and hdr.StartDateofService between '{start_date}' and '{end_date}')
    group by
        hdr.TPSMemberID
        ,base.Top_Level_Category
    ;
   
    """,cxcn)

    
    toplevel_df = pd.DataFrame(sql_query)
    
    return toplevel_df





def get_training_feature_library_top_level_category(server_name, 
                                                    client_name, 
                                           table_name, 
                                            start_date,
                                            end_date
                                           ):
    
    
    
   
    
    connection = pyodbc.connect(
        CONNECTION_STRING
     )
    
    toplevelcategories_df = get_categorical_levels(connection, 'Top_Level_Category')
    
    member_tuples = get_unique_memberid_from_table(connection,client_name, 
                                                   table_name = table_name)
    
    mem_feature_combination_df = generate_member_combinations_with_categorical_levels(toplevelcategories_df, 
                                                                                      member_tuples)
    # Replace this line with the specific column
    toplevel_claim_df = get_top_level_category_sql_dataset(connection, client_name,table_name,start_date,end_date)
    
    top_feature_library_df = generate_feature_encodings(toplevel_claim_df, 
                                                        mem_feature_combination_df, 
                                                        column_name = 'Top_Level_Category',
                                                       prefix_string = 'DXTOP_')
    
    # Save table
    filename = f'{table_name}_Feature_DxTopLevel.pickle'
    #top_feature_library_df.to_pickle(filename)
    #toplevel_claim_df.to_pickle(filename)
    #print(f'Top level Feature library is saved to: {filename}')
    
    return top_feature_library_df





#####################################2. Category

def get_category_sql_dataset(cxcn, client_name, 
                             member_tuple,
                             start_date,
                            end_date):
    
    sql_query = pd.read_sql_query(f"""
    
    select
        hdr.TPSMemberID
        ,base.Category as Category
        ,sum(hdr.TotalPaidAmount) as DXCAT_Paid
        ,count(distinct(hdr.TPSClaimID)) as DXCAT_DistinctClaims
        ,count(distinct(concat(hdr.tpsmemberid,hdr.startdateofservice))) as DXCAT_Visits
        
    
    from (
        select
            com.Category
            ,dx.DxCode
            ,dx.tpsclaimid
            ,dx.DxSeq
        from CENTENE.map.[Comorbidities] com
        inner join {client_name}.dbo.ClaimDx dx on 
            (dx.DxCode in (select distinct code from CENTENE.map.[Comorbidities])
            and com.code = dx.DxCode)
            ) base
        inner join {client_name}.dbo.Claims hdr on 
        (
        hdr.TPSMemberID in {member_tuple}
        and base.tpsclaimid = hdr.TPSClaimID
        and hdr.StartDateofService between '{start_date}' and '{end_date}')
    group by
        hdr.TPSMemberID
        ,base.Category
    ;
   
    """,cxcn)
    
    
    cat_df = pd.DataFrame(sql_query)
    
    return cat_df





def get_training_feature_library_category(server_name, 
                                          client_name,
                                           table_name, 
                                          start_date,
                                            end_date,
                                           iteration_length = 1000,
                                            ):
    '''
    Category library requires iteration; we set the default iteration to 200, you can increase or decrease it
    '''
    
    connection = pyodbc.connect(
        CONNECTION_STRING
     )
    
    categories_df = get_categorical_levels(connection, 'Category')
    member_tuples = get_unique_memberid_from_table(connection, client_name,table_name = table_name)
    
    member_tuple_df = pd.DataFrame(member_tuples)
    member_tuple_df.columns = ['TPSMemberID']

    cat_feature_library_final_df = pd.DataFrame()

    split_size = iteration_length
    df_split = np.array_split(member_tuple_df, split_size)


    for i in range(split_size):
    
    

        member_tuples = tuple(list(df_split[i]['TPSMemberID'].values))

        mem_feature_combination_df = generate_member_combinations_with_categorical_levels(categories_df,member_tuples)


        category_claim_df = get_category_sql_dataset(connection, 
                                                     client_name, 
                                                     member_tuples,
                                                     start_date,
                                                       end_date)

        cat_feature_library_df = generate_feature_encodings(category_claim_df, 
                                                            mem_feature_combination_df, 
                                                            column_name = 'Category',
                                                              prefix_string = 'DXCAT_')
        
        print(f'Category iteration #{i}/{split_size}')
        
        
        
        cat_feature_library_final_df = pd.concat([cat_feature_library_df,cat_feature_library_final_df])
        
    filename = f'{table_name}_Feature_DxCategory.pickle'
    #cat_feature_library_final_df.to_pickle(filename)
    #print(f'Category Feature library is saved to: {filename}')
    
    return cat_feature_library_final_df





##################################################3. SubCategory

def get_subcategory_sql_dataset(cxcn, 
                                client_name, 
                                member_tuple,
                               start_date,
                            end_date):
    '''
    We put in member_tuple because the subcategory codes are 1000+
    so we will iterate through with sample members
    '''
    
    sql_query = pd.read_sql_query(f"""
    
    select
        hdr.TPSMemberID
        ,base.SubCategory as SubCategory
        ,sum(hdr.TotalPaidAmount) as DXSUBCAT_Paid
        ,count(distinct(hdr.TPSClaimID)) as DXSUBCAT_DistinctClaims
        ,count(distinct(concat(hdr.tpsmemberid,hdr.startdateofservice))) as DXSUBCAT_Visits
        
    
    from (
        select
            com.SubCategory
            ,dx.DxCode
            ,dx.tpsclaimid
            ,dx.DxSeq
        from CENTENE.map.[Comorbidities] com
        inner join {client_name}.dbo.ClaimDx dx on 
            (dx.DxCode in (select distinct code from CENTENE.map.[Comorbidities])
            and com.code = dx.DxCode)
            ) base
        inner join {client_name}.dbo.Claims hdr on 
        (
        hdr.TPSMemberID in {member_tuple}
        and base.tpsclaimid = hdr.TPSClaimID
        and hdr.StartDateofService between '{start_date}' and '{end_date}')
        
    group by
        hdr.TPSMemberID
        ,base.SubCategory
    ;
   
    """,cxcn)
    
    
    subcat_df = pd.DataFrame(sql_query)
    
    
    return subcat_df






def get_training_feature_library_subcategory(server_name, 
                                             client_name,
                                           table_name, 
                                             start_date,
                                             end_date,
                                           iteration_length = 20
                                            ):
    '''
    Subcategory library requires iteration; we set the default iteration to 20, you can increase or decrease it
    '''
    
    connection = pyodbc.connect(
        CONNECTION_STRING
     )
    
    categories_df = get_categorical_levels(connection, 'SubCategory')
    member_tuples = get_unique_memberid_from_table(connection, client_name,table_name = table_name)
    
    member_tuple_df = pd.DataFrame(member_tuples)
    member_tuple_df.columns = ['TPSMemberID']

    subcat_feature_library_final_df = pd.DataFrame()

    split_size = iteration_length
    df_split = np.array_split(member_tuple_df, split_size)

    


    for i in range(split_size):
    
    

        member_tuples = tuple(list(df_split[i]['TPSMemberID'].values))

        mem_feature_combination_df = generate_member_combinations_with_categorical_levels(categories_df,member_tuples)


        subcategory_claim_df = get_subcategory_sql_dataset(connection, 
                                                           client_name, 
                                                           member_tuples,
                                                           start_date,
                                             end_date)

        subcat_feature_library_df = generate_feature_encodings(subcategory_claim_df, 
                                                            mem_feature_combination_df, 
                                                            column_name = 'SubCategory',
                                                              prefix_string = 'DXSUBCAT_')
        print(f'subcategory iteration #{i}/{split_size}')
        
        
        subcat_feature_library_final_df = pd.concat([subcat_feature_library_df,subcat_feature_library_final_df])
        
    filename = f'{table_name}_Feature_DxSubCategory.pickle'
    #subcat_feature_library_final_df.to_pickle(filename)
    #print(f'Sub Category Feature library is saved to: {filename}')
        
    return subcat_feature_library_final_df





###############################################4. Place of Service

def get_pos_categorical_levels():
    
    
    pos_codes = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22',
'23','24','25','26','27','28','31','32','33','34','35','41','42','43','49','50','51','52','53','54','55','56','57',
 '58','59','60','61','62','63','65','66','71','72','73','81','82','99']
    
    pos_codes = list(set(pos_codes))
    
    
    sql_pos_feature_levels_df = pd.DataFrame(pos_codes)
    
    col = f"normalizing_column"
    
    sql_pos_feature_levels_df.columns = [col]
    
    
    return sql_pos_feature_levels_df



def get_placeofservice_sql_dataset(cxcn, client_name, table_name, start_date,end_date ):
    
    sql_query = pd.read_sql_query(f"""
    
    select
            hdr.TPSMemberID
            ,hdr.PlaceOfServiceCode as POS
            ,sum(hdr.TotalPaidAmount) as POS_Paid
            ,count(distinct(hdr.TPSClaimID)) as POS_DistinctClaims
            ,count(distinct(concat(hdr.tpsmemberid,hdr.startdateofservice))) as POS_Visits

        from {client_name}.dbo.claims hdr
        where 
            hdr.TPSMemberID in (
                                select  
                            TPSMemberID 
                        from {client_name}.dbo.{table_name})
            and hdr.StartDateofService between '{start_date}' and '{end_date}'
        group by
            hdr.TPSMemberID
            ,hdr.PlaceOfServiceCode
        ;
   
    """,cxcn)
    
    
    pos_dataset_df = pd.DataFrame(sql_query)
    
    
    pos_dataset_df['POS'] = np.where(pd.isnull(pos_dataset_df['POS']),
                                                                pos_dataset_df['POS'],
                                                                pos_dataset_df['POS'].astype('Int64'))

    pos_dataset_df['POS'] = pos_dataset_df['POS'].astype(str)
    pos_dataset_df['POS'] = pos_dataset_df['POS'].fillna('OTHER')
    # replace nan with OTHER
    pos_dataset_df['POS'].replace('nan', 'OTHER', inplace=True)
    
    return pos_dataset_df





def get_training_feature_library_placeofservice(server_name, 
                                                client_name, 
                                           table_name, 
                                                start_date,
                                                end_date
                                            ):
    
    connection = pyodbc.connect(
        CONNECTION_STRING
     )
    
    
    pos_categories_df = get_pos_categorical_levels()
    
    member_tuples = get_unique_memberid_from_table(connection, client_name,table_name = table_name)
    
    
    mem_feature_combination_df = generate_member_combinations_with_categorical_levels(pos_categories_df, 
                                                                                      member_tuples)
    # Replace this line with the table
    pos_claim_df = get_placeofservice_sql_dataset(connection, client_name, table_name, start_date, end_date)
    
    pos_feature_library_df = generate_feature_encodings(pos_claim_df, 
                                                        mem_feature_combination_df, 
                                                        column_name = 'POS',
                                                       prefix_string = 'POS_')
    
    filename = f'{table_name}_Feature_POS.pickle'
    #pos_feature_library_df.to_pickle(filename)
    #print(f'POS Feature library is saved to: {filename}')
    
    return pos_feature_library_df





#######################################################5. CPT3
def get_cpt3_categorical_levels(cxcn):
    
    sql_cpt_feature_levels = pd.read_sql_query("""
                   select distinct HCPCS from bcbsm.dbo.ProcMaster
                    """,cxcn)
    cpt_feature_levels_df = pd.DataFrame(sql_cpt_feature_levels)
    
    cpt_feature_levels_df['HCPCS_CPT3'] = cpt_feature_levels_df['HCPCS'].astype(str).str[:3]
    
    cpt_codes  = set(list(cpt_feature_levels_df['HCPCS_CPT3'].values))
    
    cpt_code_df = pd.DataFrame(cpt_codes)
    
    cpt_code_df.columns = ['normalizing_column']
    
    
    return cpt_code_df



def get_cpt3_sql_dataset(cxcn,client_name, member_tuple,start_date, end_date):
    '''
    We put in member_tuple because the cpt3 codes are 1000+
    so we will iterate through with sample members
    '''
    sql_query = pd.read_sql_query(f"""
    
            select
            hdr.TPSMemberID
            ,SUBSTRING(dtl.CPTHCPCSCode, 1, 3) as CPT3
            ,sum(dtl.paidAmount) as CPT3_Paid
            ,count(distinct(hdr.TPSClaimID)) as CPT3_DistinctClaims
            ,count(distinct(concat(hdr.tpsmemberid,hdr.startdateofservice))) as CPT3_Visits

        from {client_name}.dbo.Claims hdr
        inner join {client_name}.dbo.claimcharges dtl on
            (hdr.TPSClaimID = dtl.TPSClaimID)
        where
            hdr.TPSMemberID in {member_tuple}
            and hdr.StartDateofService between '{start_date}' and '{end_date}'
        group by
            hdr.TPSMemberID
            ,SUBSTRING(dtl.CPTHCPCSCode, 1, 3)
        ;
   
    """,cxcn)
    
    
    cpt3_dataset_df = pd.DataFrame(sql_query)
    
    cpt3_dataset_df['CPT3'] = cpt3_dataset_df['CPT3'].astype(str)
   
    
    return cpt3_dataset_df








def get_training_feature_library_cpt3(server_name, 
                                      client_name,
                                      table_name, 
                                      start_date, 
                                      end_date,
                           iteration_length = 20
                            ):
    
    connection = pyodbc.connect(
        CONNECTION_STRING
     )
    
    cpt3_df = get_cpt3_categorical_levels(connection)
    
    member_tuples = get_unique_memberid_from_table(connection, client_name,table_name = table_name)
    
    member_tuple_df = pd.DataFrame(member_tuples)
    
    member_tuple_df.columns = ['TPSMemberID']

    cpt3_feature_library_final_df = pd.DataFrame()

    split_size = iteration_length
    
    df_split = np.array_split(member_tuple_df, split_size)

    


    for i in range(split_size):
    

        member_tuples = tuple(list(df_split[i]['TPSMemberID'].values))

        mem_feature_combination_df = generate_member_combinations_with_categorical_levels(cpt3_df,member_tuples)

        cpt3_claim_df = get_cpt3_sql_dataset(connection, client_name, member_tuples,start_date, end_date)

        cpt3_feature_library_df = generate_feature_encodings(cpt3_claim_df, 
                                                            mem_feature_combination_df, 
                                                            column_name = 'CPT3',
                                                            prefix_string = 'CPT3_')
        print(f'CPT3 iteration #{i}/{split_size}')
        
        
        
        
        cpt3_feature_library_final_df = pd.concat([cpt3_feature_library_df,cpt3_feature_library_final_df])
        
    filename = f'{table_name}_Feature_CPT3.pickle'
    #cpt3_feature_library_final_df.to_pickle(filename)
    #print(f'CPT3 Feature library is saved to: {filename}')
        
    return cpt3_feature_library_final_df






###############################################6. Chronic

def get_chronic_categorical_levels():
    
    chronic_codes = ['Asthma',
     'Rheumatoid Arthritis',
     'Alzheimers Disease',
     'Diabetes Mellitus Type 2',
     'HIV/AIDS',
     'Chronic Hepatitis',
     'Chronic Obstructive Pulmonary Disease (COPD)',
     'Chronic Kidney Disease (CKD)',
     'Depression',
     'Hypertension (High Blood Pressure)',
     'Coronary Artery Disease (CAD)',
     'Osteoarthritis',
     'Heart Failure']
    
    chronic_codes = list(set(chronic_codes))
    
    
    chronic_feature_levels_df = pd.DataFrame(chronic_codes)
    
    col = f"normalizing_column"
    
    chronic_feature_levels_df.columns = [col]
    
    return chronic_feature_levels_df


def get_chronic_sql_dataset(cxcn, client_name, table_name,start_date,end_date):
    
    sql_query = pd.read_sql_query(f"""
    
                select
                base.TPSMemberID
                ,base.ChronicDisease
                ,sum(base.PaidAmount) as CHRONIC_Paid
                ,count(distinct(base.TPSClaimID)) as CHRONIC_DistinctClaims
                ,count(distinct(concat(base.tpsmemberid,base.startdateofservice))) as CHRONIC_Visits
            from 
            (
                select
               
                    hdr.TPSMemberID
                    ,'Diabetes Mellitus Type 2' as ChronicDisease
                    ,dx.DxCode
                    ,dx.tpsclaimid
                    ,hdr.StartDateofService
                    ,dx.DxSeq
                    ,dtl.PaidAmount
                from {client_name}.dbo.ClaimDx dx 
                inner join {client_name}.dbo.ClaimCharges dtl on 
                        (dx.TPSClaimID = dtl.TPSClaimID and dx.DxSeq = dtl.ChargeSeq)
                inner join {client_name}.dbo.claims hdr on 
                        (hdr.TPSClaimID = dtl.TPSClaimID 
                        and hdr.TPSMemberID in (
                                        select  distinct
                                    TPSMemberID 
                                from {client_name}.dbo.{table_name})
                                    and hdr.StartDateofService between '{start_date}' and '{end_date}'
                        )
                where dx.DxCode like 'E11%'
                --;
                union all
                select
                --top (1000)
                    hdr.TPSMemberID
                    ,'Hypertension (High Blood Pressure)' as ChronicDisease
                    ,dx.DxCode
                    ,dx.tpsclaimid
                    ,hdr.startdateofservice
                    ,dx.DxSeq
                    ,dtl.PaidAmount
                from {client_name}.dbo.ClaimDx dx 
                inner join {client_name}.dbo.ClaimCharges dtl on 
                        (dx.TPSClaimID = dtl.TPSClaimID and dx.DxSeq = dtl.ChargeSeq)
                inner join {client_name}.dbo.claims hdr on 
                        (hdr.TPSClaimID = dtl.TPSClaimID 
                        and hdr.TPSMemberID in (
                                         select  distinct
                                    TPSMemberID 
                                from {client_name}.dbo.{table_name})
                                    and hdr.StartDateofService between '{start_date}' and '{end_date}'
                        )
                where dx.DxCode like 'I10%'
                --;
                union all
                select
                --top (1000)
                    hdr.TPSMemberID
                    ,'Chronic Obstructive Pulmonary Disease (COPD)' as ChronicDisease
                    ,dx.DxCode
                    ,dx.tpsclaimid
                    ,hdr.startdateofservice
                    ,dx.DxSeq
                    ,dtl.PaidAmount
                from {client_name}.dbo.ClaimDx dx 
                inner join {client_name}.dbo.ClaimCharges dtl on 
                        (dx.TPSClaimID = dtl.TPSClaimID and dx.DxSeq = dtl.ChargeSeq)
                inner join {client_name}.dbo.claims hdr on 
                        (hdr.TPSClaimID = dtl.TPSClaimID 
                        and hdr.TPSMemberID in (
                                         select  distinct
                                    TPSMemberID 
                                from {client_name}.dbo.{table_name})
                                    
                                    and hdr.StartDateofService between '{start_date}' and '{end_date}'
                        )
                where dx.DxCode like 'J44%'
                --;
                union all
                select
                --top (1000)
                    hdr.TPSMemberID
                    ,'Asthma' as ChronicDisease
                    ,dx.DxCode
                    ,dx.tpsclaimid
                    ,hdr.startdateofservice
                    ,dx.DxSeq
                    ,dtl.PaidAmount
                from {client_name}.dbo.ClaimDx dx 
                inner join {client_name}.dbo.ClaimCharges dtl on 
                        (dx.TPSClaimID = dtl.TPSClaimID and dx.DxSeq = dtl.ChargeSeq)
                inner join {client_name}.dbo.claims hdr on 
                        (hdr.TPSClaimID = dtl.TPSClaimID 
                        and hdr.TPSMemberID in (
                                         select  distinct
                                    TPSMemberID 
                                from {client_name}.dbo.{table_name})
                                    
                                    and hdr.StartDateofService between '{start_date}' and '{end_date}'
                        )
                where dx.DxCode like 'J45%'
                --;
                union all
                select
                --top (1000)
                    hdr.TPSMemberID
                    ,'Coronary Artery Disease (CAD)' as ChronicDisease
                    ,dx.DxCode
                    ,dx.tpsclaimid
                    ,hdr.startdateofservice
                    ,dx.DxSeq
                    ,dtl.PaidAmount
                from {client_name}.dbo.ClaimDx dx 
                inner join {client_name}.dbo.ClaimCharges dtl on 
                        (dx.TPSClaimID = dtl.TPSClaimID and dx.DxSeq = dtl.ChargeSeq)
                inner join {client_name}.dbo.claims hdr on 
                        (hdr.TPSClaimID = dtl.TPSClaimID 
                        and hdr.TPSMemberID in (
                                         select  distinct
                                    TPSMemberID 
                                from {client_name}.dbo.{table_name})
                                    
                                    and hdr.StartDateofService between '{start_date}' and '{end_date}'
                        )
                where dx.DxCode like 'I25%'
                --;
                union all
                select
                --top (1000)
                    hdr.TPSMemberID
                    ,'Heart Failure' as ChronicDisease
                    ,dx.DxCode
                    ,dx.tpsclaimid
                    ,hdr.startdateofservice
                    ,dx.DxSeq
                    ,dtl.PaidAmount
                from {client_name}.dbo.ClaimDx dx 
                inner join {client_name}.dbo.ClaimCharges dtl on 
                        (dx.TPSClaimID = dtl.TPSClaimID and dx.DxSeq = dtl.ChargeSeq)
                inner join {client_name}.dbo.claims hdr on 
                        (hdr.TPSClaimID = dtl.TPSClaimID 
                        and hdr.TPSMemberID in (
                                         select  distinct
                                    TPSMemberID 
                                from {client_name}.dbo.{table_name})
                                    
                                    and hdr.StartDateofService between '{start_date}' and '{end_date}'
                        )
                where dx.DxCode like 'I50%'
                --;
                union all
                select
                --top (1000)
                    hdr.TPSMemberID
                    ,'Chronic Kidney Disease (CKD)' as ChronicDisease
                    ,dx.DxCode
                    ,dx.tpsclaimid
                    ,hdr.startdateofservice
                    ,dx.DxSeq
                    ,dtl.PaidAmount
                from {client_name}.dbo.ClaimDx dx 
                inner join {client_name}.dbo.ClaimCharges dtl on 
                        (dx.TPSClaimID = dtl.TPSClaimID and dx.DxSeq = dtl.ChargeSeq)
                inner join {client_name}.dbo.claims hdr on 
                        (hdr.TPSClaimID = dtl.TPSClaimID 
                        and hdr.TPSMemberID in (
                                        select  distinct
                                    TPSMemberID 
                                from {client_name}.dbo.{table_name})
                                    
                                    and hdr.StartDateofService between '{start_date}' and '{end_date}'
                        )
                where dx.DxCode like 'N18%'
                --;
                union all
                select
                --top (1000)
                    hdr.TPSMemberID
                    ,'Rheumatoid Arthritis' as ChronicDisease
                    ,dx.DxCode
                    ,dx.tpsclaimid
                    ,hdr.startdateofservice
                    ,dx.DxSeq
                    ,dtl.PaidAmount
                from {client_name}.dbo.ClaimDx dx 
                inner join {client_name}.dbo.ClaimCharges dtl on 
                        (dx.TPSClaimID = dtl.TPSClaimID and dx.DxSeq = dtl.ChargeSeq)
                inner join {client_name}.dbo.claims hdr on 
                        (hdr.TPSClaimID = dtl.TPSClaimID 
                        and hdr.TPSMemberID in (
                                        select  distinct
                                    TPSMemberID 
                                from {client_name}.dbo.{table_name})
                                    
                                    and hdr.StartDateofService between '{start_date}' and '{end_date}'
                        )
                where dx.DxCode like 'M05%'
                --;
                union all
                select
                --top (1000)
                    hdr.TPSMemberID
                    ,'Osteoarthritis' as ChronicDisease
                    ,dx.DxCode
                    ,dx.tpsclaimid
                    ,hdr.startdateofservice
                    ,dx.DxSeq
                    ,dtl.PaidAmount
                from {client_name}.dbo.ClaimDx dx 
                inner join {client_name}.dbo.ClaimCharges dtl on 
                        (dx.TPSClaimID = dtl.TPSClaimID and dx.DxSeq = dtl.ChargeSeq)
                inner join {client_name}.dbo.claims hdr on 
                        (hdr.TPSClaimID = dtl.TPSClaimID 
                        and hdr.TPSMemberID in (
                                        select  distinct
                                    TPSMemberID 
                                from {client_name}.dbo.{table_name})
                                    
                                    and hdr.StartDateofService between '{start_date}' and '{end_date}'
                        )
                where dx.DxCode like 'M15%'
                --;
                union all
                select
                --top (1000)
                    hdr.TPSMemberID
                    ,'Chronic Hepatitis' as ChronicDisease
                    ,dx.DxCode
                    ,dx.tpsclaimid
                    ,hdr.startdateofservice
                    ,dx.DxSeq
                    ,dtl.PaidAmount
                from {client_name}.dbo.ClaimDx dx 
                inner join {client_name}.dbo.ClaimCharges dtl on 
                        (dx.TPSClaimID = dtl.TPSClaimID and dx.DxSeq = dtl.ChargeSeq)
                inner join {client_name}.dbo.claims hdr on 
                        (hdr.TPSClaimID = dtl.TPSClaimID 
                        and hdr.TPSMemberID in (
                                        select  distinct
                                    TPSMemberID 
                                from {client_name}.dbo.{table_name})
                                    
                                    and hdr.StartDateofService between '{start_date}' and '{end_date}'
                        )
                where dx.DxCode like 'B18%'
                --;
                union all
                select
                --top (1000)
                    hdr.TPSMemberID
                    ,'HIV/AIDS' as ChronicDisease
                    ,dx.DxCode
                    ,dx.tpsclaimid
                    ,hdr.startdateofservice
                    ,dx.DxSeq
                    ,dtl.PaidAmount
                from {client_name}.dbo.ClaimDx dx 
                inner join {client_name}.dbo.ClaimCharges dtl on 
                        (dx.TPSClaimID = dtl.TPSClaimID and dx.DxSeq = dtl.ChargeSeq)
                inner join {client_name}.dbo.claims hdr on 
                        (hdr.TPSClaimID = dtl.TPSClaimID 
                        and hdr.TPSMemberID in (
                                        select  distinct
                                    TPSMemberID 
                                from {client_name}.dbo.{table_name})
                                    
                                    and hdr.StartDateofService between '{start_date}' and '{end_date}'
                        )
                where dx.DxCode like 'B20%'
                or dx.DxCode like 'B21%' 
                or dx.DxCode like 'B22%' 
                or dx.DxCode like 'B23%'
                or dx.DxCode like 'B24%'
                --;
                union all
                select
                --top (1000)
                    hdr.TPSMemberID
                    ,'Alzheimers Disease' as ChronicDisease
                    ,dx.DxCode
                    ,dx.tpsclaimid
                    ,hdr.startdateofservice
                    ,dx.DxSeq
                    ,dtl.PaidAmount
                from {client_name}.dbo.ClaimDx dx 
                inner join {client_name}.dbo.ClaimCharges dtl on 
                        (dx.TPSClaimID = dtl.TPSClaimID and dx.DxSeq = dtl.ChargeSeq)
                inner join {client_name}.dbo.claims hdr on 
                        (hdr.TPSClaimID = dtl.TPSClaimID 
                        and hdr.TPSMemberID in (
                                        select  distinct
                                    TPSMemberID 
                                from {client_name}.dbo.{table_name})
                                    
                                    and hdr.StartDateofService between '{start_date}' and '{end_date}'
                        )
                where dx.DxCode like 'G30%' 
                --;
                union all
                select
                --top (1000)
                    hdr.TPSMemberID
                    ,'Depression' as ChronicDisease
                    ,dx.DxCode
                    ,dx.tpsclaimid
                    ,hdr.startdateofservice
                    ,dx.DxSeq
                    ,dtl.PaidAmount
                from {client_name}.dbo.ClaimDx dx 
                inner join {client_name}.dbo.ClaimCharges dtl on 
                        (dx.TPSClaimID = dtl.TPSClaimID and dx.DxSeq = dtl.ChargeSeq)
                inner join {client_name}.dbo.claims hdr on 
                        (hdr.TPSClaimID = dtl.TPSClaimID 
                        and hdr.TPSMemberID in (
                                        select  distinct
                                    TPSMemberID 
                                from {client_name}.dbo.{table_name})
                                    
                                    and hdr.StartDateofService between '{start_date}' and '{end_date}'
                        )
                where dx.DxCode like 'F32%'

            ) base
            group by 
                base.TPSMemberID
                ,base.ChronicDisease
                ;

                """,cxcn)
    chronic_dataset_df = pd.DataFrame(sql_query)
    
    return chronic_dataset_df





def get_training_feature_library_chronic_disease(server_name, 
                                                 client_name, 
                                                table_name, 
                                                 start_date,
                                                 end_date
                                            ):
    
    connection = pyodbc.connect(
        CONNECTION_STRING
     )
    
    chronic_categories_df = get_chronic_categorical_levels()
    
    member_tuples = get_unique_memberid_from_table(connection, client_name,table_name = table_name)
    
    
    mem_feature_combination_df = generate_member_combinations_with_categorical_levels(chronic_categories_df, 
                                                                                      member_tuples)
    # Replace this line with the table
    chronic_claim_df = get_chronic_sql_dataset(connection, 
                                               client_name, 
                                               table_name,
                                               start_date,
                                                 end_date)
    
    chronic_feature_library_df = generate_feature_encodings(chronic_claim_df, 
                                                        mem_feature_combination_df, 
                                                        column_name = 'ChronicDisease',
                                                           prefix_string = 'CHRONIC_DISEASE_')
    
    filename = f'{table_name}_Feature_ChronicDisease.pickle'
    #chronic_feature_library_df.to_pickle(filename)
    #print(f'Chronic Disease Feature library is saved to: {filename}')
    
    return chronic_feature_library_df






#######################################7. TotalPaid

def get_training_feature_library_totalpaid(server_name, 
                                           client_name, 
                                           table_name,
                                           start_date,
                                            end_date ):
    '''
    Total paid
                                    
    '''
    
    connection = pyodbc.connect(
        CONNECTION_STRING
     )
    
    sql_query = pd.read_sql_query(f"""
                select
                hdr.TPSMemberID
                ,sum(hdr.TotalPaidAmount) as TotalPaid
                ,count(distinct(hdr.TPSClaimID)) as TotalDistinctClaims
            from {client_name}.dbo.claims hdr
            where 
                hdr.TPSMemberID in (
                                    select distinct
                                    TPSMemberID 
                                from {client_name}.dbo.{table_name})
                
                and hdr.StartDateofService between '{start_date}' and '{end_date}'
            group by
                hdr.TPSMemberID
            ;   
   
            """,connection)
    
    
    totalpaid_dataset_df = pd.DataFrame(sql_query)
    
    total_agg_members_df = totalpaid_dataset_df.groupby(['TPSMemberID'], as_index=False).agg({
                                                                           'TotalPaid': lambda x:x.sum(),
                                                                           'TotalDistinctClaims': lambda x:x.sum(),
                                                                            })
    total_agg_members_df = total_agg_members_df.rename(columns = {'TotalPaid':'TOTAL_TotalPaid',
                                                                 'TotalDistinctClaims':'TOTAL_TotalDistinctClaims'})
    filename = f'{table_name}_Feature_TotalPaid.pickle'
    
    
    return total_agg_members_df






############################################8. HOSP Visit

def get_training_feature_library_HOSP_visits(server_name, client_name, table_name,start_date, end_date):
    '''
    Hospital Visits
                                    
    '''
    connection = pyodbc.connect(
        CONNECTION_STRING
     )
    
    sql_all_members = pd.read_sql_query(f""" 
                                      select distinct
                                    TPSMemberID 
                                    from {client_name}.dbo.{table_name}
                                      """,connection)
    
    sql_query = pd.read_sql_query(f"""
                select distinct
                hdr.TPSMemberID
                ,1 as HOSP
                ,sum(dtl.PaidAmount) as HOSP_Paid
                ,count(distinct(hdr.TPSClaimID)) as HOSP_DistinctClaims
                ,count(distinct(concat(hdr.tpsmemberid,hdr.startdateofservice))) as HOSP_Visits

            from {client_name}.dbo.Claims hdr 
            inner join {client_name}.dbo.ClaimCharges dtl on
                (dtl.CPTHCPCSCode in 
                    (
                    '99231', '99232', '99233', '99234', '99235', '99236', '99237', '99238', '99239',
                    '99221', '99222', '99223'
                    )
                and hdr.TPSClaimID = dtl.TPSClaimID)
            where 
                hdr.TPSMemberID in (
                                    select distinct
                                    TPSMemberID 
                                from {client_name}.dbo.{table_name})
                --and year(hdr.StartDateofService) = year(getdate()-545)
                --and datepart(q,hdr.StartDateofService) = datepart(q,(getdate()-545))
                and hdr.StartDateofService between '{start_date}' and '{end_date}'
            group by
                Hdr.TPSMemberID
            ;
   
            """,connection)
    
    
    
    all_members_df = pd.DataFrame(sql_all_members)
    
    hospvisit_dataset_df = pd.DataFrame(sql_query)
    
    hospvisit_agg_members_df = hospvisit_dataset_df.groupby(['TPSMemberID'], as_index=False).agg({
                                                                           'HOSP':  lambda x:x.max(),
                                                                           'HOSP_Paid': lambda x:x.sum(),
                                                                           'HOSP_Visits': lambda x:x.sum(),
                                                                           'HOSP_DistinctClaims': lambda x:x.sum(),
                                                                            })
    
    # Do a left join
    hospvisit_agg_all_members_df = all_members_df.merge(hospvisit_agg_members_df,
                                                       how='left',
                                                       on='TPSMemberID')
    hospvisit_agg_all_members_df = hospvisit_agg_all_members_df.fillna(0)
    hospvisit_agg_all_members_df['HOSP'] = hospvisit_agg_all_members_df['HOSP'].astype(int)
    
    filename = f'{table_name}_Feature_HOSP.pickle'
    #hospvisit_agg_all_members_df.to_pickle(filename)
    #print(f'Hospital Visit Feature library is saved to: {filename}')
    
    
    return hospvisit_agg_all_members_df




############################################9. EM 

def get_training_feature_library_EM_visits(server_name, client_name, table_name, start_date, end_date ):
    '''
    EM Visits
                                    
    '''
    connection = pyodbc.connect(
        CONNECTION_STRING
     )
    
    sql_all_members = pd.read_sql_query(f""" 
                                      select distinct
                                    TPSMemberID 
                                    from {client_name}.dbo.{table_name}
                                      """,connection)
    
    sql_query = pd.read_sql_query(f"""
                            select
                            hdr.TPSMemberID

                            ,sum(dtl.PaidAmount) as EM_Paid
                            ,count(distinct(hdr.TPSClaimID)) as EM_DistinctClaims
                            ,count(distinct(concat(hdr.tpsmemberid,hdr.startdateofservice))) as EM_Visits

                        --INTO {client_name}.dbo.DS_Training_Feature_EM
                        from {client_name}.dbo.claimcharges dtl
                        inner join {client_name}.dbo.claims hdr on
                            (hdr.TPSClaimID = dtl.TPSClaimID
                            and dtl.CPTHCPCSCode like '992%')
                        where 
                            hdr.TPSMemberID in (
                                            select distinct
                                            TPSMemberID 
                                        from {client_name}.dbo.{table_name})
                            --and year(hdr.StartDateofService) = year(getdate()-545)
                            --and datepart(q,hdr.StartDateofService) = datepart(q,(getdate()-545))
                            and hdr.StartDateofService between '{start_date}' and '{end_date}'
                        group by
                            hdr.TPSMemberID
                        ;
   
            """,connection)
    
    
    
    all_members_df = pd.DataFrame(sql_all_members)
    
    emvisit_dataset_df = pd.DataFrame(sql_query)
    
    emvisit_agg_members_df = emvisit_dataset_df.groupby(['TPSMemberID'], as_index=False).agg({
                                                                           'EM_Paid': lambda x:x.sum(),
                                                                           'EM_Visits': lambda x:x.sum(),
                                                                           'EM_DistinctClaims': lambda x:x.sum(),
                                                                            })
    
    # Do a left join
    em_agg_all_members_df = all_members_df.merge(emvisit_agg_members_df,
                                                       how='left',
                                                       on='TPSMemberID')
    em_agg_all_members_df = em_agg_all_members_df.fillna(0)
    
    filename = f'{table_name}_Feature_EM.pickle'
    #em_agg_all_members_df.to_pickle(filename)
    #print(f'EM Visit Feature library is saved to: {filename}')
    
    
    
    return em_agg_all_members_df







############################################10. AGE_Gender

def get_training_feature_library_mbr_demographics(server_name, client_name, table_name, start_date, end_date ):
    '''
    Age/Gender Visits
                                    
    '''
    connection = pyodbc.connect(
        CONNECTION_STRING
     )
    
    sql_query = pd.read_sql_query(f"""
                            select
                            mbr.TPSMemberID
                            ,datediff(dd,mbr.DateofBirth,getdate())/365.25 as Age
                            ,mbr.Ethnicity
                            ,mbr.Gender
                        
                        From {client_name}.dbo.members mbr 
                        where
                            mbr.TPSMemberID in (
                                                select distinct
                                            TPSMemberID 
                                        from {client_name}.dbo.{table_name})
                                    ;

                        """,connection)
    
    
    
   
    
    age_gender_dataset_df = pd.DataFrame(sql_query)
    
    mean_value = age_gender_dataset_df['Age'].mean()
    age_gender_dataset_df['Age'].fillna(value=mean_value, inplace=True)
    age_gender_dataset_df['Gender'] = age_gender_dataset_df['Gender'].fillna('M')
    age_gender_dataset_df['Gender'] = age_gender_dataset_df['Gender'].replace(to_replace='', value='M')
    
    age_gender_agg_members_df = age_gender_dataset_df.groupby(['TPSMemberID'], as_index=False).agg({
                                                                           'Age': lambda x:x.mean(),
                                                                           'Gender': lambda x:x.to_list()[0],
                                                                           'Ethnicity': lambda x:x.to_list()[0]
                                                                            })
   
    age_gender_agg_members_df['Gender'] = age_gender_agg_members_df['Gender'].apply({'F':0, 'M':1}.get)

    age_gender_agg_members_df = age_gender_agg_members_df.rename(columns = {'Age':'DEMO_Age',
                                                                   'Gender':'DEMO_Gender',
                                                                   'Ethnicity':'DEMO_Ethnicity'})
    
    filename = f'{table_name}_Feature_MbrDemographics.pickle'
   

    
    
    return age_gender_agg_members_df







############################################11. ER Visits

def get_training_feature_library_ER_visits(server_name, client_name,table_name, start_date, end_date):
    '''
    ER Visits
                                    
    '''
    
    connection = pyodbc.connect(
        CONNECTION_STRING
     )
    
    sql_all_members = pd.read_sql_query(f""" 
                                      select distinct
                                    TPSMemberID 
                                    from {client_name}.dbo.{table_name}
                                      """,connection)
    
    sql_query = pd.read_sql_query(f"""
                select distinct
                    hdr.TPSMemberID
                    ,1 as ER
                    ,sum(dtl.PaidAmount) as ER_Paid
                    ,count(distinct(hdr.TPSClaimID)) as ER_DistinctClaims
                    ,count(distinct(concat(hdr.tpsmemberid,hdr.startdateofservice))) as ER_Visits
                
                from {client_name}.dbo.Claims hdr with (nolock)
                inner join {client_name}.dbo.ClaimCharges dtl  with (nolock) on
                    (
                    (dtl.RevenueCode IN ('450','451','452','456','459')
                    or dtl.CPTHCPCSCode IN ('99281','99282','99283','99284','99285'))
                    and hdr.TPSClaimID = dtl.TPSClaimID)
                where 
                    hdr.TPSMemberID in (select distinct
                                    TPSMemberID 
                                    from {client_name}.dbo.{table_name}
                                        )
                    --and year(hdr.StartDateofService) = year(getdate()-545)
                    --and datepart(q,hdr.StartDateofService) = datepart(q,(getdate()-545))
                    and hdr.StartDateofService between '{start_date}' and '{end_date}'
                    and (SUBSTRING(CONVERT(varchar,hdr.PlaceOfServiceCode),1,2) IN ('23') 
                                            or TRY_CONVERT(INT,hdr.AdmissionType) = 1
                        )
                group by
                    Hdr.TPSMemberID
                ;
   
            """,connection)
    
    
    
    all_members_df = pd.DataFrame(sql_all_members)
    
    ervisit_dataset_df = pd.DataFrame(sql_query)
    
    ervisit_agg_members_df = ervisit_dataset_df.groupby(['TPSMemberID'], as_index=False).agg({
                                                                           'ER':  lambda x:x.max(),
                                                                           'ER_Paid': lambda x:x.sum(),
                                                                           'ER_Visits': lambda x:x.sum(),
                                                                           'ER_DistinctClaims': lambda x:x.sum(),
                                                                            })
    
    # Do a left join
    ervisit_agg_all_members_df = all_members_df.merge(ervisit_agg_members_df,
                                                       how='left',
                                                       on='TPSMemberID')
    ervisit_agg_all_members_df = ervisit_agg_all_members_df.fillna(0)
    ervisit_agg_all_members_df['ER'] = ervisit_agg_all_members_df['ER'].astype(int)
    
    filename = f'{table_name}_Feature_ER.pickle'
    #ervisit_agg_all_members_df.to_pickle(filename)
    #print(f'ER Feature library is saved to: {filename}')
    
    
    return ervisit_agg_all_members_df






############################################12. SNF Visits

def get_training_feature_library_SNF_visits(server_name, client_name,table_name, start_date, end_date):
    '''
    SNF Visits
                                    
    '''
    connection = pyodbc.connect(
        CONNECTION_STRING
     )
    
    sql_all_members = pd.read_sql_query(f""" 
                                      select distinct
                                    TPSMemberID 
                                    from {client_name}.dbo.{table_name}
                                      """,connection)
    
    sql_query = pd.read_sql_query(f"""
                select distinct
                hdr.TPSMemberID
                ,1 as SNF
                ,sum(dtl.PaidAmount) as SNF_Paid
                ,count(distinct(hdr.TPSClaimID)) as SNF_DistinctClaims
                ,count(distinct(concat(hdr.tpsmemberid,hdr.startdateofservice))) as SNF_Visits
            from {client_name}.dbo.Claims hdr with (nolock)
            left join {client_name}.dbo.ClaimCharges dtl  with (nolock) on
                (hdr.TPSClaimID = dtl.TPSClaimID)
            where 
                hdr.TPSMemberID in (select distinct
                                    TPSMemberID 
                                    from {client_name}.dbo.{table_name}
                                   )
                --and year(hdr.StartDateofService) = year(getdate()-545)
                --and datepart(q,hdr.StartDateofService) = datepart(q,(getdate()-545))
                and hdr.StartDateofService between '{start_date}' and '{end_date}'
                and (hdr.PlaceOfServiceCode = 31 OR left(hdr.TypeofBill, 1) ='2')
            group by
                Hdr.TPSMemberID
            ;
   
            """,connection)
    
    
    
    all_members_df = pd.DataFrame(sql_all_members)
    
    snf_dataset_df = pd.DataFrame(sql_query)
    
    snf_agg_members_df = snf_dataset_df.groupby(['TPSMemberID'], as_index=False).agg({
                                                                           'SNF':  lambda x:x.max(),
                                                                           'SNF_Paid': lambda x:x.sum(),
                                                                           'SNF_Visits': lambda x:x.sum(),
                                                                           'SNF_DistinctClaims': lambda x:x.sum(),
                                                                            })
    
    # Do a left join
    snf_agg_all_members_df = all_members_df.merge(snf_agg_members_df,
                                                       how='left',
                                                       on='TPSMemberID')
    snf_agg_all_members_df = snf_agg_all_members_df.fillna(0)
    snf_agg_all_members_df['SNF'] = snf_agg_all_members_df['SNF'].astype(int)
    
    filename = f'{table_name}_Feature_SNF.pickle'
    #snf_agg_all_members_df.to_pickle(filename)
    #print(f'SNF Feature library is saved to: {filename}')
    
    
    return snf_agg_all_members_df






def get_training_feature_library_HomeCare_visits(server_name, client_name,table_name, start_date, end_date):
    '''
    Homecare
                                    
    '''
    
    connection = pyodbc.connect(
        CONNECTION_STRING
     )
    
    sql_all_members = pd.read_sql_query(f""" 
                                      select distinct
                                    TPSMemberID 
                                    from {client_name}.dbo.{table_name}
                                      """,connection)
    
    sql_query = pd.read_sql_query(f"""
                select distinct
                hdr.TPSMemberID
                ,1 as HomeCare
                ,sum(dtl.PaidAmount) as HomeCare_Paid
                ,count(distinct(hdr.TPSClaimID)) as HomeCare_DistinctClaims
                ,count(distinct(concat(hdr.tpsmemberid,hdr.startdateofservice))) as HomeCare_Visits
            
            from {client_name}.dbo.Claims hdr with (nolock)
            left join {client_name}.dbo.ClaimCharges dtl  with (nolock) on
                (hdr.TPSClaimID = dtl.TPSClaimID)
            where 
                hdr.TPSMemberID in (
                                    select distinct
                                    TPSMemberID 
                                    from {client_name}.dbo.{table_name})
                and (hdr.PlaceOfServiceCode = 12 OR left(hdr.TypeofBill, 2) ='32')
                --and year(hdr.StartDateofService) = year(getdate()-545)
                --and datepart(q,hdr.StartDateofService) = datepart(q,(getdate()-545))
                and hdr.StartDateofService between '{start_date}' and '{end_date}'
            group by
                Hdr.TPSMemberID
            ;
   
            """,connection)
    
    
    
    all_members_df = pd.DataFrame(sql_all_members)
    
    homecare_dataset_df = pd.DataFrame(sql_query)
    
    homecare_agg_members_df = homecare_dataset_df.groupby(['TPSMemberID'], as_index=False).agg({
                                                                           'HomeCare':  lambda x:x.max(),
                                                                           'HomeCare_Paid': lambda x:x.sum(),
                                                                           'HomeCare_Visits': lambda x:x.sum(),
                                                                           'HomeCare_DistinctClaims': lambda x:x.sum(),
                                                                            })
    
    # Do a left join
    homecare_agg_all_members_df = all_members_df.merge(homecare_agg_members_df,
                                                       how='left',
                                                       on='TPSMemberID')
    homecare_agg_all_members_df = homecare_agg_all_members_df.fillna(0)
    homecare_agg_all_members_df['HomeCare'] = homecare_agg_all_members_df['HomeCare'].astype(int)
    
    filename = f'{table_name}_Feature_HomeCare.pickle'
    #homecare_agg_all_members_df.to_pickle(filename)
    #print(f'Home Care Feature library is saved to: {filename}')
    
    
    return homecare_agg_all_members_df





############################################14. Office Visits

def get_training_feature_library_OfficeVisit(server_name, client_name,table_name, start_date, end_date):
    '''
    OfficeVisit
                                    
    '''
    connection = pyodbc.connect(
        CONNECTION_STRING
     )
    
    sql_all_members = pd.read_sql_query(f""" 
                                      select distinct
                                    TPSMemberID 
                                    from {client_name}.dbo.{table_name}
                                      """,connection)
    
    sql_query = pd.read_sql_query(f"""
                select distinct
            hdr.TPSMemberID
            ,1 as OfficeVisit

            ,sum(dtl.PaidAmount) as OfficeVisit_Paid
            ,count(distinct(hdr.TPSClaimID)) as OfficeVisit_DistinctClaims
            ,count(distinct(concat(hdr.tpsmemberid,hdr.startdateofservice))) as OfficeVisit_Visits
        from {client_name}.dbo.Claims hdr with (nolock)
        left join {client_name}.dbo.ClaimCharges dtl  with (nolock) on
            (hdr.TPSClaimID = dtl.TPSClaimID)
        where 
            hdr.TPSMemberID in (
                                select distinct
                                    TPSMemberID 
                                    from {client_name}.dbo.{table_name})
            and hdr.PlaceOfServiceCode = 11
            --and year(hdr.StartDateofService) = year(getdate()-545)
            --and datepart(q,hdr.StartDateofService) = datepart(q,(getdate()-545))
            and hdr.StartDateofService between '{start_date}' and '{end_date}'
        group by
            Hdr.TPSMemberID
        ;
   
            """,connection)
    
    
    
    all_members_df = pd.DataFrame(sql_all_members)
    
    officevisit_dataset_df = pd.DataFrame(sql_query)
    
    officevisit_agg_members_df = officevisit_dataset_df.groupby(['TPSMemberID'], as_index=False).agg({
                                                                           'OfficeVisit':  lambda x:x.max(),
                                                                           'OfficeVisit_Paid': lambda x:x.sum(),
                                                                           'OfficeVisit_Visits': lambda x:x.sum(),
                                                                           'OfficeVisit_DistinctClaims': lambda x:x.sum(),
                                                                            })
    
    # Do a left join
    officevisit_agg_members_df = all_members_df.merge(officevisit_agg_members_df,
                                                       how='left',
                                                       on='TPSMemberID')
    officevisit_agg_members_df = officevisit_agg_members_df.fillna(0)
    officevisit_agg_members_df['OfficeVisit'] = officevisit_agg_members_df['OfficeVisit'].astype(int)
    
    filename = f'{table_name}_Feature_OfficeVisit.pickle'
    #officevisit_agg_members_df.to_pickle(filename)
    #print(f'Office Visit Feature library is saved to: {filename}')
    
    
    return officevisit_agg_members_df





############################################15. OutPatient Hospital

def get_training_feature_library_OutPatientHospital(server_name, client_name,table_name, start_date, end_date):
    '''
    OutPatient
                                    
    '''
    connection = pyodbc.connect(
        CONNECTION_STRING
     )
    
    sql_all_members = pd.read_sql_query(f""" 
                                      select distinct
                                    TPSMemberID 
                                    from {client_name}.dbo.{table_name}
                                      """,connection)
    
    sql_query = pd.read_sql_query(f"""
                select distinct
                        hdr.TPSMemberID
                        ,1 as OutPatientHospital

                        ,sum(dtl.PaidAmount) as OutPatientHospital_Paid
                        ,count(distinct(hdr.TPSClaimID)) as OutPatientHospital_DistinctClaims
                        ,count(distinct(concat(hdr.tpsmemberid,hdr.startdateofservice))) as OutPatientHospital_Visits
                    
                    from {client_name}.dbo.Claims hdr with (nolock)
                    left join {client_name}.dbo.ClaimCharges dtl  with (nolock) on
                        (hdr.TPSClaimID = dtl.TPSClaimID)
                    where 
                        hdr.TPSMemberID in (
                                            select distinct
                                    TPSMemberID 
                                    from {client_name}.dbo.{table_name})
                        and (hdr.PlaceOfServiceCode = 22 OR hdr.PlaceOfServiceCode = 19 OR left(hdr.TypeofBill, 2) ='13')
                        --and year(hdr.StartDateofService) = year(getdate()-545)
                        --and datepart(q,hdr.StartDateofService) = datepart(q,(getdate()-545))
                        and hdr.StartDateofService between '{start_date}' and '{end_date}'
                    group by
                        Hdr.TPSMemberID
                    ;
   
            """,connection)
    
    
    
    all_members_df = pd.DataFrame(sql_all_members)
    
    outpatient_dataset_df = pd.DataFrame(sql_query)
    
    outpatient_agg_members_df = outpatient_dataset_df.groupby(['TPSMemberID'], as_index=False).agg({
                                                                           'OutPatientHospital':  lambda x:x.max(),
                                                                           'OutPatientHospital_Paid': lambda x:x.sum(),
                                                                           'OutPatientHospital_Visits': lambda x:x.sum(),
                                                                           'OutPatientHospital_DistinctClaims': lambda x:x.sum(),
                                                                            })
    
    # Do a left join
    outpatient_agg_members_df = all_members_df.merge(outpatient_agg_members_df,
                                                       how='left',
                                                       on='TPSMemberID')
    outpatient_agg_members_df = outpatient_agg_members_df.fillna(0)
    outpatient_agg_members_df['OutPatientHospital'] = outpatient_agg_members_df['OutPatientHospital'].astype(int)
    
    filename = f'{table_name}_Feature_OutPatientHospital.pickle'
    #outpatient_agg_members_df.to_pickle(filename)
    #print(f'OutPatientHospital Feature library is saved to: {filename}')
    
    
    return outpatient_agg_members_df







############################################16. ER Related

def get_training_feature_library_ER_Related(server_name, client_name,table_name, start_date, end_date):
    '''
    ERRelated
                                    
    '''
    connection = pyodbc.connect(
        CONNECTION_STRING
     )
    
    sql_all_members = pd.read_sql_query(f""" 
                                      select distinct
                                    TPSMemberID 
                                    from {client_name}.dbo.{table_name}
                                      """,connection)
    
    sql_query = pd.read_sql_query(f"""
                select distinct
                    hdr.TPSMemberID
                    ,1 as ER_Related

                    ,sum(dtl.PaidAmount) as ER_Related_Paid
                    ,count(distinct(hdr.TPSClaimID)) as ER_Related_DistinctClaims
                    ,count(distinct(concat(hdr.tpsmemberid,hdr.startdateofservice))) as ER_Related_Visits
                
                from {client_name}.dbo.Claims hdr with (nolock)
                left join {client_name}.dbo.ClaimCharges dtl  with (nolock) on
                    (hdr.TPSClaimID = dtl.TPSClaimID)
                where 
                    hdr.TPSMemberID in (
                                         select distinct
                                    TPSMemberID 
                                    from {client_name}.dbo.{table_name})
                    and (hdr.PlaceOfServiceCode = 22 OR hdr.PlaceOfServiceCode = 19 OR left(hdr.TypeofBill, 2) ='13')
                    --and year(hdr.StartDateofService) = year(getdate()-545)
                    --and datepart(q,hdr.StartDateofService) = datepart(q,(getdate()-545))
                    and hdr.StartDateofService between '{start_date}' and '{end_date}'
                group by
                    Hdr.TPSMemberID
                ;
   
            """,connection)
    
    
    
    all_members_df = pd.DataFrame(sql_all_members)
    
    errelated_dataset_df = pd.DataFrame(sql_query)
    
    errelated_agg_members_df = errelated_dataset_df.groupby(['TPSMemberID'], as_index=False).agg({
                                                                           'ER_Related':  lambda x:x.max(),
                                                                           'ER_Related_Paid': lambda x:x.sum(),
                                                                           'ER_Related_Visits': lambda x:x.sum(),
                                                                           'ER_Related_DistinctClaims': lambda x:x.sum(),
                                                                            })
    
    # Do a left join
    errelated_agg_members_df = all_members_df.merge(errelated_agg_members_df,
                                                       how='left',
                                                       on='TPSMemberID')
    errelated_agg_members_df = errelated_agg_members_df.fillna(0)
    errelated_agg_members_df['ER_Related'] = errelated_agg_members_df['ER_Related'].astype(int)
    
    filename = f'{table_name}_Feature_ER_Related.pickle'
    #errelated_agg_members_df.to_pickle(filename)
    #print(f'ER_Related Feature library is saved to: {filename}')
    
    
    return errelated_agg_members_df






############################################17. Behavioral Health

def get_training_feature_library_Behavioral_Health(server_name, client_name,table_name, start_date, end_date):
    '''
    Behavioral_Health
                                    
    '''
    connection = pyodbc.connect(
        CONNECTION_STRING
     )
    
    sql_all_members = pd.read_sql_query(f""" 
                                      select distinct
                                    TPSMemberID 
                                    from {client_name}.dbo.{table_name}
                                      """,connection)
    
    sql_query = pd.read_sql_query(f"""
                select distinct
                            hdr.TPSMemberID
                            ,1 as BHealth

                            ,sum(dtl.PaidAmount) as BHealth_Paid
                            ,count(distinct(hdr.TPSClaimID)) as BHealth_DistinctClaims
                            ,count(distinct(concat(hdr.tpsmemberid,hdr.startdateofservice))) as BHealth_Visits
                        
                        from {client_name}.dbo.ClaimCharges dtl with (nolock)
                        left join {client_name}.dbo.Claims hdr  with (nolock) on
                            (hdr.TPSClaimID = dtl.TPSClaimID)
                            and hdr.TPSMemberID in (
                                                select distinct
                                                TPSMemberID 
                                                from {client_name}.dbo.{table_name})
                            --and year(hdr.StartDateofService) = year(getdate()-545)
                            --and datepart(q,hdr.StartDateofService) = datepart(q,(getdate()-545))
                            and hdr.StartDateofService between '{start_date}' and '{end_date}'
                        where dtl.CPTHCPCSCode in (
                                                    '90791','90792','90832','90834','90837',
                                                    '90846','90847','90785','90853','99492',
                                                    '99493', '99494','90875','90876'
                                                    )
                        group by
                            Hdr.TPSMemberID
                        ;
   
            """,connection)
    
    
    
    all_members_df = pd.DataFrame(sql_all_members)
    
    bhealth_dataset_df = pd.DataFrame(sql_query)
    
    bhealth_agg_members_df = bhealth_dataset_df.groupby(['TPSMemberID'], as_index=False).agg({
                                                                           'BHealth':  lambda x:x.max(),
                                                                           'BHealth_Paid': lambda x:x.sum(),
                                                                           'BHealth_Visits': lambda x:x.sum(),
                                                                           'BHealth_DistinctClaims': lambda x:x.sum(),
                                                                            })
    
    # Do a left join
    bhealth_agg_members_df = all_members_df.merge(bhealth_agg_members_df,
                                                       how='left',
                                                       on='TPSMemberID')
    bhealth_agg_members_df = bhealth_agg_members_df.fillna(0)
    bhealth_agg_members_df['BHealth'] = bhealth_agg_members_df['BHealth'].astype(int)
    
    filename = f'{table_name}_Feature_BH.pickle'
    #bhealth_agg_members_df.to_pickle(filename)
    #print(f'BH Feature library is saved to: {filename}')
    
    
    return bhealth_agg_members_df











def log_details(name, load_type,step, activity, timestamp, duration,status,msg,logging_df):
    
    logging_df.loc[len(logging_df.index)] = [
    name,
    load_type,
    step,
    activity,
    timestamp,
    duration,
    status,
    msg]
    
    return 


def upload_logging_data(logging_df):
    
    LOG_TABLE_INSERT_STATEMENT = "INSERT INTO [Log].[dbo].[log_feature_library] VALUES (?,?,?,?,?,?,?,?)"
    
    
    logging_df['Duration']=logging_df['Duration'].astype(str)
    
    logging_df = logging_df[["Name","Type","Step", "Activity", "TimeStamp","Duration" ,"Status", "Msg"]]
    
    cnxn = pyodbc.connect(
        CONNECTION_STRING
     )
    log_cursor = cnxn.cursor()
    log_cursor.fast_executemany = True
    log_cursor.executemany(LOG_TABLE_INSERT_STATEMENT, logging_df.values.tolist())
    log_cursor.commit()
    log_cursor.close()
    cnxn.close()
    return
    







def run_training_feature_library(server_name, client_name, table_name,start_date, end_date, logging_df):

    global NAME, STEP, ACTIVITY

    
    start_time = datetime.now()
    print(f'---------------#1. Start Top Level data pull {start_time}')
    NAME = 'DxTopLevel'
    STEP  = 1
    ACTIVITY = 'Start DxTopLevel'
    DURATION = (start_time - start_time)
    STATUS = 'Success'
    MESSAGE = 'Success'
    # Log
    log_details(NAME,LOADTYPE,STEP,ACTIVITY, start_time, DURATION,STATUS,MESSAGE,logging_df)
    
    toplevel_df = get_training_feature_library_top_level_category(server_name,
                                                    client_name,
                                                    table_name,
                                                    start_date,
                                                    end_date)
    DURATION = datetime.now() - start_time
    ACTIVITY = 'End DxTopLevel'
    log_details(NAME,LOADTYPE,STEP,ACTIVITY, datetime.now(), DURATION,STATUS,MESSAGE,logging_df)
    print('Time elapsed (hh:mm:ss.ms) {}'.format(DURATION))
    
    # Open connection and upload the data
    upload_logging_data(logging_df)
    logging_df.drop(logging_df.index, inplace=True)
    
    print(f'---------------#1. Done Top Level data pull {datetime.now()}')

    
    start_time = datetime.now()
    print(f'---------------#2. Start POS data pull {start_time} ')
    NAME = 'POS'
    STEP  = STEP + 1
    ACTIVITY = 'Start POS'
    DURATION = (start_time - start_time)
    STATUS = 'Success'
    MESSAGE = 'Success'
    # Log
    log_details(NAME,LOADTYPE,STEP,ACTIVITY, start_time, DURATION,STATUS,MESSAGE,logging_df)
    pos_df = get_training_feature_library_placeofservice(server_name,
                                                    client_name,
                                                    table_name,
                                                    start_date,
                                                    end_date
                                            )
    DURATION = datetime.now() - start_time
    ACTIVITY = 'End POS'
    log_details(NAME,LOADTYPE,STEP,ACTIVITY, datetime.now(), DURATION,STATUS,MESSAGE,logging_df)
    print('Time elapsed (hh:mm:ss.ms) {}'.format(DURATION))
    
    # Open connection and upload the data
    # Open connection and upload the data
    upload_logging_data(logging_df)
    logging_df.drop(logging_df.index, inplace=True)
    print(f'---------------#2. Done POS data pull {datetime.now()}')


    start_time = datetime.now()
    print(f'---------------#3. Start Chronic Disease data pull {start_time}')
    NAME = 'ChronicDisease'
    STEP  = STEP + 1
    ACTIVITY = 'Start ChronicDisease'
    DURATION = (start_time - start_time)
    STATUS = 'Success'
    MESSAGE = 'Success'
    # Log
    log_details(NAME,LOADTYPE,STEP,ACTIVITY, start_time, DURATION,STATUS,MESSAGE,logging_df)
    chronic_df = get_training_feature_library_chronic_disease(server_name,
                                                    client_name,
                                                    table_name,
                                                             start_date,
                                                             end_date)
    DURATION = datetime.now() - start_time
    ACTIVITY = 'End ChronicDisease'
    log_details(NAME,LOADTYPE,STEP,ACTIVITY, datetime.now(), DURATION,STATUS,MESSAGE,logging_df)
    print('Time elapsed (hh:mm:ss.ms) {}'.format(DURATION))
    
    upload_logging_data(logging_df)
    logging_df.drop(logging_df.index, inplace=True)
    print(f'---------------#3. Done Chronic Disease data pull {datetime.now()}')



    start_time = datetime.now()
    print(f'---------------#4. Start TotalPaid data pull {start_time}')
    
    NAME = 'TotalPaid'
    STEP  = STEP + 1
    ACTIVITY = 'Start TotalPaid'
    DURATION = (start_time - start_time)
    STATUS = 'Success'
    MESSAGE = 'Success'
    # Log
    log_details(NAME,LOADTYPE,STEP,ACTIVITY, start_time, DURATION,STATUS,MESSAGE,logging_df)
    
    totalpaid_df = get_training_feature_library_totalpaid(server_name,
                                                    client_name,
                                                    table_name,
                                                    start_date,
                                                    end_date)
    DURATION = datetime.now() - start_time
    ACTIVITY = 'End TotalPaid'
    log_details(NAME,LOADTYPE,STEP,ACTIVITY, datetime.now(), DURATION,STATUS,MESSAGE,logging_df)
    print('Time elapsed (hh:mm:ss.ms) {}'.format(DURATION))
    
    upload_logging_data(logging_df)
    logging_df.drop(logging_df.index, inplace=True)
    print(f'---------------#4. Done TotalPaid data pull {datetime.now()}')



    start_time = datetime.now()
    print(f'---------------#5. Start HOSP visit data pull {start_time}')
    
    NAME = 'HOSP'
    STEP  = STEP +1
    ACTIVITY = 'Start HOSP'
    DURATION = (start_time - start_time)
    STATUS = 'Success'
    MESSAGE = 'Success'
    # Log
    log_details(NAME,LOADTYPE,STEP,ACTIVITY, start_time, DURATION,STATUS,MESSAGE,logging_df)
    hosp_df = get_training_feature_library_HOSP_visits(server_name,
                                                    client_name,
                                                    table_name,
                                                      start_date,
                                                      end_date)
    DURATION = datetime.now() - start_time
    ACTIVITY = 'End HOSP'
    log_details(NAME,LOADTYPE,STEP,ACTIVITY, datetime.now(), DURATION,STATUS,MESSAGE,logging_df)
    print('Time elapsed (hh:mm:ss.ms) {}'.format(DURATION))
    
    upload_logging_data(logging_df)
    logging_df.drop(logging_df.index, inplace=True)
    print(f'---------------#5. Done  HOSP visit data pull {datetime.now()}')


    start_time = datetime.now()
    print(f'---------------#6. Start EM  visit data pull {start_time}')
    
    NAME = 'EM'
    STEP  = STEP +1
    ACTIVITY = 'Start EM'
    DURATION = (start_time - start_time)
    STATUS = 'Success'
    MESSAGE = 'Success'
    # Log
    log_details(NAME,LOADTYPE,STEP,ACTIVITY, start_time, DURATION,STATUS,MESSAGE,logging_df)
    emvisit_df = get_training_feature_library_EM_visits(server_name,
                                                    client_name,
                                                    table_name,
                                                       start_date,
                                                       end_date)
    DURATION = datetime.now() - start_time
    activity = 'End EM'
    log_details(NAME,LOADTYPE,STEP,ACTIVITY, datetime.now(), DURATION,STATUS,MESSAGE,logging_df)
    print('Time elapsed (hh:mm:ss.ms) {}'.format(DURATION))
    
    upload_logging_data(logging_df)
    logging_df.drop(logging_df.index, inplace=True)
    print(f'---------------#6. Done  EM visit data pull {datetime.now()}')
    
    
    start_time = datetime.now()
    print(f'---------------#7. Start MbrDemographics data pull {start_time}')
    
    NAME = 'MbrDemographics'
    STEP  = STEP +1
    ACTIVITY = 'Start MbrDemographics'
    DURATION = (start_time - start_time)
    STATUS = 'Success'
    MESSAGE = 'Success'
    # Log
    log_details(NAME,LOADTYPE,STEP,ACTIVITY, start_time, DURATION,STATUS,MESSAGE,logging_df)
    demo_df = get_training_feature_library_mbr_demographics(server_name,
                                                    client_name,
                                                    table_name,
                                                    start_date,
                                                       end_date)
    DURATION = datetime.now() - start_time
    activity = 'End  MbrDemographics'
    log_details(NAME,LOADTYPE,STEP,ACTIVITY, datetime.now(), DURATION,STATUS,MESSAGE,logging_df)
    print('Time elapsed (hh:mm:ss.ms) {}'.format(DURATION))
    
    upload_logging_data(logging_df)
    logging_df.drop(logging_df.index, inplace=True)
    print(f'---------------#7. Done  MbrDemographics data pull {datetime.now()}')


    start_time = datetime.now()
    print(f'---------------#8. Start ER  visit data pull {start_time}')
    
    NAME = 'ER'
    STEP  = STEP + 1
    ACTIVITY = 'Start ER'
    DURATION = (start_time - start_time)
    STATUS = 'Success'
    MESSAGE = 'Success'
    # Log
    log_details(NAME,LOADTYPE,STEP,ACTIVITY, start_time, DURATION,STATUS,MESSAGE,logging_df)
    ervisit_df = get_training_feature_library_ER_visits(server_name,
                                                    client_name,
                                                    table_name,
                                                       start_date,
                                                       end_date)
    DURATION = datetime.now() - start_time
    ACTIVITY = 'End  ER'
    log_details(NAME,LOADTYPE,STEP,ACTIVITY, datetime.now(), DURATION,STATUS,MESSAGE,logging_df)
    print('Time elapsed (hh:mm:ss.ms) {}'.format(DURATION))
    
    upload_logging_data(logging_df)
    logging_df.drop(logging_df.index, inplace=True)
    print(f'---------------#8. Done  ER visit data pull {datetime.now()}')



    start_time = datetime.now()
    print(f'---------------#9. Start SNF  visit data pull {start_time}')
    
    NAME = 'SNF'
    STEP  = STEP + 1
    ACTIVITY = 'Start SNF'
    DURATION = (start_time - start_time)
    STATUS = 'Success'
    MESSAGR = 'Success'
    # Log
    log_details(NAME,LOADTYPE,STEP,ACTIVITY, start_time, DURATION,STATUS,MESSAGE,logging_df)
    snf_df = get_training_feature_library_SNF_visits(server_name,
                                                    client_name,
                                                    table_name,
                                                    start_date,
                                                    end_date)
    DURATION = datetime.now() - start_time
    ACTIVITY = 'End  SNF'
    log_details(NAME,LOADTYPE,STEP,ACTIVITY, datetime.now(), DURATION,STATUS,MESSAGE,logging_df)
    print('Time elapsed (hh:mm:ss.ms) {}'.format(DURATION))
    
    upload_logging_data(logging_df)
    logging_df.drop(logging_df.index, inplace=True)
    print(f'---------------#9. Done  SNF visit data pull {datetime.now()}')


    start_time = datetime.now()
    print(f'---------------#10.Start  HomeCare visit data pull {start_time}')
    
    NAME = 'HomeCare'
    STEP  = STEP + 1
    ACTIVITY = 'Start HomeCare'
    DURATION = (start_time - start_time)
    STATUS = 'Success'
    MESSAGE = 'Success'
    # Log
    log_details(NAME,LOADTYPE,STEP,ACTIVITY, start_time, DURATION,STATUS,MESSAGE,logging_df)
    homecare_df = get_training_feature_library_HomeCare_visits(server_name,
                                                    client_name,
                                                    table_name,
                                                              start_date,
                                                              end_date)
    DURATION = datetime.now() - start_time
    ACTIVITY = 'End  HomeCare'
    log_details(NAME,LOADTYPE,STEP,ACTIVITY, datetime.now(), DURATION,STATUS,MESSAGE,logging_df)
    print('Time elapsed (hh:mm:ss.ms) {}'.format(DURATION))
    
    upload_logging_data(logging_df)
    logging_df.drop(logging_df.index, inplace=True)
    print(f'---------------#10. Done  HomeCare visit data pull {datetime.now()}')
    
    
    
    start_time = datetime.now()
    print(f'---------------#11. Start  OfficeVisit data pull {start_time}')
    
    NAME = 'OfficeVisit'
    STEP  = STEP + 1
    ACTIVITY = 'Start OfficeVisit'
    DURATION = (start_time - start_time)
    STATUS = 'Success'
    MESSAGE = 'Success'
    # Log
    log_details(NAME,LOADTYPE,STEP,ACTIVITY, start_time, DURATION,STATUS,MESSAGE,logging_df)
    officevisit_df = get_training_feature_library_OfficeVisit(server_name,
                                                    client_name,
                                                    table_name,
                                                              start_date,
                                                              end_date)
    DURATION = datetime.now() - start_time
    ACTIVITY = 'End  OfficeVisit'
    log_details(NAME,LOADTYPE,STEP,ACTIVITY, datetime.now(), DURATION,STATUS,MESSAGE,logging_df)
    print('Time elapsed (hh:mm:ss.ms) {}'.format(DURATION))
    
    upload_logging_data(logging_df)
    logging_df.drop(logging_df.index, inplace=True)
    print(f'---------------#11. Done  OfficeVisit data pull {datetime.now()}')


    start_time = datetime.now()
    print(f'---------------#12.Start  Outpatient visit data pull {start_time}')
    
    NAME = 'OutPatientHospital'
    STEP  = STEP + 1
    ACTIVITY = 'Start OutPatientHospital'
    DURATION = (start_time - start_time)
    STATUS = 'Success'
    MESSAGE = 'Success'
    # Log
    log_details(NAME,LOADTYPE,STEP,ACTIVITY, start_time, DURATION,STATUS,MESSAGE,logging_df)
    outpatient_df = get_training_feature_library_OutPatientHospital(server_name,
                                                    client_name,
                                                    table_name,
                                                              start_date,
                                                              end_date)
    DURATION = datetime.now() - start_time
    ACTIVITY = 'End  OutPatientHospital'
    log_details(NAME,LOADTYPE,STEP,ACTIVITY, datetime.now(), DURATION,STATUS,MESSAGE,logging_df)
    print('Time elapsed (hh:mm:ss.ms) {}'.format(DURATION))
    
    upload_logging_data(logging_df)
    logging_df.drop(logging_df.index, inplace=True)
    print(f'---------------#12. Done  OutPatientHospital data pull {datetime.now()}')
    
    
    
    
    start_time = datetime.now()
    print(f'---------------#13. Start  ER Related visit data pull {start_time}')
    
    NAME = 'ER_Related'
    STEP  = STEP + 1
    ACTIVITY = 'Start ER_Related'
    DURATION = (start_time - start_time)
    STATUS = 'Success'
    MESSAGE = 'Success'
    # Log
    log_details(NAME,LOADTYPE,STEP,ACTIVITY, start_time, DURATION,STATUS,MESSAGE,logging_df)
    errelated_df = get_training_feature_library_ER_Related(server_name,
                                                    client_name,
                                                    table_name,
                                                              start_date,
                                                              end_date)
    DURATION = datetime.now() - start_time
    ACTIVITY = 'End  ER_Related'
    log_details(NAME,LOADTYPE,STEP,ACTIVITY, datetime.now(), DURATION,STATUS,MESSAGE,logging_df)
    print('Time elapsed (hh:mm:ss.ms) {}'.format(DURATION))
    
    
    upload_logging_data(logging_df)
    logging_df.drop(logging_df.index, inplace=True)
    print(f'---------------#13. Done  ER Related data pull  {datetime.now()}')
    
    
    
    
    start_time = datetime.now()
    print(f'---------------#14. Start  BH data pull {start_time}')
    
    NAME = 'BH'
    STEP  = STEP + 1
    ACTIVITY = 'Start BH'
    DURATION = (start_time - start_time)
    STATUS = 'Success'
    MESSAGE = 'Success'
    # Log
    log_details(NAME,LOADTYPE,STEP,ACTIVITY, start_time, DURATION,STATUS,MESSAGE,logging_df)
    bh_df = get_training_feature_library_Behavioral_Health(server_name,
                                                    client_name,
                                                    table_name,
                                                              start_date,
                                                              end_date)
    DURATION = datetime.now() - start_time
    ACTIVITY = 'End  BH'
    log_details(NAME,LOADTYPE,STEP,ACTIVITY, datetime.now(), DURATION,STATUS,MESSAGE,logging_df)
    print('Time elapsed (hh:mm:ss.ms) {}'.format(DURATION))
    
    upload_logging_data(logging_df)
    logging_df.drop(logging_df.index, inplace=True)
    print(f'---------------#14. Done  BH data pull {datetime.now()}')



    start_time = datetime.now()
    print(f'---------------#15. Start SubCategory data pull {start_time}')
    
    NAME = 'DxSubCategory'
    STEP  = STEP + 1
    ACTIVITY = 'Start DxSubCategory'
    DURATION = (start_time - start_time)
    STATUS = 'Success'
    MESSAGE = 'Success'
    # Log
    log_details(NAME,LOADTYPE,STEP,ACTIVITY, start_time, DURATION,STATUS,MESSAGE,logging_df)
    subcategory_df = get_training_feature_library_subcategory(server_name,
                                                    client_name,
                                                    table_name,
                                                    start_date,
                                                    end_date,
                                           iteration_length = 100
                                            )
    DURATION = datetime.now() - start_time
    ACTIVITY = 'End  DxSubCategory'
    log_details(NAME,LOADTYPE,STEP,ACTIVITY, datetime.now(), DURATION,STATUS,MESSAGE,logging_df)
    print('Time elapsed (hh:mm:ss.ms) {}'.format(DURATION))
    
    upload_logging_data(logging_df)
    logging_df.drop(logging_df.index, inplace=True)
    print(f'---------------#15. Done SubCategory data pull {datetime.now()}')



    start_time = datetime.now()
    print(f'---------------#16. Start CPT3 data pull {start_time}')
    NAME = 'CPT3'
    STEP  = STEP + 1
    ACTIVITY = 'Start CPT3'
    DURATION = (start_time - start_time)
    STATUS = 'Success'
    MESSAGE = 'Success'
    # Log
    log_details(NAME,LOADTYPE,STEP,ACTIVITY, start_time, DURATION,STATUS,MESSAGE,logging_df)
    cpt3_df = get_training_feature_library_cpt3(server_name,
                                                    client_name,
                                                    table_name,
                                                  start_date,
                                                          end_date,
                                           iteration_length = 100
                                            )
    time_elapsed = datetime.now() - start_time
    ACTIVITY = 'End  CPT3'
    log_details(NAME,LOADTYPE,STEP,ACTIVITY, datetime.now(), DURATION,STATUS,MESSAGE,logging_df)
    print('Time elapsed (hh:mm:ss.ms) {}'.format(DURATION))
    
    upload_logging_data(logging_df)
    logging_df.drop(logging_df.index, inplace=True)
    print(f'---------------#16. Done CPT3 data pull {datetime.now()}')
    





    
    start_time = datetime.now()
    print(f'---------------#17. Start Category data pull {start_time}')
    NAME = 'Category'
    STEP  = STEP + 1
    ACTIVITY = 'Start Category'
    DURATION = (start_time - start_time)
    STATUS = 'Success'
    MESSAGE = 'Success'
    # Log
    log_details(NAME,LOADTYPE,STEP,ACTIVITY, start_time, DURATION,STATUS,MESSAGE,logging_df)
    category_df = get_training_feature_library_category(server_name,
                                                    client_name,
                                                    table_name,
                                                        start_date,
                                                        end_date,
                                           iteration_length = 100
                                            )
    DURATION = datetime.now() - start_time
    ACTIVITY = 'End  Category'
    log_details(NAME,LOADTYPE,STEP,ACTIVITY, datetime.now(), DURATION,STATUS,MESSAGE,logging_df)
    print('Time elapsed (hh:mm:ss.ms) {}'.format(DURATION))

    upload_logging_data(logging_df)
    logging_df.drop(logging_df.index, inplace=True)
    
    print(f'---------------#17. Done Category data pull {datetime.now()}')
    
    

     
    

    
    
    
    
    
    ################################################ Merge
    # First join demo vs. totalpaid
    
    totalpaid_df['TPSMemberID'] = totalpaid_df['TPSMemberID'].astype(str)
    demo_df['TPSMemberID'] = demo_df['TPSMemberID'].astype(str)
    all_files_df = demo_df.merge(totalpaid_df, on='TPSMemberID', how='inner')
    
    all_files_df['TPSMemberID'] = all_files_df['TPSMemberID'].astype(str)
    
    
    toplevel_df['TPSMemberID'] = toplevel_df['TPSMemberID'].astype(str)
    pos_df['TPSMemberID'] = pos_df['TPSMemberID'].astype(str)
    chronic_df['TPSMemberID'] = chronic_df['TPSMemberID'].astype(str)
    hosp_df['TPSMemberID'] = hosp_df['TPSMemberID'].astype(str)
    emvisit_df['TPSMemberID'] = emvisit_df['TPSMemberID'].astype(str)
    ervisit_df['TPSMemberID'] = ervisit_df['TPSMemberID'].astype(str)
    snf_df['TPSMemberID'] = snf_df['TPSMemberID'].astype(str)
    homecare_df['TPSMemberID'] = homecare_df['TPSMemberID'].astype(str)
    officevisit_df['TPSMemberID'] = officevisit_df['TPSMemberID'].astype(str)
    outpatient_df['TPSMemberID'] = outpatient_df['TPSMemberID'].astype(str)
    errelated_df['TPSMemberID'] = errelated_df['TPSMemberID'].astype(str)
    bh_df['TPSMemberID'] = bh_df['TPSMemberID'].astype(str)
    subcategory_df['TPSMemberID'] = subcategory_df['TPSMemberID'].astype(str)
    cpt3_df['TPSMemberID'] = cpt3_df['TPSMemberID'].astype(str)
    category_df['TPSMemberID'] = category_df['TPSMemberID'].astype(str)
    
    
    
    all_files_df = all_files_df.merge(toplevel_df,  on='TPSMemberID', how='left')
    all_files_df = all_files_df.merge(pos_df,  on='TPSMemberID', how='left')
    all_files_df = all_files_df.merge(chronic_df,  on='TPSMemberID', how='left')
    all_files_df = all_files_df.merge(hosp_df,  on='TPSMemberID', how='left')
    all_files_df = all_files_df.merge(emvisit_df,  on='TPSMemberID', how='left')
    all_files_df = all_files_df.merge(ervisit_df,  on='TPSMemberID', how='left')
    all_files_df = all_files_df.merge(snf_df,  on='TPSMemberID', how='left')
    all_files_df = all_files_df.merge(homecare_df,  on='TPSMemberID', how='left')
    all_files_df = all_files_df.merge(officevisit_df,  on='TPSMemberID', how='left')
    all_files_df = all_files_df.merge(outpatient_df,  on='TPSMemberID', how='left')
    all_files_df = all_files_df.merge(errelated_df,  on='TPSMemberID', how='left')
    all_files_df = all_files_df.merge(bh_df,  on='TPSMemberID', how='left')
    all_files_df = all_files_df.merge(subcategory_df,  on='TPSMemberID', how='left')
    all_files_df = all_files_df.merge(cpt3_df,  on='TPSMemberID', how='left')
    all_files_df = all_files_df.merge(category_df,  on='TPSMemberID', how='left')
    all_files_df = all_files_df.fillna(0)
    
    filename = f'{table_name}_training_all_features.pickle'
    all_files_df.to_pickle(filename)

    filename_sparse = f'{table_name}_training_all_features_sparse_matrix.pickle'
    # Drop the MemberID before converting to sparse matrix
    # TODO: Save the MemberIDs as list and save the columns sequence too
    all_files_sparse_matrix = csr_matrix(all_files_df.set_index('TPSMemberID').values)

    with open(filename_sparse, 'wb') as sparse_file:
        pickle.dump(all_files_sparse_matrix, sparse_file)

    ###################Get the file back
    # Get the columns and MemberIDs
    #df = pd.DataFrame.sparse.from_spmatrix(matrix, columns=['A', 'B', 'C', 'D'])

    #TODO :Append DEMO to the Demographic dataset

    
    #Category
    #ToplevelCategory
    #SubCategory

    # Next
    # Feed the server throughout the code

    print(f'Feature library is saved to: {filename}')
    
    
    return 









if __name__ == '__main__':
    
   # run this code like this
    #%run final_feature_library_utilities_eds01.py  --server_name TPS-PRD-DS02  --client_name Centene  --table_name DS_314B   --start_date 2022-07-01 --end_date 2022-07-30
    
    
    
    try:
        LOADTYPE = 'NA'
        NAME = 'NA'
        STEP  = 0
        ACTIVITY = ''
        STATUS = 'NA'
        MESSAGE = 'NA'
        parser = argparse.ArgumentParser(description='Feature Library')

        
        parser.add_argument('--server_name', metavar='server_name', required=True,
                            help='ex: TPS-PRD-DS02')
        parser.add_argument('--client_name', metavar='client_name', required=True,
                            help='ex: Centene')
        parser.add_argument('--table_name', metavar='table_name', required=True,
                            help='ex: DSC_314')
        parser.add_argument('--start_date', metavar='start_date', required=True,
                            help='ex: 2022-07-01')
        parser.add_argument('--end_date', metavar='end_date', required=True,
                            help='ex: 2022-09-30')

        args = parser.parse_args()
    
    
        

        LOADTYPE = 'Training'

        run_training_feature_library(server_name = args.server_name,
                                    client_name = args.client_name,
                                    table_name = args.table_name,
                                     start_date = args.start_date,
                                     end_date = args.end_date,
                                    logging_df = log_df)
            

       
    except Exception as e:
        
        
        STATUS = 'ERROR'
        MESSAGE = str(e)
        #print(LOADTYPE)
        #print(NAME)
        #print(STEP)
        #print(ACTIVITY)
        #print(MESSAGE)
        DURATION = (datetime.now() - datetime.now())

        log_details(NAME, LOADTYPE,STEP, ACTIVITY, datetime.now(), DURATION,STATUS,MESSAGE,log_df)
        
    
        
        
        
        
    



