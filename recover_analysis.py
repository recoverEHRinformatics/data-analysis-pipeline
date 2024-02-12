# if any of the modules above are not already installed please use the command below in your notebook to install the module
# !pip install NameOfYourModule (e.g. !pip install pandas)

import pandas as pd
import numpy as np
import datetime

import os
backup_env_path = os.environ["PATH"]
os.environ["PATH"] += os.pathsep + 'D:/sajjad/Graphviz/bin/'
import dask.dataframe as dd

from typing import Union
import multiprocessing

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy import text as sqlalchemy_text

import pyarrow.parquet as pq
import pyarrow as pa

import contextlib
import io
import sys

#####################################################################
######################## memory optimization ########################
#####################################################################
data_types_dict = {
    # all tables
    'syn_pt_id': 'object',
    'site': 'category',

    # demographic
    'birth_date': 'datetime64[ns]',
    'race': 'category',
    'hispanic': 'category',
    'sex': 'category', 

    # encounter
    'enc_type': 'category',
    'admit_date': 'datetime64[ns]',
    'discharge_date': 'datetime64[ns]',
    'discharge_status': 'category',
    'discharge_disposition': 'category',
    'payer_type_primary': 'category',
    'payer_type_secondary': 'category',

    # diagnosis
    'dx': 'category',
    'dx_source': 'category',

    # index_all
    'index_date': 'datetime64[ns]',
    'index_type': 'category',
    'index_result': 'category',

    # flu
    'flu_date': 'datetime64[ns]',
    'flu_type': 'category',

    # death
    'death_date': 'datetime64[ns]',

    # obs_gen
    'obsgen_start_date': 'datetime64[ns]',
    'obsgen_type': 'category',
    'obsgen_code': 'category',
    'obsgen_result_num': 'float32', # TODO: consider implementing a dynamic float downcasting
    'obsgen_result_qual': 'category',
    'obsgen_result_text': 'category',

    # prescribing
    'rx_order_date': 'datetime64[ns]',
    'rx_start_date': 'datetime64[ns]',
    'rx_end_date': 'datetime64[ns]',
    'rxnorm_cui': 'category',
    'atc_code': 'category',
    'atc_name': 'category',
    'rxnorm_name': 'category',

    # procedure
    'px': 'category',
    'px_date': 'datetime64[ns]',

    # lds_address_history
    'address_period_start': 'datetime64[ns]',

    # immunization
    'vx_record_date': 'datetime64[ns]',
    'vx_admin_date': 'datetime64[ns]',

    # lab_result_cm
    'result_date': 'datetime64[ns]',
    'result_qual': 'category',
    'result_modifier': 'category',
    'result_num': 'float32'
}

def optimize_memory(df:pd.DataFrame, data_types_dict:dict):
    section_title = "memory optimization"
    section_border =  "=" * (len(section_title) + 22)
    section_boundary = "=" * (len(section_title) + 22)
    print(f"""{section_border}\n{"="*10} {section_title} {"="*10}\n{section_border}""")

    initial_usage = round(df.memory_usage().sum() / 1024**2, 2)
    print(f"memory usage before optimization: ~{format(initial_usage, ',')} MB")
    print(section_boundary)
    print("summary:")
    
    # drop patid column if syn_pt_id column already exists
    # syn_pt_id combines patid and site column and can be used as a unique patient identifier
    if ('patid' in df.columns) & ('syn_pt_id' in df.columns):
        df.drop(columns='patid', inplace=True)
        print(f"- patid column was dropped")

    # reassign data types only if the column exists and data type is not optimized
    affected_cols = []
    already_optimized_cols = []
    for col_name, col_dtype in data_types_dict.items():
        # check if it is already optimized
        if (col_name in df.columns) and (df[col_name].dtype == col_dtype):
            already_optimized_cols.append(col_name)
        
        elif (col_name in df.columns) and (df[col_name].dtype != col_dtype):
                # 
                if col_dtype == 'datetime64[ns]':
                    affected_cols.append(col_name)
                    df[col_name] = pd.to_datetime(df[col_name], errors = 'coerce')
                else:
                    affected_cols.append(col_name)
                    df[col_name] = df[col_name].astype(data_types_dict[col_name])
    
    print(f"- the following column(s) data type was changed: {affected_cols}")
    print(f"- the following column(s) were already optimized: {already_optimized_cols}")
    print(f"- the following column(s) were unaffected: {list(set(df.columns) - set(affected_cols) - set(already_optimized_cols))}")
    print(f"- consider dropping the column(s) that were unaffected if they are not being utilized")

    final_usage = round(df.memory_usage().sum() / 1024**2, 2)

    print(section_boundary)
    print(f"memory usage after optimization: ~{format(final_usage, ',')} MB")
    print(f"memory usage was reduced by {round((initial_usage - final_usage) * 100 / initial_usage, 2)}%")
    print(section_boundary)
    return df

#####################################################################
######################## raw data extraction ########################
#####################################################################
def extract_raw_data(query: str, site_names: list, source_data_path: str, data_name: str, database_engine: str, saved_format='parquet'):
    '''extract_raw_data is a function to query the live data for all sites in the analysis, concatenate them together, and save them as parquet files.

    Args:
        query (str): SQL query to be executed aginst the live data in database.
        site_names (list): a list of all site names (schemas as they appear in the database) used in the analysis.
        source_data_path (str): the source data folder path where the final data frame will be saved.
        data_name (str): name of the table as it appears in the database.
        database_engine (str): the database engine address.

    Returns:
        final_df (DataFrame): a pandas dataframe containing all sites data
    '''

    final_df = pd.DataFrame()
    counter = 1

    section_title = "raw data extraction"
    section_border =  "=" * (len(section_title) + 22)
    section_boundary = "=" * (len(section_title) + 22)
    print(f"""{section_border}\n{"="*10} {section_title} {"="*10}\n{section_border}""")

    for site in site_names:
        try:
            print(f"Query {counter}. {site}'s {data_name} started")

            # replace where input query indicates "SiteSchema" with the site's real schema
            modified_query = query.replace("SiteSchema", f"{site}")

            df = pd.read_sql(
                modified_query, database_engine
            )
            # creating an additional column to indicate the site
            df['site'] = site
            print(f"{site}'s {data_name} query is finished")

            # optional lines to generate information about each site's data
            print(f"{site} table shape: {df.shape}")

            # # memory optimization
            # final_df = optimize_memory(df=final_df, data_types_dict=data_types_dict)
            # print("memory optimization is done\n\n")

            # concatenate individual site data into one data frame (i.e. final_df)
            final_df = pd.concat([final_df, df], ignore_index=True)

            del df
            counter += 1
            print(section_boundary)

        # error-agnostic pass which will leave out the individual site
        # make sure to investigate further why the query failed for a specific site
        # best way to investigate the issue is to run the SQL query in PgAdmin
        # possible issues could be data type mismatch for certain columns
        except:
            error_msg = f"= {counter}. {site}'s {data_name} WAS NOT PROCESSED ="
            print(section_boundary)
            print(error_msg)
            print(section_boundary)

    print("individual site raw data extraction is done\n\n")

    # memory optimization
    final_df = optimize_memory(df=final_df, data_types_dict=data_types_dict)
    print("memory optimization is done\n\n")

    if saved_format.lower() == 'parquet':
        # saving the table with all sites data concatenated as PARQUET format in source data folder
        pq.write_table(pa.Table.from_pandas(final_df), f"{source_data_path}/{data_name}.parquet", compression="BROTLI")
        print(f"All sites {data_name} data have been saved as a parquet file in:")
        print(f"{source_data_path}/{data_name}.parquet")
    elif saved_format.lower() == 'csv':
            # saving the table with all sites data concatenated as CSV format in source data folder
            final_df.to_csv(f"{source_data_path}/{data_name}.csv")
            print(f"All sites {data_name} data have been saved as a csv file in:")
            print(f"{source_data_path}/{data_name}.csv")
    else: # default is PARQUET format
        # saving the table with all sites data concatenated as PARQUET format in source data folder
        pq.write_table(pa.Table.from_pandas(final_df), f"{source_data_path}/{data_name}.parquet", compression="BROTLI")
        print(f"All sites {data_name} data have been saved as a parquet file in:")
        print(f"{source_data_path}/{data_name}.parquet")

    # optional lines to generate information about all sites data
    print(f"The final table shape: {final_df.shape}")
    # print(f"{site} table has: {len(final_df.syn_pt_id.unique())} unique patients")
    print(f"The final table has: {len(final_df.site.unique())} unique sites")

    return final_df

#########################################################################################
########################### site specific raw data extraction ###########################
#########################################################################################
def extract_raw_data_site(query: str, site_names: list, source_data_path: str, data_name: str, database_engine: str, saved_format='parquet'):

    counter = 1
    section_title = "raw data extraction"
    section_border =  "=" * (len(section_title) + 22)
    section_boundary = "=" * (len(section_title) + 22)
    print(f"""{section_border}\n{"="*10} {section_title} {"="*10}\n{section_border}""")

    for site in site_names:
        try:
            print(f"Query {counter}. {site}'s {data_name} started")

            # replace where input query indicates "SiteSchema" with the site's real schema
            modified_query = query.replace("SiteSchema", f"{site}")

            df = pd.read_sql(
                modified_query, database_engine
            )
            # creating an additional column to indicate the site
            df['site'] = site
            print(f"{site}'s {data_name} query is finished")

            # memory optimization
            df = optimize_memory(df=df, data_types_dict=data_types_dict)
            print("memory optimization is done\n\n")

            if saved_format.lower() == 'parquet':
                # saving the table with all sites data concatenated as PARQUET format in source data folder
                pq.write_table(pa.Table.from_pandas(df), f"{source_data_path}/{site}/{data_name}.parquet", compression="BROTLI")
                print(f"{site}'s {data_name} data have been saved as a parquet file in:")
                print(f"{source_data_path}/{site}/{data_name}.parquet")
            elif saved_format.lower() == 'csv':
                    # saving the table with all sites data concatenated as CSV format in source data folder
                    df.to_csv(f"{source_data_path}/{site}/{data_name}.csv")
                    print(f"{site}'s {data_name} data have been saved as a csv file in:")
                    print(f"{source_data_path}/{site}/{data_name}.csv")
            else: # default is PARQUET format
                # saving the table with all sites data concatenated as PARQUET format in source data folder
                pq.write_table(pa.Table.from_pandas(df), f"{source_data_path}/{site}/{data_name}.parquet", compression="BROTLI")
                print(f"{site}'s {data_name} data have been saved as a parquet file in:")
                print(f"{source_data_path}/{site}/{data_name}.parquet")

        # error-agnostic pass which will leave out the individual site
        # make sure to investigate further why the query failed for a specific site
        # best way to investigate the issue is to run the SQL query in PgAdmin
        # possible issues could be data type mismatch for certain columns
        except:
            error_msg = f"= {counter}. {site}'s {data_name} WAS NOT PROCESSED ="
            print(section_boundary)
            print(error_msg)
            print(section_boundary)
        counter += 1
        print(section_boundary)

##############################################################################
########################### identify PASC patients ###########################
##############################################################################
def get_pasc_all(diagnosis:Union[pd.DataFrame, dd.DataFrame], PASC_definition:Union[pd.DataFrame, dd.DataFrame], patid_column='syn_pt_id', category='ccsr_category', **kwargs):
    '''get_pasc_all finds all instances of PASC subphenotypes among patients. A patient can have more than one PASC subphenotype.

    Args:
        diagnosis (pd.DataFrame): standard diagnosis table from PCORnet CDM containing all diagnoses for patients.
        PASC_definition (pd.DataFrame): a reference spreadsheet containing all ICD-10 codes and diagnosis categories of PASC-like symptoms.
        patid_column (str, optional): _description_. Defaults to 'syn_pt_id'.
        category (str, optional): _description_. Defaults to 'ccsr_category'.
        **kwargs: allows you to provide additional named arguments. To be used if the diagnosis table does not have an index_date columns

    Returns:
        pd.DataFrame: a dataframe with all PASC subphenotypes per patient per subphenotype
    '''
    # convert the diagnosis input variable to a dask dataframe if it's a pandas dataframe
    if isinstance(diagnosis, pd.DataFrame):
        pasc_diagnoses = dd.from_pandas(diagnosis, npartitions = 2 * multiprocessing.cpu_count()) # TODO: look into picking the optimal number


    # create a smaller subset of the diagnosis table containing only the PASC like diagnoses
    pasc_diagnoses = dd.merge(
        pasc_diagnoses,
        PASC_definition[['i10_code', 'ccsr_category']],
        left_on='dx',
        right_on='i10_code', 
        how='inner'
    )
    # dropping duplicated column
    pasc_diagnoses = pasc_diagnoses.drop(columns=(['i10_code']))

    # save the index argument if provided
    index = kwargs.get('index', None)

    # Check if the diagnosis table contains an index_date column
    if isinstance(index, pd.DataFrame) or isinstance(index, dd.DataFrame):
        if 'index_to_admit' not in diagnosis.columns:
            # merge with index table to get the first instance of index event
            pasc_diagnoses = dd.merge(
                pasc_diagnoses,
                index[[patid_column, 'index_date']],
                on=patid_column, how='inner'
            ).drop_duplicates()

            # calculate the difference in days between the diagnosis date and index event date
            # date_diff_from_index < 0 means the index event date was recorded before the diagnosis 
            # date_diff_from_index > 0 means the index event date was recorded after the diagnosis
            pasc_diagnoses = pasc_diagnoses.assign(index_to_admit = (pasc_diagnoses['index_date'] - pasc_diagnoses['admit_date']) / np.timedelta64(1, 'D'))

        else:
            error_msg = "You provided an index table, and your diagnosis table already has an index_date column \
                \nSuggested solutions: \
                \n\t- Drop the following columns in your diagnosis table: 'index_date' and 'index_to_admit' \
                \n\t- Or remove the index table from this function and keep the 'index_date' and 'index_to_admit' columns in the diagnosis table"
            return print(error_msg)
        
    else:
        if 'index_to_admit' not in diagnosis.columns:
            error_msg = "You must provide an index table with an index_date column"
            return print(error_msg)

    # for better readibility flip the number's sign of index_to_admit column and rename
    pasc_diagnoses = pasc_diagnoses.assign(days_from_index = -1 * pasc_diagnoses['index_to_admit'])
    pasc_diagnoses = pasc_diagnoses.drop(columns=['site', 'index_to_admit'])

    # throw away any diagnoses in the blackout period and
    # balckout period is defined as 7 days before and 30 days after the index date
    pasc_diagnoses = pasc_diagnoses[~(pasc_diagnoses['days_from_index'].between(-7, 30, inclusive='both'))]

    # throw away any diagnoses 180 days after the index date
    pasc_diagnoses = pasc_diagnoses[pasc_diagnoses['days_from_index'] <= 180]

    # select the necessary columns and drop the duplicates
    # by only including the CCSR category column (i.e. ccsr_category) and excluding the ICD-10 code column (i10_code)
    # we ensure that if there are several ICD-10 codes within the same category, we count them as the same
    pasc_diagnoses = pasc_diagnoses[[patid_column, 'days_from_index', category, 'admit_date']].drop_duplicates().reset_index(drop=True)

    # find the first time earliest incidence of pasc appeared per patient per category
    pasc_diagnoses = pasc_diagnoses.groupby([patid_column, category]).min()
    pasc_diagnoses = pasc_diagnoses.rename(columns={
        'admit_date': 'date_incidence', # indicating the earliest date of PASC evidence for determining incidence
        'days_from_index': 'days_incidence' # indicating how long after the index date earliest date of PASC evidence incidence appeared
        })

    # only keep the diagnoses that happened after the index date for the first time
    pasc_diagnoses = pasc_diagnoses[pasc_diagnoses.days_incidence >= 0]

    # get year and month of the pasc indcidence
    pasc_diagnoses = pasc_diagnoses.assign(year_incidence = pasc_diagnoses['date_incidence'].apply(lambda x: x.year, meta=('date_incidence', 'int8')))
    pasc_diagnoses = pasc_diagnoses.assign(month_incidence = pasc_diagnoses['date_incidence'].apply(lambda x: x.month, meta=('date_incidence', 'int8')))

    # keeping patid_column (i.e. syn_pt_id) and category (i.e. ccsr_category) columns as a column rather than an index
    pasc_diagnoses = pasc_diagnoses.reset_index()

    pasc_diagnoses = pasc_diagnoses.compute()

    return pasc_diagnoses
############################################################################
######################## charlson comorbidity index ########################
############################################################################
def categorize_charlson(diagnosis, charlson_mapping, **kwargs):

    # ensuring all diagnoses are prior to index date
    charlson_output = diagnosis[diagnosis['index_to_admit']>0][['syn_pt_id', 'dx']].drop_duplicates().reset_index(drop=True)

    # create a dataframe with unique diagnoses present in the clinical data
    charlson_output_unique = pd.DataFrame()
    charlson_output_unique['dx'] = list(set(charlson_output['dx']))

    # function to find a matching ICD code prefix and return the category
    def find_category(dx_df):
        prefixes = charlson_mapping['dx_clean'].unique()
        for prefix in prefixes:
            if dx_df.startswith(prefix):
                category = charlson_mapping[charlson_mapping['dx_clean'] == prefix]['category'].iloc[0]
                # score = mapping_df[mapping_df['dx_clean'] == prefix]['score'].iloc[0]
                return category
        return None

    # find the category based on the ICD-10 codes it starts with and the mapping table
    charlson_output_unique['category'] = charlson_output_unique['dx'].apply(find_category)

    charlson_output_unique = charlson_output_unique[charlson_output_unique['category'].notnull()].reset_index(drop=True)

    # create a mapping dictionary to assign score
    score_mapping_dict = charlson_mapping[['category', 'score']].set_index('category').to_dict()['score']
    # assign score based on the category using the mapping dictionary
    charlson_output_unique['score'] = charlson_output_unique['category'].map(score_mapping_dict)

    # get score per pt per dx
    charlson_output = pd.merge(
        charlson_output,
        charlson_output_unique,
        on='dx',
        how='inner'
    )

    # get score per category
    charlson_output = charlson_output.pivot_table(
        index=['syn_pt_id'],
        columns='category',
        values='score'
    ).fillna(0).reset_index(drop=False)

    # sum all the categories scores
    charlson_output['CCI_score'] = charlson_output.sum(axis=1)

    # categorize the scores
    charlson_output['CCI_category'] = np.select(
        [
            charlson_output['CCI_score']==0,
            charlson_output['CCI_score'].between(1,3,inclusive='both'),
            charlson_output['CCI_score']>=4
        ],
        [
            '0',
            '1-3',
            '4+'
        ],
    )

    return charlson_output
    
############################################################################
# ######################## charlson comorbidity index ########################
# ############################################################################
# def categorize_charlson(diagnosis, charlson_mapping, demographic, patid_column='syn_pt_id', **kwargs):
#     charlson_dx = diagnosis.copy()

#     # function to find a matching ICD code prefix and return the category
#     def find_category(dx_df):
#         prefixes = charlson_mapping['dx_clean'].unique()
#         for prefix in prefixes:
#             if dx_df.startswith(prefix):
#                 category = charlson_mapping[charlson_mapping['dx_clean'] == prefix]['category'].iloc[0]
#                 # score = mapping_df[mapping_df['dx_clean'] == prefix]['score'].iloc[0]
#                 return category
#         return None

#     # find the category based on the ICD-10 codes it starts with and the mapping table
#     charlson_dx['category'] = charlson_dx['dx'].apply(find_category)
#     # create a smaller subset of the diagnosis table containing only the charlson diagnoses
#     charlson_dx = charlson_dx[charlson_dx['category'].notnull()].reset_index(drop=True)

#     # create a mapping dictionary to assign score
#     score_mapping_dict = charlson_mapping[['category', 'score']].set_index('category').to_dict()['score']
#     # assign score based on the category using the mapping dictionary
#     charlson_dx['score'] = charlson_dx['category'].map(score_mapping_dict)

#     # save the index argument if provided
#     index = kwargs.get('index', None)

#     # Check if the diagnosis table contains an index_date column
#     if isinstance(index, pd.DataFrame):
#         if 'index_to_admit' not in diagnosis.columns:
#             # merge with index table to get the first instance of index event
#             charlson_dx = pd.merge(
#                 charlson_dx,
#                 index[[patid_column, 'index_date']],
#                 on=patid_column, how='inner'
#             ).drop_duplicates()
#         else:
#             error_msg = "You provided an index table, and your diagnosis table already has an index_date column \
#                 \nSuggested solutions: \
#                 \n\t- Drop the following columns in your diagnosis table: 'index_date' and 'index_to_admit' \
#                 \n\t- Or remove the index table from this function and keep the 'index_date' and 'index_to_admit' columns in the diagnosis table"
#             return print(error_msg)
        
#     else:
#         if 'index_to_admit' not in diagnosis.columns:
#             error_msg = "You must provide an index table with an index_date column"
#             return print(error_msg)

#     charlson_dx = charlson_dx[charlson_dx['admit_date'] <= charlson_dx['index_date']]

#     charlson_dx = charlson_dx[['syn_pt_id', 'index_date', 'admit_date', 'dx', 'category', 'score']].drop_duplicates()

#     charlson_dx = charlson_dx.pivot_table(
#         index=['syn_pt_id', 'index_date'],
#         columns='category',
#         values='score'
#     ).fillna(0).reset_index(drop=False)

#     charlson_dx['CCI_score'] = charlson_dx.sum(axis=1)

#     charlson_dx['CCI_category'] = np.select(
#         [
#             charlson_dx['CCI_score']==0,
#             charlson_dx['CCI_score'].between(1,3,inclusive='both'),
#             charlson_dx['CCI_score']>=4
#         ],
#         [
#             '0',
#             '1-3',
#             '4+'
#         ],
#     )


#     charlson_dx = charlson_dx.merge(
#         demographic[['syn_pt_id', 'birth_date']],
#         on='syn_pt_id',
#         how='left'
#     )

#     # adding age score and update the CCI score and category in a separate column
#     charlson_dx['age_as_of_index'] = (charlson_dx['index_date'] - charlson_dx['birth_date']) / np.timedelta64(1, 'Y')

#     charlson_dx['age_score'] = np.select(
#         [
#             charlson_dx['age_as_of_index'] < 50,
#             charlson_dx['age_as_of_index'].between(50, 60, inclusive='left'),
#             charlson_dx['age_as_of_index'].between(60, 70, inclusive='left'),
#             charlson_dx['age_as_of_index'].between(70, 80, inclusive='left'),
#             charlson_dx['age_as_of_index'] >= 80
#         ],
#         [
#             0,
#             1,
#             2,
#             3,
#             4
#         ],
#         default= -111 # to catch errors, no patient should have this scroe
#     )

#     if charlson_dx[charlson_dx['age_score']<0]['syn_pt_id'].nunique() > 0:
#         print("WARNING THERE ARE PATIENTS WITH WRONG AGE SCORE CALCULATED")
#         print("PLEASE QA PATIENTS WITH AGE_SCORE LESS THAN 0")

#     charlson_dx.drop(columns=['index_date', 'birth_date', 'age_as_of_index'], inplace=True)

#     charlson_dx['CCI_score_age_based'] = charlson_dx['CCI_score'] + charlson_dx['age_score']

#     charlson_dx['CCI_category_age_based'] = np.select(
#         [
#             charlson_dx['CCI_score_age_based']==0,
#             charlson_dx['CCI_score_age_based'].between(1,3,inclusive='both'),
#             charlson_dx['CCI_score_age_based']>=4
#         ],
#         [
#             '0',
#             '1-3',
#             '4+'
#         ],
#     )

#     return charlson_dx
