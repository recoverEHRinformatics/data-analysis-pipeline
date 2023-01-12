# if any of the modules above are not already installed please use the command below in your notebook to install the module
# !pip install NameOfYourModule (e.g. !pip install pandas)

import pandas as pd
import numpy as np
import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy import text as sqlalchemy_text

import pyarrow.parquet as pq
import pyarrow as pa

import os

#####################################################################
######################## raw data extraction ########################
#####################################################################

def extract_raw_data(query: str, site_names: list, source_data_path: str, data_name: str, database_engine: str):
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
            print(f"{site} table has: {len(df.syn_pt_id.unique())} unique patients")

            # concatenate individual site data into one data frame (i.e. final_df)
            final_df = pd.concat([final_df, df], ignore_index=True)

            del df
            counter += 1
            print("*"*50)

        # error-agnostic pass which will leave out the individual site
        # make sure to investigate further why the query failed for a specific site
        # best way to investigate the issue is to run the SQL query in PgAdmin
        # possible issues could be data type mismatch for certain columns
        except:
            error_msg = f"# {counter}. {site}'s {data_name} WAS NOT PROCESSED #"
            print("#"*len(error_msg))
            print(error_msg)
            print("#"*len(error_msg))

    # saving the table with all sites data concatenated as parquet format in source data folder
    pq.write_table(pa.Table.from_pandas(
        final_df), f"{source_data_path}/{data_name}.parquet", compression="BROTLI")
    print(f"All sites {data_name} data have been saved as a parquet file in:")
    print(f"{source_data_path}/{data_name}.parquet")

    # optional lines to generate information about all sites data
    print(f"{site} table shape: {final_df.shape}")
    print(f"{site} table has: {len(final_df.syn_pt_id.unique())} unique patients")
    print(f"{site} table has: {len(final_df.site.unique())} unique sites")

    return final_df

#################################################################################
############################ identify COVID patients ############################
#################################################################################

def get_lab_pts(index_all: pd.DataFrame, patid_column='syn_pt_id'):
    '''get_lab_pts finds the list of all patients with at least one positive COVID-19 PCR or antigen lab 

    Args:
        index_all (pd.DataFrame): a dataframe contianing all COVID-19 indications.
        patid_column (str, optional): the column in the dataframe indicating the patient identifier. Defaults to 'syn_pt_id'.

    Returns:
        list: all patients with at least one COVID-19 PCR or antigen test
    '''

    # at least 1 positive PCR or antigen test
    covid_lab = index_all.query(
        "index_type == 'lab' and index_result == 'positive'")
    covid_lab = list(set(covid_lab[patid_column]))

    return covid_lab


def get_ip_dx_pts(index_all: pd.DataFrame, patid_column='syn_pt_id'):
    '''get_ip_dx_pts finds the list of all patients with at least one COVID-19 dx in an inpatient setting

    Args:
        index_all (pd.DataFrame): a dataframe contianing all COVID-19 indications.
        patid_column (str, optional): the column in the dataframe indicating the patient identifier. Defaults to 'syn_pt_id'.

    Returns:
        list: all patients with at least one dx in an inpatient setting
    '''

    covid_ip = index_all[(index_all['index_type'] == 'covid_dx') & (
        index_all['enc_type'].isin(['IP', 'EI']))]

    covid_ip = list(set(covid_ip[patid_column]))

    return covid_ip


def get_av_dx_pts(index_all: pd.DataFrame, patid_column='syn_pt_id'):
    '''get_av_dx_pts finds the list of all patients with at least one COVID-19 dx in an outpatient setting

    Args:
        index_all (pd.DataFrame): a dataframe contianing all COVID-19 indications.
        patid_column (str, optional): the column in the dataframe indicating the patient identifier. Defaults to 'syn_pt_id'.

    Returns:
        list: all patients with at least one dx in an outpatient setting
    '''

    covid_av = index_all[(index_all['index_type'] == 'covid_dx') & (
        index_all['enc_type'].isin(['AV', 'ED', 'TH', 'OA']))]

    covid_av = list(set(covid_av[patid_column]))

    return covid_av


def get_two_av_dx_pts(index_all: pd.DataFrame, patid_column='syn_pt_id'):
    '''get_two_av_dx_pts finds the list of all patients with at least two COVID-19 dx in an outpatient setting

    Args:
        index_all (pd.DataFrame): a dataframe contianing all COVID-19 indications.
        patid_column (str, optional): the column in the dataframe indicating the patient identifier. Defaults to 'syn_pt_id'.

    Returns:
        list: all patients with at least two dx in an outpatient setting
    '''

    covid_av_two = index_all[(index_all['index_type'] == 'covid_dx') & (
        index_all['enc_type'].isin(['AV', 'ED', 'TH', 'OA']))]

    # count the number of outpatient dx per patient
    covid_av_two = covid_av_two[[patid_column, 'index_date']].groupby(
        patid_column).nunique().reset_index()
    # patients with at least 2 outpatient dx
    covid_av_two = covid_av_two[covid_av_two['index_date'] >= 2]
    covid_av_two = list(set(covid_av_two[patid_column]))

    return covid_av_two


def get_paxlovid_pts(index_all:pd.DataFrame, patid_column='syn_pt_id'):
    '''get_paxlovid_pts finds the list of all patients with at least one paxlovid prescription

    Args:
        index_all (pd.DataFrame): a dataframe contianing all COVID-19 indications.
        patid_column (str, optional): the column in the dataframe indicating the patient identifier. Defaults to 'syn_pt_id'.

    Returns:
        list: all patients with at one paxlovid prescription
    '''

    covid_paxlovid = index_all[index_all.index_type=='paxlovid']
    covid_paxlovid = list(set(covid_paxlovid[patid_column]))

    return covid_paxlovid


def get_index_event(df: pd.DataFrame, index_date_column='index_date', patid_column='syn_pt_id', start_date='03/01/2020', end_date='03/01/2023'):
    '''get_index_event function finds the first instance of an index event per patient

    Args:
        df (pd.DataFrame): a dataframe with all instances of covid indication for all patients (i.e. positive lab, dx, and etc.)
        index_date_column (str, optional): the column in the dataframe indicating the date. Defaults to 'index_date'.
        patid_column (str, optional): the column in the dataframe indicating the patient identifier. Defaults to 'syn_pt_id'.
        start_date (str, optional): start of the study period. Defaults to '03/01/2020' variable.
        end_date (str, optional): end of the study period. Defaults to '03/01/2023' variable.

    Returns:
        pd.DataFrame: returns a dataframe with one row per patient inidicating the first instance of the index event
    '''

    start_date = pd.to_datetime(start_date).date()
    end_date = pd.to_datetime(end_date).date()

    index = df[(df[index_date_column] >= start_date)
               & (df[index_date_column] <= end_date)]

    index = index.sort_values(index_date_column).drop_duplicates(patid_column)
    index.reset_index(drop=True, inplace=True)

    return index


##############################################################################
########################### identify PASC patients ###########################
##############################################################################

def get_pasc_category(diagnosis: pd.DataFrame, index: pd.DataFrame, PASC_definition_reference: pd.DataFrame, patid_column='syn_pt_id', category='ccsr_category'):
    '''get_pasc_category function finds the date of first instance of all PASC like diagnosis for each patient.
    The resulting dataframes from this function will be used to identify date of PASC diagnosis and subphenotypes. 

    Args:
        diagnosis (pd.DataFrame): standard diagnosis table from PCORnet CDM containing all diagnoses for patients.
        index (pd.DataFrame): custom index table created using a pre-defined function containing the index dates for each patient.
        PASC_definition_reference (pd.DataFrame): a reference spreadsheet containing all ICD-10 codes and diagnosis categories of PASC-like symptoms.
        patid_column (str, optional): the column in the dataframe indicating the patient identifier. Defaults to 'syn_pt_id'.
        category (str, optional): Diagnosis category column in the PASC_definition_reference table. Defaults to 'ccsr_category'.

    Returns:
        A tuple of two pandas dataframe. Both dataframes have one unique row per patient and each diagnosis category as a column. 
        categorized_diff: the values for each column is the time difference (in days) between the index date and the first instance of the diagnosis
        categorized_date: the date of first instance of the diagnosis
    '''

    # merge with index table to get the first instance of index event
    dx = pd.merge(
        diagnosis,
        index[[patid_column, 'index_date']],
        on=patid_column, how='inner'
    ).drop_duplicates()

    # calculate the difference in days between the diagnosis date and index event date
    # date_diff_from_index < 0 means the diagnosis was recorded before the index event date
    # date_diff_from_index > 0 means the diagnosis was recorded after the index event date
    dx['date_diff_from_index'] = (
        dx['admit_date'] - dx['index_date']) / np.timedelta64(1, 'D')

    # select the columns needed and drop duplicates
    dx.drop(columns=['site'], inplace=True)
    dx.drop_duplicates(inplace=True)

    # join to PASC_defintion to get the dx category if it is a PASC dx
    dx = pd.merge(
        dx,
        PASC_definition_reference[['i10_code', category]],
        left_on='dx',
        right_on='i10_code',
        how='inner'
    )

    # throw away any diagnoses in the blackout period and
    # balckout period is defined as 7 days before and 30 days after the index date
    dx = dx[
        ~(dx['date_diff_from_index'].between(-7, 30, inclusive='neither'))
    ]

    # throw away any diagnoses 180 days after the index date
    dx = dx[dx['date_diff_from_index'] <= 180]

    # select the necessary columns and drop the duplicates
    # by only including the CCSR category column (i.e. ccsr_category) and excluding the ICD-10 code column (i10_code)
    # we ensure that if there are several ICD-10 codes with the same category, we count them as the same
    dx = dx[[patid_column, 'date_diff_from_index', category, 'admit_date']].copy()
    dx.drop_duplicates(inplace=True)
    dx.reset_index(drop=True, inplace=True)

    # create a pivot table with each column representing the smallest value of date_diff_from_index
    # negative number means this is not a PASC diagnosis and it was previously present for this patient
    # positive number means this is a PASC diagnosis and the patient developed this diagnosis after index event date
    # 0 as a value means this diagnosis was developed at the same time as the index event date
    # NaN means the patient has never been diagnosed with this particular diagnosis
    categorized_diff = dx.pivot_table(
        index=[patid_column],
        columns=[category],
        values='date_diff_from_index',
        aggfunc='min')
    categorized_diff.drop_duplicates(inplace=True)

    # create a pivot table with each column representing the date of the first instance of a diagnosis in that category
    # NaN means the patient has never been diagnosed
    categorized_date = dx.sort_values(
        [patid_column, 'admit_date']).drop_duplicates(patid_column)
    categorized_date = categorized_date.pivot(
        index=[patid_column], columns=[category], values='admit_date')

    categorized_date.reset_index(inplace=True)
    categorized_diff.reset_index(level=patid_column, inplace=True)

    return categorized_diff, categorized_date


def get_pasc_subphenotype(pasc_diff: pd.DataFrame, patid_column='syn_pt_id'):
    '''get_pasc_subphenotype function identifies one subphenotype per patient

    Args:
        pasc_diff (pd.DataFrame): the first returned result (i.e. categorized_diff) from get_pasc_category function
        patid_column (str, optional): the column in the dataframe indicating the patient identifier. Defaults to 'syn_pt_id'.

    Returns:
        pd.DataFrame: a dataframe with a unique row per patient indicating one PASC subphenotype
    '''

    # set patid_column as the index
    temp_df = pasc_diff.copy()
    temp_df.set_index(patid_column, inplace=True)
    # replace negative values with nan to only focus on the real PASC diagnoses
    # negative values represent pre-existing diagnosis and are not PASC
    temp_df[temp_df < 0] = np.nan

    # find the column NAME that has the smallest value (.idxmin(axis=1))
    # column NAME will indicate the subphenotype name
    pasc_subphenotype = pd.DataFrame(temp_df.idxmin(
        axis=1, skipna=True), columns=['subphenotype_name'])

    # find the smallest column VALUE (.min(axis=1))
    # the smallest value across all columns indicate date difference (in days) between the index date and the first instance of PASC diagnosis
    pasc_subphenotype = pasc_subphenotype.merge(
        pd.DataFrame(temp_df.min(axis=1, skipna=True),
                     columns=['subphenotype_days']),
        on=patid_column,
        how='inner'
    )

    # resetting the index will make the patid_column to be a regular column rather than the index for this dataframe
    pasc_subphenotype.reset_index(inplace=True)

    # categorize the interval
    pasc_subphenotype['subphenotype_interval'] = np.select(
        [
            pasc_subphenotype['subphenotype_days'].between(30, 59, inclusive='both'), 
            pasc_subphenotype['subphenotype_days'].between(60, 89, inclusive='both'), 
            pasc_subphenotype['subphenotype_days'].between(90, 119, inclusive='both'),
            pasc_subphenotype['subphenotype_days'].between(120, 149, inclusive='left'), 
            pasc_subphenotype['subphenotype_days'] >= 150
        ], [
            '30-59', 
            '60-89', 
            '90-119', 
            '120-149', 
            '150+'
        ], default=np.NaN
    )

    pasc_subphenotype = pasc_subphenotype.query("~subphenotype_name.isnull()")
    pasc_subphenotype.reset_index(drop=True, inplace=True)

    return pasc_subphenotype


def get_pasc_pts(index:pd.DataFrame, pasc_yn:pd.DataFrame, pasc_subphenotype:pd.DataFrame, patid_column='syn_pt_id'):
    '''get_pasc_pts function takes in a series of custom tables resulting from other pre-defined function to generate a list of patients
    with their PASC status, subphenotype, and the index date. Please note this function only works for when the patient has one subphenotype.

    Args:
        index (pd.DataFrame): dataframe generated by get_index_event function.
        pasc_yn (pd.DataFrame): dataframe with information whether a diagnosis category is PASC or pre-existing.
        pasc_subphenotype (pd.DataFrame): dataframe generated by get_pasc_subphenotype function.
        patid_column (str, optional): the column in the dataframe indicating the patient identifier. Defaults to 'syn_pt_id'.

    Returns:
        pd.DataFrame: a dataframe with PASC and subphenotype information for all patients with an index date.
    '''

    # list of all patients with an index date
    pasc_pts = index[[patid_column, 'index_date']].copy()

    # dichotomous variable indicating PASC status
    pasc_yn.set_index(patid_column, inplace=True)
    pasc_pts['pasc_yn'] = np.where(pasc_pts[patid_column].isin(list(pasc_yn[(pasc_yn == 1).any(axis=1)].index)), 1, 0)
    pasc_yn.reset_index(inplace=True)

    pasc_pts = pd.merge(
        pasc_pts,
        pasc_subphenotype,
        on='syn_pt_id',
        how='left'
    )

    return pasc_pts


#####################################################################
######################## demographic cleanup ########################
#####################################################################

def categorize_age(df: pd.DataFrame, age_column: str):
    '''categorize_age function takes a table containing a column with age of the patient and categorize the patient's age.

    Args:
        df (pd.DataFrame): Any dataframe with an age column.
        age_column (str): Name of the column that contains the age of the patient. The column values should be int or float. This is often age of patient as of index event.

    Returns:
        pd.series: returns a series that can be directly assigned as a new column to any dataframe.
    '''

    age_group = np.select(
        [
            round(df[age_column]).between(0, 1, inclusive='left'),
            round(df[age_column]).between(1, 4, inclusive='both'),
            round(df[age_column]).between(5, 9, inclusive='both'),
            round(df[age_column]).between(10, 15, inclusive='both'),
            round(df[age_column]).between(16, 20, inclusive='both'),
            round(df[age_column]).between(21, 35, inclusive='both'),
            round(df[age_column]).between(36, 45, inclusive='both'),
            round(df[age_column]).between(46, 55, inclusive='both'),
            round(df[age_column]).between(56, 65, inclusive='both'),
            round(df[age_column]) > 65
        ],
        [
            '<1',
            '1-4',
            '5-9',
            '10-15',
            '16-20',
            '21-35',
            '36-45',
            '46-55',
            '56-65',
            '66+'
        ],
        default='unknown'
    )

    return age_group


def clean_sex(df: pd.DataFrame, sex_column='sex'):
    '''clean_sex function replaces PCORnet CDM value sets of sex with a human-readble value taken from the official PCORnet CDM dictionary. 

    Args:
        df (pd.DataFrame): Any dataframe with ethnicity column with standard reference terminology values of PCORnet CDM. This is often the standard DEMOGRAPHIC table.
        sex_column (str, optional): Name of the column containing the sex information. Defaults to 'sex'.

    Returns:
        pd.DataFrame: the same input dataframe (i.e. df) with the values of sex_column replaced accordingly.
    '''
    df.replace({
        sex_column: {
            'A': 'Other/Missing/Unknown',
            'F': 'Female',
            'M': 'Male',
            'NI': 'Other/Missing/Unknown',
            'UN': 'Other/Missing/Unknown',
            'OT': 'Other/Missing/Unknown'
        }}, inplace=True)

    return df


def clean_race(df: pd.DataFrame, race_column='race'):
    '''clean_race function replaces PCORnet CDM value sets of race with a human-readble value taken from the official PCORnet CDM dictionary.

    Args:
        df (pd.DataFrame): Any dataframe with RACE column with standard reference terminology values of PCORnet CDM. This is often the standard DEMOGRAPHIC table.
        race_column (str, optional): Name of the column containing the race information. Defaults to 'race'.

    Returns:
        pd.DataFrame: the same input dataframe (i.e. df) with the values of race_column replaced accordingly.
    '''

    df.replace({
        race_column: {
            '01': 'American Indian or Alaska Native',
            '1': 'American Indian or Alaska Native', # not a standard reference terminology
            '02': 'Asian',
            '2': 'Asian',  # not a standard reference terminology
            '03': 'Black or African American',
            '3': 'Black or African American',  # not a standard reference terminology
            '04': 'Native Hawaiian or Other Pacific Islander',
            '4': 'Native Hawaiian or Other Pacific Islander', # not a standard reference terminology
            '05': 'White',
            '5': 'White',  # not a standard reference terminology
            '06': 'Multiple race',
            '6': 'Multiple race',  # not a standard reference terminology
            '07': 'Refuse to answer',
            '7': 'Refuse to answer',  # not a standard reference terminology
            'NI': 'No race information',
            '0': 'Unknown',  # not a standard reference terminology
            'UN': 'Unknown',
            'OT': 'Other'
        }}, inplace=True)

    return df


def clean_ethnicity(df: pd.DataFrame, ethnicity_column='hispanic'):
    '''clean_ethnicity function replaces PCORnet CDM value sets of ethnicity with a human-readble value taken from the official PCORnet CDM dictionary.

    Args:
        df (pd.DataFrame): Any dataframe with ethnicity column with standard reference terminology values of PCORnet CDM. This is often the standard DEMOGRAPHIC table.
        ethnicity_column (str, optional): Name of the column containing the ethnicity information. Defaults to 'hispanic'.

    Returns:
        pd.DataFrame: the same input dataframe (i.e. df) with the values of ethnicity_column replaced accordingly.
    '''

    df.replace({
        ethnicity_column: {
            'Y': 'Hispanic',
            'N': 'Not hispanic',
            'R': 'Refuse to answer',
            'NI': 'No ethnicity information',
            'UN': 'Unknown',
            'OT': 'Other'
        }}, inplace=True)

    return df


def categorize_race_ethnicity(df: pd.DataFrame, ethnicity_column='hispanic', race_column='race'):
    '''categorize_race_ethnicity function uses the already processed race and ethnicity values to combine and categorize the patients per qtwg's categories.

    Args:
        df (pd.DataFrame): Any dataframe with ethnicity column with standard reference terminology values of PCORnet CDM. This is often the standard DEMOGRAPHIC table.
        ethnicity_column (str, optional): Name of the column containing the ethnicity information. Defaults to 'hispanic'.
        race_column (str, optional): Name of the column containing the race information. Defaults to 'race'.

    Returns:
        pd.series: returns a series that can be directly assigned as a new column to any dataframe.
    '''

    race_ethnicity = np.select(
        [
            ((df[ethnicity_column].isin(['Not hispanic'])) & (df[race_column].isin(['White']))),
            ((df[ethnicity_column].isin(['Not hispanic'])) & (df[race_column].isin(['Black or African American']))),
            (df[ethnicity_column].isin(['Hispanic'])),
            ((df[ethnicity_column].isin(['Not hispanic'])) & (df[race_column].isin(['Asian']))),
            (
                (df[race_column].isin(['Native Hawaiian or Other Pacific Islander']))
                | (df[race_column].isin(['American Indian or Alaska Native']))
                | (df[ethnicity_column].isin(['Other']))
                | (df[race_column].isin(['Other', 'Multiple race']))
            ), 
            (
                (df[ethnicity_column].isin(
                    ['Unknown', 'Refuse to answer', 'No ethnicity information', '']))
                | (df[race_column].isin(['Unknown', 'Refuse to answer', 'No race information', '']))
            )
        ], [
            'Non-Hispanic white',
            'Non-Hispanic black',
            'Hispanic',
            'Non-hispanic Asian',
            'Other',
            'Missing/Unknown'
        ], default='ISSUE WITH RACE OR ETHNICITY COLUMN'
    )

    return race_ethnicity


