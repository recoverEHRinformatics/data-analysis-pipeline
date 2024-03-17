import os
import requests
import re

def print_in_box(text):
    # Split the text into lines
    lines = text.split('\n')
    # Find the length of the longest line
    max_length = max(len(line) for line in lines)
    # Create the top and bottom border
    top_bottom_border = '*' * (max_length + 4)  # Adding 4 for padding

    # Print the top border
    print(top_bottom_border)
    # Print each line of text within the border
    for line in lines:
        # Right-pad lines so they all match the longest line
        padded_line = line.ljust(max_length)
        print(f'* {padded_line} *')
    # Print the bottom border
    print(top_bottom_border)

# get mandatory user variables
def get_user_input():
    print_in_box("Please provide the information below to setup your project")
    # server_name = input("Enter the name of the server you will be using e.g. recover_dev_s11: ")

    main_path = input("Enter project folder path (e.g. D:/FolderName/SubFolder1/SubFolder2): ")
    main_path_pattern = r"D:((\/[\w-]+)+)\/?"
    # Check if the folder path is valid
    while True:
        if re.match(main_path_pattern, main_path):
            break
        else:
            main_path = input("Invalid project folder path was entered. Please try again and make sure the correct folder path is entered (e.g D:/FolderName/SubFolder1/SubFolder2): ")
    
    # main_path = os.path(main_path)
            
    def is_valid_date(date_string):
        # Check if the date format is valid
        from datetime import datetime
        try:
            datetime.strptime(date_string, "%Y-%m-%d")
            return True
        except ValueError:
            return False
        
    study_start_date = input("Enter the study period start date (e.g., YYYY-MM-DD): ")
    while not is_valid_date(study_start_date):
        study_start_date = input("Invalid date format. Please try again.\nEnter the study period start date (e.g., YYYY-MM-DD): ")

    study_end_date = input("Enter the study period end date (e.g., YYYY-MM-DD): ")
    while not is_valid_date(study_end_date):
        study_end_date = input("Invalid date format. Please try again.\nEnter the study period end date (e.g., YYYY-MM-DD): ")

    site_names = input("Enter the site names, separated by commas (e.g. wcm, nyu, intermountain): ").replace(" ", "").split(',')

    return main_path, study_start_date, study_end_date, site_names

# create a folder based on the main_path provided
def create_project_folder(main_path):
    if not os.path.exists(main_path):
        os.makedirs(main_path)
        print_in_box(f"Directory {main_path} was created.")
    else:
        print_in_box(f"Directory {main_path} already exists. Using the existing directory.")

    # where the raw data will be saved
    source_data_path = os.path.join(main_path, "source_data")
    # create an empty folder if it does not exist
    if os.path.exists(source_data_path) != True:
        os.makedirs(source_data_path)

    # where the interim processed data will be saved
    interim_data_path = os.path.join(main_path, "interim_data")
    # create an empty folder if it does not exist
    if os.path.exists(interim_data_path) != True:
        os.makedirs(interim_data_path)

    # where the results will be saved
    result_path = os.path.join(main_path, "result")
    # create an empty folder if it does not exist
    if os.path.exists(result_path) != True:
        os.makedirs(result_path)

# this will create a config file for the project
def create_config_file(main_path, study_start_date, study_end_date, site_names):
    # below are content of the configuration ptyhon file that will be generated using the user-provided values
    # in addition to some default information
    config_content = f"""
import os

{'#'*100}
# server/database information
server = "aurora-stack-auroradbcluster-1rbqp9aty8v4q.cluster-cbs1thv2ku8o.us-east-1.rds.amazonaws.com"
database = ""
username = ""
password = ""
port = '5432'

""" + "database_string = f'postgresql://{username}:{password}@{server}:{port}/{database}'" + f"""

{'#'*100}
# project paths
main_path = "{main_path}"
source_data_path = "{main_path}/source_data"
interim_data_path = "{main_path}/interim_data"
result_path = "{main_path}/result"

{'#'*100}
# query specific information
study_start_date = "{study_start_date}"
study_end_date = "{study_end_date}"
site_names = {site_names}

{'#'*100}
# create any named variables you would like to use across the project e.g.:
# lookback_period_start_date = "2018-01-01"
"""
    config_file_path = os.path.join(main_path, "query_config.py")

    # Write the config content to the Python file
    with open(config_file_path, 'w') as config_file:
        config_file.write(config_content)

    print_in_box(f"""Config file created at {config_file_path}
WARNING: please make sure to modify the query_config.py file and enter the correct server information""")

# getting the most up to date code base
def get_code_base(main_path):
    # you may download the latest version here: https://raw.githubusercontent.com/recoverEHRinformatics/data-analysis-pipeline/main/recover_analysis.py
    url = 'https://raw.githubusercontent.com/recoverEHRinformatics/data-analysis-pipeline/main/recover_analysis.py'
    r = requests.get(url, allow_redirects=True)

    open(os.path.join(main_path, "recover_analysis.py"), 'wb').write(r.content)

    print_in_box(f"""The most up to date recover code set was saved at {main_path}/recover_analysis.py""")

import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

def create_raw_data_extraction_notebook(main_path):
    nb = new_notebook()

    nb.cells.append(new_markdown_cell("""# READ BEFORE RUNNING
Please make sure to review the pre-existing code blocks in this notebook.

These pre-existing code blocks are automatically generated to demonstrate the best way to set up your raw data extraction pipeline.
    """))

    nb.cells.append(new_code_cell("""import datetime
time_started = datetime.datetime.now()
    """))

    nb.cells.append(new_markdown_cell("""# setup
    """))

    nb.cells.append(new_code_cell("""import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy import text as sqlalchemy_text
import pyarrow.parquet as pq
import pyarrow as pa
import os
"""))

    nb.cells.append(new_code_cell("""# import all variables stored in the query_config.py
import query_config
                                  
main_path = query_config.main_path
interim_data_path = query_config.interim_data_path
source_data_path = query_config.source_data_path
result_path = query_config.result_path
database_string = query_config.database_string
site_names = query_config.site_names
study_start_date = query_config.study_start_date
study_end_date = query_config.study_end_date
"""))

    nb.cells.append(new_code_cell("""# import all functions in the code base
import recover_analysis as rp
"""))

    nb.cells.append(new_code_cell("""database_engine = create_engine(database_string)"""))

    nb.cells.append(new_markdown_cell("""# total_network"""))

    nb.cells.append(new_code_cell('''query_total_network_pt = """
SELECT COUNT(DISTINCT patid) AS total_pt 
FROM SiteSchema_pcornet_all.demographic;"""

total_network_pt = rp.extract_raw_data(
    query=query_total_network_pt, 
    site_names=site_names, 
    source_data_path=source_data_path, 
    data_name='total_network_pt',
    database_engine=database_engine)
                                  
del query_total_network_pt
del total_network_pt
'''))

    nb.cells.append(new_markdown_cell("""# index_all"""))

    nb.cells.append(new_code_cell('''# please note we are limiting the records in this table up to the end of the study period
query_index_all = f"""
SELECT
CONCAT({"'SiteSchema'"}, '_', patid) AS syn_pt_id
, t1.index_date, t1.index_type, t1.index_result, t1.enc_type
FROM qtwg.SiteSchema_index_all t1
WHERE t1.index_date >= '{study_start_date}' AND t1.index_date <= '{study_end_date}'
AND (
        index_type in ('covid_dx', 'paxlovid', 'remdesivir', 'pasc', 'B94.8')
        OR (index_type = 'lab' AND index_result = 'positive')
    );
"""

index_all = rp.extract_raw_data(
    query=query_index_all, 
    site_names=site_names, 
    source_data_path=source_data_path, 
    data_name='index_all',
    database_engine=database_engine)


del query_index_all
del index_all
'''))

    with open(os.path.join(main_path, "raw_data_extraction.ipynb"), 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print_in_box(f"A raw data extraction jupyter notebook was created at {main_path}/raw_data_extraction.ipynb")




def main():
    main_path, study_start_date, study_end_date, site_names = get_user_input()
    # print("="*150)
    print("")
    create_project_folder(main_path)
    # print("="*150)
    print("")
    create_config_file(main_path,study_start_date, study_end_date, site_names)
    # print("="*150)
    print("")
    get_code_base(main_path)
    # print("="*150)
    print("")
    create_raw_data_extraction_notebook(main_path)

if __name__ == "__main__":
    main()