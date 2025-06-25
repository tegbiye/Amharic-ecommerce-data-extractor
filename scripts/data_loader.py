import pandas as pd
import logging
import os


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def load_data(file, file_type='csv'):
    """
    Load data from a file into a pandas DataFrame.

    Parameters:
    - file: str, path to the file
    - file_type: str, type of the file ('csv', 'json', 'excel')

    Returns:
    - DataFrame containing the loaded data
    """
    if not os.path.exists(file):
        logging.error(f"File {file} does not exist.")
        return None

    if file_type == 'csv':
        return pd.read_csv(file, encoding='utf-8')
    elif file_type == 'json':
        return pd.read_json(file, encoding='utf-8')
    elif file_type == 'excel':
        return pd.read_excel(file, encoding='utf-8', engine='openpyxl')
    else:
        logging.error(f"Unsupported file type: {file_type}")
        return None


def label_content(row):
    if pd.notnull(row['Message']) and pd.notnull(row['Media Path']):
        return 'Both'
    elif pd.notnull(row['Message']):
        return 'Text Only'
    elif pd.notnull(row['Media Path']):
        return 'Media Only'
    else:
        return 'None'
