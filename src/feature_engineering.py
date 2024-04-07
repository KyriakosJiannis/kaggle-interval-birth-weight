"""
Script is designed for data preprocessing, tailored for a machine learning (ML) pipeline#
It t includes a series of steps to clean, encode, and transform the input data before it is fed into an ML model
"""

import pandas as pd
import numpy as np

from src.labeling_encoding import keep_as_it, imputation_custom, categorical_encoding, numerical_to_grouped_encoding, \
    bi_encoding, string_encoding, missing_values_encoding


# from src.labeling_encoding_group import keep_as_it, imputation_custom, categorical_encoding, \
#    numerical_to_grouped_encoding, \
#    bi_encoding, string_encoding, missing_values_encoding


def calculate_pregnancy_length(input_data, dob_mm_col='DOB_MM', dlmp_mm_col='DLMP_MM'):
    """
    Calculate the pregnancy length based on the Date of Birth month and
    the Last Menstrual Period month.
    https://www.kaggle.com/code/paddykb/lgbm-mapie-birth-weight-oh-my
    """
    # Initial calculation of pregnancy length, handling invalid DLMP_MM values
    input_data['PREG_LEN'] = np.where(input_data[dlmp_mm_col] != 99, input_data[dob_mm_col] - input_data[dlmp_mm_col],
                                      np.nan)

    # Adjusting for negative pregnancy lengths by adding 12 months
    input_data['PREG_LEN'] = np.where(input_data['PREG_LEN'] < 0, input_data['PREG_LEN'] + 12, input_data['PREG_LEN'])

    # Setting pregnancy length to NaN if <= 4 months
    # input_data.loc[input_data['PREG_LEN'] <= 4, 'PREG_LEN'] = np.nan

    # Capping pregnancy length at 9 months if it's > 9 months
    # input_data.loc[input_data['PREG_LEN'] > 9, 'PREG_LEN'] = 9

    return input_data


def data_pre_process(input_data):
    """
    Preprocesses input data for a ML.

    """

    input_data = input_data.copy()

    # List of columns to drop from the input data

    drop_cols = [
        "PAY",
        # "PAY_REC",
        #"FEDUC",
        #"FAGECOMB",
        "RF_CESAR",
        # "RF_CESARN",
        'NO_MMORB',
        #'NO_INFEC'
    ]

    # Define feature sets for different preprocessing treatments
    feature_sets = {
        'keep': ["PAY_REC", "FEDUC", "FAGECOMB", "RF_CESARN", "PREG_LEN", "NO_INFEC"],
        'set_1': ['DOB_MM', 'DOB_WK', 'NO_RISKS', 'MAGER', 'PREG_LEN'],
        'set_2': ['DOB_TT'],
        'set_3': ['ATTEND', 'BFACIL', 'FEDUC', 'MBSTATE_REC', 'MEDUC', 'RDMETH_REC',
                  'RESTATUS'],
        'set_4': ['CIG_0'],
        'set_5': ['LD_INDL', 'SEX'],
        'set_6': ['DMAR'],
        'set_7': ['BMI', 'DLMP_MM', 'ILLB_R', 'ILOP_R', 'ILP_R', 'M_Ht_In', 'PRECARE', 'PREVIS', 'PRIORDEAD',
                  'PRIORLIVE', 'PRIORTERM', 'PWgt_R', 'RF_CESARN', 'WTGAIN']
    }

    # Drop specified columns
    input_data.drop(columns=drop_cols, inplace=True)

    # Apply specific preprocessing functions to each set of features
    data_sets = {
        key: globals()[f"{func_name}"](input_data, features=features)
        for key, (func_name, features) in zip(
            feature_sets.keys(),
            [
                ('keep_as_it', feature_sets['set_1']),
                ('imputation_custom', feature_sets['set_2']),
                ('categorical_encoding', feature_sets['set_3']),
                ('numerical_to_grouped_encoding', feature_sets['set_4']),
                ('bi_encoding', feature_sets['set_5']),
                ('string_encoding', feature_sets['set_6']),
                ('missing_values_encoding', feature_sets['set_7']),
            ],
        )
    }

    # Concatenate preprocessed features and 'DBWT' column
    if 'DBWT' in input_data.columns:
        out_data = pd.concat([data_sets[key] for key in data_sets] + [input_data[['DBWT']]], axis=1)
    else:
        out_data = pd.concat([data_sets[key] for key in data_sets], axis=1)

    # Replace spaces in column names with underscores
    out_data.columns = out_data.columns.str.replace(' ', '_')

    # Convert bool columns to int
    bool_cols = out_data.select_dtypes(include='bool').columns
    out_data[bool_cols] = out_data[bool_cols].astype(int)

    # Generate list of processed features for output
    processed_features = sum(feature_sets.values(), [])

    return out_data, processed_features


def main(input_filepath):
    try:
        # Load data
        data = pd.read_csv(input_filepath)
        # Preprocess data
        data = calculate_pregnancy_length(data)
        preprocessed_data, processed_features = data_pre_process(data)
        print(f"Processed features: {processed_features}")

    except Exception as e:
        print(f"Error during preprocessing: {e}")


if __name__ == '__main__':
    input_filepath = 'https://filedn.com/lK8J7mCaIwsQFcheqaDLG5z/data/Kaggle-interval-birth/train.csv'
    # Execute the main function
    main(input_filepath)
