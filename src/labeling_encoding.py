import pandas as pd
import numpy as np


def keep_as_it(input_data, features):
    return input_data[features]


def imputation_custom(input_data, features):
    if 'DOB_TT' in features:
        input_data['DOB_TT'].replace(9999, pd.NA, inplace=True)
        median_value = input_data['DOB_TT'].median()
        input_data['DOB_TT'].fillna(median_value, inplace=True)
    return input_data[features]


def categorical_encoding(input_data, features, feature_mappings=None):
    """
    Labelling and Hot-Encoding the Categorical Columns
    """
    feature_mappings = {
        'ATTEND': {
            1: "Doctor of Medicine",
            2: "Doctor of Osteopathy",
            3: "Certified Nurse Midwife",
            4: "Other Midwife",
            5: "Other",
            9: "Unknown or not stated"
        },
        'BFACIL': {
            1: "Hospital",
            2: "Freestanding Birth Center",
            3: "Home intended",
            4: "Home notintended",
            5: "Home unknown",
            6: "Clinic",
            7: "Doctor Office",
            8: "Other",
            9: "Unknown"
        },
        'FEDUC': {
            1: "8th grade or less",
            2: "9to12th grade",
            3: "High School graduate",
            4: "Some college credit",
            5: "Associate degree",
            6: "Bachelor degree",
            7: "Master degree",
            8: "Doctorate",
            9: "Unknown"
        },
        'MBSTATE_REC': {
            1: "born in US",
            2: "born not in US",
            3: "Unknown"
        },
        'MEDUC': {
            1: "8th grade or less",
            2: "9to12th grade",
            3: "High School graduate",
            4: "Some college credit",
            5: "Associate degree",
            6: "Bachelor degree",
            7: "Master degree",
            8: "Doctorate",
            9: "Unknown"
        },
        'NO_INFEC': {
            1: "Yes",
            2: "No",
            9: "Not Reported"
        },
        'NO_MMORB': {
            1: "Yes",
            2: "No",
            9: "Not Reported"
        },
        'PAY': {
            1: "Medicaid",
            2: "Private insurance",
            3: "Selfpay",
            4: "Indian Health Service",
            5: "CHAMPUS_TRICARE",
            6: "Other government",
            8: "Other",
            9: "Unknown"
        },
        'PAY_REC': {
            1: "Medicaid",
            2: "Private Insurance",
            3: "SelfPay",
            4: "Other",
            9: "Unknown"
        },
        'RDMETH_REC': {
            1: "Vaginal",
            2: "Vaginal after previous Csection",
            3: "Primary Csection",
            4: "Repeat Csection",
            5: "Vaginal birth",
            6: "Csection",
            9: "Unknown"
        },
        'RESTATUS': {
            1: "Resident",
            2: "Intrastate nonresident",
            3: "Interstate nonresident",
            4: "Foreign resident"
        }
    }

    # Apply mappings to each categorical feature
    for feature in features:
        if feature in feature_mappings:
            input_data[feature] = input_data[feature].map(feature_mappings[feature])

    # Create df with encoding
    output_data = input_data[features]
    output_data = pd.get_dummies(output_data, columns=features)

    return output_data


def numerical_to_grouped_encoding(input_data, features, feature_mappings=None):
    """
    Grouped in specific bins the numerical and apply On hot encoding
    """

    encoded_features = pd.DataFrame(index=input_data.index)

    feature_mappings = {
        'BMI': {
            'bin_edges': [-np.inf, 16.5, 18.5, 25, 30, 35, 40, 99, np.inf],
            'labels': ['ExtremeUnderweight', 'Underweight', 'Normalweight',
                       'Overweight', 'Obesity1', 'Obesity2',
                       'ExtremeObesity', 'Unknown']
        },
        'CIG_0': {
            'bin_edges': [-1, 0.1, 10, 20, 98.9, np.inf],
            'labels': ['Notsmoker', '1to10', '10to20', 'More20', 'Unknown']
        }
    }
    for feature in features:
        if feature in feature_mappings and feature in input_data.columns:
            # Get the bin edges and labels for the current feature.
            bin_edges = feature_mappings[feature]['bin_edges']
            labels = feature_mappings[feature]['labels']

            # Group the feature into bins and then apply one-hot encoding.
            binned_data = pd.cut(input_data[feature], bins=bin_edges, labels=labels, right=True)
            encoded_data = pd.get_dummies(binned_data, prefix=feature)

            # Concatenate the newly encoded data with the accumulated DataFrame.
            encoded_features = pd.concat([encoded_features, encoded_data], axis=1)
        else:
            # Optionally, handle or notify about missing features here.
            print(f"Warning: Feature '{feature}' is not in the input data or mappings.")

    return encoded_features


def bi_encoding(input_data, features, feature_mappings=None):
    """
    to 0-1
    """

    output_data = input_data[features].copy()

    feature_mappings = {
        'LD_INDL': {'Y': 1, 'N': 0},
        'SEX': {'F': 1, 'M': 0},
        'RF_CESAR': {'Y': 1, 'N': 0}
    }

    for feature in features:
        if feature in feature_mappings:
            mapping = feature_mappings[feature]
            output_data[feature] = output_data[feature].map(mapping)

    return output_data


def string_encoding(input_data, features, feature_mappings=None):
    """
    Re-label the string and On hot encoding, exceptions applied
    """

    output_data = input_data[features].copy()

    feature_mappings = {
        'DMAR': {
            '1': 'Married',
            '2': 'Unmarried',
            '': 'Unknown'}
    }

    # Apply mappings to each categorical feature
    for feature in features:
        if feature in feature_mappings:
            output_data[feature] = output_data[feature].replace(feature_mappings[feature])
            if feature == "DMAR":
                output_data['DMAR'] = output_data['DMAR'].str.strip().replace('', 'Unknown')

    # Create df with encoding
    output_data = pd.get_dummies(output_data, columns=features)

    return output_data


def missing_values_encoding(input_data, features, feature_mappings=None):
    """
    Replaces specified values in numerical features with NaN and creates a new binary variable for missing-ness.
    """

    feature_mappings = {
        'BMI': {
            'Missing': [99.9]
        },

        'DLMP_MM': {
            'Missing': [99]
        },

        "FAGECOMB": {
            'Unknown': [99]
        },

        'ILLB_R': {
            'Unknown': [999],
            'NotApplicated': [888]
        },

        'ILOP_R': {
            'Unknown': [999],
            'NotApplicated': [888]
        },

        'ILP_R': {
            'Unknown': [999],
            'NotApplicated': [888]
        },

        'M_Ht_In': {
            'Unknown': [99]

        },

        'PRECARE': {
            'Unknown': [99]
        },

        "PREVIS": {
            'Unknown': [99]
        },

        "PRIORDEAD": {
            'Unknown': [99]
        },

        "PRIORLIVE": {
            'Unknown': [99]
        },

        "PRIORTERM": {
            'Unknown': [99]
        },
        #
        "PWgt_R": {
            'Unknown': [999],
          #  '375': [375]
        },
        "RF_CESARN": {
            'Unknown': [99]
        },
        "WTGAIN": {
            'Unknown': [99],
           # 'Over': [88]
        },

    }

    output_data = pd.DataFrame()

    for feature, mappings in feature_mappings.items():
        if feature in features:
            all_special_values = []

            # Create binary variables for each category
            for category, values in mappings.items():
                category_variable_name = f"{feature}_{category.lower()}"
                output_data[category_variable_name] = input_data[feature].isin(values).astype(int)
                all_special_values.extend(values)

            # Replace all specified category values with NaN
            all_special_values = set(all_special_values)
            output_data[feature] = input_data[feature].replace(list(all_special_values), np.nan)

    return output_data


if __name__ == '__main__':
    pass
