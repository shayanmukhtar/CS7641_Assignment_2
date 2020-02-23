import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import scale


def process_census_data(path="./../Dataset/Census_Income"):
    file = path + "/adult.data"
    census_data = np.ndarray([32561, 15])
    row_counter = 0

    workclass = ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay',
                 'Never-worked', '?']
    education = ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th',
                 '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool', '?']
    marital = ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent',
               'Married-AF-spouse', '?']
    occupation = ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty',
                  'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving',
                  'Priv-house-serv', 'Protective-serv', 'Armed-Forces', '?']
    relationship = ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried', '?']
    race = ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black', '?']
    sex = ['Female', 'Male', '?']
    native_country = ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany',
                      'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran',
                      'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal',
                      'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia',
                      'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador',
                      'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands', '?']
    classification = ['<=50K', '>50K']

    with open(file, 'r') as f:
        lines = f.readlines()
        for row in lines:
            col_data = row.split(',')
            for col in range(0, len(col_data)):
                if col == 1:
                    census_data[row_counter, col] = workclass.index(col_data[col].strip(' '))
                elif col == 3:
                    census_data[row_counter, col] = education.index(col_data[col].strip(' '))
                elif col == 5:
                    census_data[row_counter, col] = marital.index(col_data[col].strip(' '))
                elif col == 6:
                    census_data[row_counter, col] = occupation.index(col_data[col].strip(' '))
                elif col == 7:
                    census_data[row_counter, col] = relationship.index(col_data[col].strip(' '))
                elif col == 8:
                    census_data[row_counter, col] = race.index(col_data[col].strip(' '))
                elif col == 9:
                    census_data[row_counter, col] = sex.index(col_data[col].strip(' '))
                elif col == 13:
                    census_data[row_counter, col] = native_country.index(col_data[col].strip(' '))
                elif col == 14:
                    census_data[row_counter, col] = classification.index(col_data[col].strip())
                else:
                    census_data[row_counter, col] = col_data[col]
            row_counter += 1
    # normalize the continuous rows
    census_data[:, 2] = scale(census_data[:, 2])
    census_data[:, 2] = np.nan_to_num(census_data[:, 2])

    census_data[:, 10] = scale(census_data[:, 10])
    census_data[:, 10] = np.nan_to_num(census_data[:, 10])

    census_data[:, 11] = scale(census_data[:, 11])
    census_data[:, 11] = np.nan_to_num(census_data[:, 11])

    census_data[:, 12] = scale(census_data[:, 12])
    census_data[:, 12] = np.nan_to_num(census_data[:, 12])

    # ret_data = panda_df.to_numpy()
    return census_data[:, :-2], census_data[:, -1]
