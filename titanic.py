from numpy import *
import csv
import math
import pandas as pd
import re

# Public: Read data from a parsed input file.
#
# input_file - data input file (i.e. "normalized-train.csv")
#
# Returns:
#  numpy array of data
#  numpy array of labels
#  array of feature-names corresponding to data columns
def read_data(input_file):
    feature_names = []
    labels = []
    data = []
    with open(input_file, "rb") as input:
        csv_reader = csv.reader(input)
        feature_names = next(csv_reader, None)
        for row in csv_reader:
            labels.append(row[0])
            data.append(row[1:])
    feature_names.remove("Survived")
    return array(data, dtype=float), array(labels), feature_names

embarked_codes = {
    "C": 0,
    "Q": 1,
    "S": 2,
    pd.np.nan: -1
}

titles = {
    'Miss.': 0,
    'Mme.': 1,
    'Rev.': 2,
    'Jonkheer.': 3,
    'Sir.': 4,
    'Mlle.': 5,
    'Mrs.': 6,
    'Capt.': 7,
    'Col.': 8,
    'Ms.': 9,
    'Mr.': 10,
    'Lady.': 11,
    'Dr.': 12,
    'the': 13,
    'Master.': 14,
    'Major.': 15,
    'Don.': 16,
    pd.np.nan: -1
}

def convert_embarked_to_int(embarked):
    return embarked_codes[embarked]

def honorific(name):
    last_name, rest_of_name = name.split(", ", 1)
    title, first_names = rest_of_name.split(' ', 1)
    return titles[title]

conversions = {
    "Sex":       [ "Sex",      lambda x: 0 if x == "male" else 1 ],
    "Ticket":    [ "Ticket",   lambda x: 0 if re.match("[A-Za-z]", x) else 1 ],
    "Age":       [ "Age",      lambda x: -1 if math.isnan(x) else x ],
    "Fare":      [ "Fare",     lambda x: int(x * 100) ],
    "Embarked":  [ "Embarked", convert_embarked_to_int ],
    "Honorific": [ "Name",     honorific ]
}

# Public: Converts raw data to data in which the information is
# converted to ints.
#
# input_data - raw input (pandas) data with the following columns:
#                 Survived: 0=no, 1=yes
#                 Pclass: 0-2
#                 Sex: 0=male, 1=female
#                 Age: -1=unknown, 0-80
#                 SibSp: 0-8 (# siblings & spouses)
#                 Parch: 0-6 (# parents & children)
#                 Ticket: 0=alphanumeric, 1=numeric
#                 Fare: 0-51300
#                 Embarked: 0=Cherbourg, 1=Queenstown, 2=Southampton
#                 Honorific: 0=Miss, 1=Mme, 2=Rev, 3=Jonkheer, 4=Sir, 5=Mlle,
#                   6=Mrs, 7=Capt, 8=Col, 9=Ms, 10=Mr, 11=Lady, 12=Dr,
#                   13=, 14=Master, 15=Major, 16=Don
#
# Returns nothing.  Changes data in-place.
def clean_data(data):
    for key,val in conversions.items():
        src_key    = val[0]
        conversion = val[1]
        data[key] = data[src_key].apply(conversion)

unused_columns = ["PassengerId", "Cabin", "Name"]

def remove_unused_columns(data):
    return data.drop(unused_columns, 1)

normalize_columns = ["Pclass", "Age", "SibSp", "Parch", "Fare", "Honorific"]

# Public: normalize data using mean normalization
#
# input_data: input pandas data
#
# Returns nothing.  Changes data in-place.
def mean_normalize(data):
    for norm_col in normalize_columns:
        mean_normalize_column(data, norm_col)

# Public: normalize column data using mean normalization
#
# input_data - input pandas data
# column_name - column to normalize
#
# Returns nothing.  Changes data in-place.
def mean_normalize_column(data, column_name):
    column = data[column_name]
    size = data.shape[0]

    col_max = column.max(1)
    col_min = column.min(1)
    col_mean = column.mean(1)

    divisor = col_max - col_min
    data[column_name] = column.apply(lambda x: (x - col_mean)/divisor)

# Public: Returns the decade for the age.
#
# age - Age of passenger
#
# Returns:
#  0: ages 0-9
#  1: ages 10-19
#  2: ages 20-29
#  3: ages 30-39
#  4: ages 40-49
#  ...
#  7: ages 70-79
def discretize_age(age):
    return -1 if age == -1 else int(age)/10

# Public: Returns (int) number of $100's paid for this ticket.
#
# fare - fare paid by passenger
#
# Returns:
#  0: fare $0.00-$99.99
#  1: fare $100.00-$199.99
#  2: fare $200.00-$299.99
#  3: fare $300.00-$399.99
#  4: fare $400.00-$499.99
#  5: fare $500.00-$599.99
def discretize_fare(fare):
    return int(fare)/10000

discretizers = {
    "Age": discretize_age,
    "Fare": discretize_fare,
    "SibSp": lambda x: 1 if x > 0 else 0,
    "Parch": lambda x: 1 if x > 0 else 0
    }

# Public: Make continuous columns discrete
#
# data: input pandas data (cleaned but not noramlized)
#
# Returns nothing.  Changes data in-place.
def discretize_data(data):
    for key, val in discretizers.items():
        data[key] = data[key].apply(val)
