#!/usr/bin/env python

# This Python 3 environment comes with many helpful analytics
# libraries installed It is defined by the kaggle/python docker image:
# https://github.com/kaggle/docker-python For example, here's several
# helpful packages to load in

import math
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# from subprocess import check_output
# print(check_output(["ls", "../input"]).decode("utf8"))

dict = {
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
    return dict[embarked]

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

# Any results you write to the current directory are saved as output.
input = pd.read_csv("train.csv")
print("SIZE: %d %d" % input.shape)

input.drop(["PassengerId"], axis=1)

for key,val in conversions.items():
    src_key    = val[0]
    conversion = val[1]
    input[key] = input[src_key].apply(conversion)

print(input[0:10])
