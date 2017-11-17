#!/usr/bin/env python

# This Python 3 environment comes with many helpful analytics
# libraries installed It is defined by the kaggle/python docker image:
# https://github.com/kaggle/docker-python For example, here's several
# helpful packages to load in

import math
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import titanic

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# from subprocess import check_output
# print(check_output(["ls", "../input"]).decode("utf8"))


# Any results you write to the current directory are saved as output.
input = pd.read_csv("train.csv")
titanic.clean_data(input)
titanic.mean_normalize(input)
input = titanic.remove_unused_columns(input)
print(input[0:10])
input.to_csv("normalized-train.csv", index=False)
