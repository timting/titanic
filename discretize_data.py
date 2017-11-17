#!/usr/bin/env python

import pandas as pd
import titanic

for filename in ["train.csv", "test.csv"]:
    input = pd.read_csv(filename)
    titanic.clean_data(input)
    titanic.discretize_data(input)
    input = titanic.remove_unused_columns(input)
    print(input[0:10])
    input.to_csv("discretized-%s" % filename, index=False)

