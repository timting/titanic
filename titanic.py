from numpy import *
import csv

def read_data(input_file):
    labels = []
    data = []
    with open(input_file, "rb") as input:
        csv_reader = csv.reader(input)
        headers = next(csv_reader, None)
        for row in csv_reader:
            labels.append(row[0])
            data.append(row[1:])
    return array(data, dtype=float),labels
