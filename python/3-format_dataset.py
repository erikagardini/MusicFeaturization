import numpy as np
import csv
import pandas as pd
import os

music_dic = {"classical": 1,
              "baroque": 2,
              "rock": 3,
              "opera": 4,
              "medieval": 5,
              "jazz": 6}

def fortmatData(array_data):

    rows = []

    #Get data
    for i in range(0, array_data.shape[0]):
        res = array_data[i, 0].split(".")
        genre = music_dic.get(res[0])
        aut_title = res[1]
        row = []
        row.append(genre)
        row.append(aut_title)
        for j in range(1, array_data.shape[1]):
            row.append(array_data[i, j])

        rows.append(row)

    return np.array(rows)


input_data = pd.read_csv('../results/music_dataset_testing.csv', header=None).values
output_data = fortmatData(input_data)
with open('../results/music_dataset.csv', 'w') as csv_file:
    writer = csv.writer(csv_file)
    for i in range(0, output_data.shape[0]):
        writer.writerow(output_data[i, :])