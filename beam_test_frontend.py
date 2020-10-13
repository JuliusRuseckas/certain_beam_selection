#!/usr/bin/env python3
# coding: utf-8

# # Processing of coordinate data

# ## Configuration

# Imports

import argparse
from pathlib import Path
import numpy as np
import csv

# Argument parsing

parser = argparse.ArgumentParser(description='Configure the files before training the net.')
parser.add_argument('data_folder', help='Location of the data directory', type=str)
args = parser.parse_args()

# Configuration

DATA_DIR = Path(args.data_folder)

def get_file_name(folder_name):
    ind = folder_name.find('_')
    assert ind > -1
    s_name = folder_name[ind+1:].lower()
    return f'CoordVehiclesRxPerScene_{s_name}.csv'


COORD_FILE = DATA_DIR / 'raw_data' / get_file_name(DATA_DIR.name)

# ## Functions

def get_coordinates(filename):
    coordinates_test = []
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)        
        for row in reader:
            is_valid = row['Val'] #V or I are the first element of the line
            if is_valid == 'V': #check if the channel is valid
                episode_id = int(row['EpisodeID'])
                coord = [float(row['x']), float(row['y']), float(row['z'])]
                coordinates_test.append(coord)

    return coordinates_test

# ## Data

coordinates_test = get_coordinates(COORD_FILE)
num_test = len(coordinates_test)

print("Number of test records:", num_test)

np.savez(DATA_DIR /'baseline_data' / 'coord_input' / 'my_coord_test.npz', coordinates=coordinates_test)
