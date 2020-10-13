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
COORD_FILE = DATA_DIR / 'raw_data' / 'CoordVehiclesRxPerScene_s008.csv'

MAX_TRAIN_EPISODE_ID = 1564

# ## Functions

def get_coordinates(filename, max_train_episode_id):
    coordinates_train = []
    coordinates_test = []
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            is_valid = row['Val'] #V or I are the first element of the line
            if is_valid == 'V': #check if the channel is valid
                episode_id = int(row['EpisodeID'])
                coord = [float(row['x']), float(row['y']), float(row['z'])]
                if episode_id <= max_train_episode_id:
                    coordinates_train.append(coord)
                else:
                    coordinates_test.append(coord)

    return coordinates_train, coordinates_test

# ## Data

coordinates_train, coordinates_val = get_coordinates(COORD_FILE, MAX_TRAIN_EPISODE_ID)
num_train = len(coordinates_train)

print("Number of train records:", num_train)

np.savez(DATA_DIR /'baseline_data' / 'coord_input' / 'my_coord_train.npz', coordinates=coordinates_train)
np.savez(DATA_DIR /'baseline_data' / 'coord_input' / 'my_coord_validation.npz', coordinates=coordinates_val)
