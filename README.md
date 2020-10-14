# ITU Artificial Intelligence/Machine Learning in 5G Challenge
# ML5G-PHY [beam selection]
# Team CERTAIN

## 1. Environment

This submission uses baseline environment

## 2. Training the model

### 2.1 Feature extraction

Traing of the model uses lidar baseline features, located in the folder
`baseline_data/lidar_input/`. In addition, we extract GPS coordinates and save
to the files `my_coord_train.npz` and `my_coord_validation.npz` in the folder
`baseline_data/coord_input/`

```
python3 beam_train_frontend.py Raymobtime_s008
```

usage: beam_train_frontend.py [-h] data_folder

Configure the files before training the net.

positional arguments:
  data_folder  Location of the data directory

optional arguments:
  -h, --help   show this help message and exit


### 2.2 Training

This step trains the model and saves model to a JSON file `my_model.json` and
model weights to a HDF5 file `my_model_weights.h5`.  The statistics of GPS
coordinates of the training data set is saved to a file
`coord_train_stats.npz`

Before training additional features must be extracted as is described in step
2.1

The training script uses additional python code in the files `beam_utils.py`,
`utils.py` and `resnet.py`. Those files are included in the submission.

```
python3 beam_train_model.py Raymobtime_s008
```

usage: beam_train_model.py [-h] data_folder

Configure the files before training the net.

positional arguments:
  data_folder  Location of the data directory

optional arguments:
  -h, --help   show this help message and exit


## 3. Pre-trained weights

Files `my_model.json` containing model description and `my_model_weights.h5`
containing model weights are provided.


## 4. Testing the model

### 4.1 Feature extraction

Traing of the model uses lidar baseline features, located in the folder
`baseline_data/lidar_input/`. In addition, we extract GPS coordinates and save
to the file `my_coord_test.npz` in the folder `baseline_data/coord_input/`

```
python3 beam_test_frontend.py --dataset s009 Raymobtime_s009
```

usage: beam_test_frontend.py [-h] [--dataset DATASET] data_folder

Configure the files before training the net.

positional arguments:
  data_folder        Location of the data directory

optional arguments:
  -h, --help         show this help message and exit
  --dataset DATASET  Dataset name. Default s010


### 4.2 Testing

This steps test the model. Required files for testing are model description
'my_model.json', model weights 'my_model_weights.h5' and statistics of GPS
coordinates of the training data set 'coord_train_stats.npz' The testing
results are saved to the file 'beam_test_pred.csv'

Before testing additional features must be extracted as is described in step
4.1

```
python3 beam_test_model.py Raymobtime_s009
```

usage: beam_test_model.py [-h] data_folder

Configure the files before training the net.

positional arguments:
  data_folder  Location of the data directory

optional arguments:
  -h, --help   show this help message and exit

