# Generative adversarial imitation learning for hydrogen refueling station scheduling (GAIL - HRS)

This repository contains the implementation for our paper: ["Robust real-time energy management for a hydrogen refueling station using generative adversarial imitation learning"](https://doi.org/10.1016/j.apenergy.2024.123847), published in the Applied Energy.

<!-- ## Environment 

- tensorflow: 2.0
- torch: 1.9 -->

<!-- ## Dataset
We opensource in this repository the model used for the ISO-NE test case. Code for ResNetPlus model can be found in /ISO-NE/ResNetPlus_ISONE.py

The dataset contains load and temperature data from 2003 to 2014. -->

## Setup 

```bash
conda env create -n torchtf --file env.yml
conda activate torchtf
```


## Structure

```bash
.
├── data/                                  # Directory for data
│   ├── generated/
│   ├── processed/
│   ├── raw/
│   └── trajectories/
└── src/
    ├── solver/
    │   ├── methods/
    │   │   ├── data_loader.py
    │   │   ├── dataset_aggregation.py
    │   │   ├── feature_engineering.py
    │   │   ├── optimization.py
    │   │   └── util.py
    │   ├── platform/
    │   │   ├── components/                # Components of the microgrid platform
    │   │   │   ├── __init__.py
    │   │   │   ├── compressor.py
    │   │   │   ├── electrolyzer.py
    │   │   │   ├── fuel_cell.py
    │   │   │   ├── hydrogen_storage.py
    │   │   │   ├── renewables.py
    │   │   │   └── utility_grid.py
    │   │   ├── env.py                     # Hydrogen staion environment setup and management (for training & testing)
    │   │   ├── hydrogen_station.py        # Microgrid optimization logic (for data generation)
    │   │   ├── trajs_env.py
    │   │   └── util.py
    │   ├── utils/
    │   │   ├── __init__.py
    │   │   ├── file_util.py
    │   │   └── numeric_util.py
    │   ├── __init__.py
    │   └── config.py                      # Configuration file for parameters
    ├── utils/                             # High-level utility scripts
    │   ├── __init__.py
    │   ├── common_util.py
    │   ├── preprocessing_util.py
    │   ├── test_util.py
    │   └── train_util.py
    ├── convert_trajs.py                   # Convert trajectories scripts
    ├── data_generation.py                 # Data generation scripts
    ├── preprocessing.py                   # Data preprocessing scripts
    ├── test_gail.py                       # Model testing scripts
    └── train_gail.py                      # Model training scripts
```


## How to run

### 1. Data Generation
To generate the necessary data for training and testing:

```
python3 data_generation.py
```

### 2. Data Preprocessing
Prepare the data for training by running the preprocessing script:

```
python3 preprocessing.py
```

### 3. Convert trajectories
Convert the collected trajectories into the required format:

```
python3 convert_trajs.py
```

### 4. Training the ResnesD Model
Train the ResnesD model using the generated data:

```
python3 train_gail.py --adversarial_algo gail --demo_batch_size 1024 \\
                      --gen_replay_buffer_capacity 512 --n_disc_updates_per_round 8 \\
                      --gen_algo ppo --policy MlpPolicy \\
                      --batch_size 64 --ent_coef 0.0 \\
                      --learning_rate 0.0004 --gamma 0.95 --n_epochs 5 \\
```

### 5. Testing the Model
Test the trained model on the microgrid environment:

```
python3 test_gail.py --env microgrid \\
                      --num_test_scenarios 91 \\
                      --adversarial_algo gail \\
                      --gen_algo ppo \\
                      --learning_rate 0.00015 \\
                      --batch_size 8 \\
                      --n_steps 2048 --verbose \\
```

## Citation
If you find the code useful in your research, please consider citing our paper:
```
@article{Huy2024,
   author = {Truong Hoang Bao Huy and Nguyen Thanh Minh Duy and Pham Van Phu and Tien-Dat Le and Seongkeun Park and Daehee Kim},
   doi = {10.1016/j.apenergy.2024.123847},
   issn = {03062619},
   journal = {Applied Energy},
   month = {11},
   pages = {123847},
   title = {Robust real-time energy management for a hydrogen refueling station using generative adversarial imitation learning},
   volume = {373},
   year = {2024},
}
```
## License
[MIT LICENSE](LICENSE)