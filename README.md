# Real-time power scheduling for isolated microgrid using Dense Residual Neural Network (ResnesD - IMG)

This repository contains the implementation for our paper: ["Robust real-time energy management for a hydrogen refueling station using generative adversarial imitation learning"](https://doi.org/10.1016/j.apenergy.2024.123847), published in the Applied Energy.

<!-- ## Environment 

- tensorflow: 2.0
- torch: 1.9 -->

<!-- ## Dataset
We opensource in this repository the model used for the ISO-NE test case. Code for ResNetPlus model can be found in /ISO-NE/ResNetPlus_ISONE.py

The dataset contains load and temperature data from 2003 to 2014. -->

## Structure

```bash
.
├── data/                                  # Directory for data
│   ├── raw/                               # Raw input data
│   ├── processed/                         # Processed datasets
│   └── generated/                         # Generated datasets
└── src/
    ├── solver/
    │   ├── methods/
    │   │   ├── data_loader.py             # Data loading methods
    │   │   ├── dataset_aggregation.py     # Dataset aggregation logic
    │   │   ├── feature_engineering.py     # Feature engineering scripts
    │   │   ├── run_scenario.py            # Running scenarios or simulations
    │   │   └── util.py                    # Utility functions specific to methods
    │   ├── platform/
    │   │   ├── components/                # Components of the microgrid platform
    │   │   │   ├── __init__.py
    │   │   │   ├── compressor.py          # Compressor component logic
    │   │   │   ├── electrolyzer.py        # Electrolyzer component logic
    │   │   │   ├── fuel_cell.py           # Fuel cell component logic
    │   │   │   ├── hydrogen_storage.py    # Hydrogen storage component logic
    │   │   │   ├── renewables.py          # Renewable energy sources (PV, Wind, etc.)
    │   │   │   └── utility_grid.py        # Utility grid connection logic
    │   │   ├── hrs_env.py                 # Hydrogen staion environment setup and management (for training & testing)
    │   │   ├── hydrogen_station.py        # Microgrid optimization logic (for data generation)
    │   │   └── util.py                    # Utility functions for the platform
    │   ├── utils/                         # General utility functions
    │   │   ├── __init__.py
    │   │   ├── file_util.py               # File handling utilities
    │   │   └── numeric_util.py            # Numerical operations utilities
    │   ├── __init__.py
    │   └── config.py                      # Configuration file for parameters
    ├── utils/                             # High-level utility scripts
    │   ├── __init__.py
    │   ├── common_util.py                 # Common utility functions
    │   ├── preprocessing_util.py          # Preprocessing utility functions
    │   ├── test_util.py                   # Utility functions for testing
    │   └── train_util.py                  # Utility functions for training
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

### 3. Training the ResnesD Model
Train the ResnesD model using the generated data:

```
python3 train_gail.py --adversarial_algo gail --demo_batch_size 1024 \\
                      --gen_replay_buffer_capacity 512 --n_disc_updates_per_round 8 \\
                      --gen_algo ppo --policy MlpPolicy \\
                      --batch_size 64 --ent_coef 0.0 \\
                      --learning_rate 0.0004 --gamma 0.95 --n_epochs 5 \\
```

### 4. Testing the Model
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