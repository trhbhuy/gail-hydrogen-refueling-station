# src/solver/methods/data_loader.py

import os
import pandas as pd
import numpy as np
from typing import Dict, Optional
from .. import config as cfg

def load_data(
    is_train: Optional[bool] = None,
    data_start_time: Optional[str] = None,
    data_end_time: Optional[str] = None,
    file_path: str = os.path.join(cfg.RAW_DATA_DIR, 'historical_data.csv')
) -> Dict[str, np.ndarray]:
    """Load the dataset into specific data sequences with defined time ranges based on training or testing period.
    
    Args:
        is_train (Optional[bool]): If True, load training data. If False, load testing data. If None, load all data.
        data_start_time (Optional[str]): Start time for filtering the data. Overrides is_train if provided.
        data_end_time (Optional[str]): End time for filtering the data. Overrides is_train if provided.
        file_path (str): Path to the CSV file containing the data.

    Returns:
        Dict[str, np.ndarray]: 
            - 'rtp': Real-time pricing (array)
            - 'p_pv_max': Solar power (array)
            - 'g_fcev_demand': Fuel-cell electric vehicle demand (array)
    """
    # Load the data
    dataframe = pd.read_csv(file_path, index_col='DateTime', parse_dates=True)

    # Determine the time range based on data_start_time, data_end_time, and is_train
    if data_start_time and data_end_time:
        # Use the provided custom time range
        data = dataframe.loc[data_start_time:data_end_time]
    elif is_train is None:
        # Consider the entire dataframe
        data = dataframe
    elif is_train:
        # Load training data
        data_start_time, data_end_time = '2018-01-01 00:00:00', '2019-12-31 23:00:00'
        data = dataframe.loc[data_start_time:data_end_time]
    else:
        # Load testing data
        data_start_time, data_end_time = '2020-01-01 00:00:00', '2020-03-31 23:00:00'
        data = dataframe.loc[data_start_time:data_end_time]

    # Store the data in a dictionary
    output_data = {
        'rtp': data['RTP'].values,
        'p_pv_max': data['PV Power'].values,
        'g_fcev_demand': data['FCEV demand'].values,
    }
    
    return output_data