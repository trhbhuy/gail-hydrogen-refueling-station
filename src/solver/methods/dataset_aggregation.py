# src/solver/methods/dataset_aggregation.py

import numpy as np
from typing import Dict

def dataset_aggregation(records: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Process the loaded results into a single dataset for model training.

    Args:
        results (dict): Dictionary containing the optimization results.

    Returns:
        dict: A dictionary containing X_data (features) and y_data (target).
    """
    # Combine data_seq features
    data_seq = np.vstack([records['time_step'], records['rtp'].ravel(), records['p_pv_max'].ravel(), records['g_fcev_demand'].ravel(), records['sop_hss_prev']]).T

    # Define the target variable label
    label = np.vstack([records['g_ez'].ravel(), records['g_fc'].ravel(), records['g_fcev'].ravel()]).T

    # Create the dataset dictionary
    dataset = {
        'data_seq': data_seq,
        'label': label
    }

    return dataset