import numpy as np
from typing import Dict

def feature_engineering(microgrid: object, records: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Calculate and add auxiliary variables such as time_step, net load, and previous SOC to the records.

    Args:
        microgrid (Microgrid): An instance of the Microgrid class containing relevant parameters.
        records (Dict[str, np.ndarray]): A dictionary containing optimization results.

    Returns:
        Dict[str, np.ndarray]: Updated records with additional auxiliary variables.
    """
    # Extract relevant parameters
    T_num = microgrid.T_num
    num_scenarios = len(records['ObjVal'])
    sop_hss_setpoint = microgrid.hss.sop_hss_setpoint

    # Create the time_step array (repeated for each scenario)
    records['time_step'] = np.tile(np.arange(T_num), num_scenarios)

    # Calculate the SOP of HSS at the previous time step
    sop_hss_flattened = records['sop_hss'].ravel()
    records['sop_hss_prev'] = np.roll(sop_hss_flattened, shift=1)
    records['sop_hss_prev'][0] = sop_hss_setpoint

    return records