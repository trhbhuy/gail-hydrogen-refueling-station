# src/solver/config.py

import os
import pandas as pd
import numpy as np

# Define base paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Data directories
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
GENERATED_DATA_DIR = os.path.join(DATA_DIR, 'generated')
TRAJECTORIES_DIR = os.path.join(DATA_DIR, 'trajectories')

# General parameters
T_NUM = 24  # 24 hours
T_SET = np.arange(T_NUM)
DELTA_T = 24 / T_NUM  # Time step in hours
PENALTY_COEFFICIENT = 100

# Grid exchange parameters
P_GRID_PUR_MAX = 500  # Maximum power purchase from grid (kW)
R_GRID_PUR = 300  # Ramp rate for grid purchase (kW/h)
P_GRID_EXP_MAX = 500  # Maximum power export to grid (kW)
R_GRID_EXP = 300  # Ramp rate for grid export (kW/h)
PHI_RTP = 0.8  # Real-time pricing factor

# Solar PV parameters
P_PV_RATE = 250  # Rated power of PV (kW)
N_PV = 0.167  # Efficiency factor for PV
PHI_PV = 0.14  # Loss factor for PV

# Electrolyzer (EZ) Parameters
LHV = 39.72  # Lower Heating Value (kWh/kg)
P_EZ_MAX = 400  # Maximum power input to electrolyzer (kW)
P_EZ_MIN = 25  # Minimum power input to electrolyzer (kW)
R_EZ = 300  # Ramp rate for electrolyzer (kW/h)
N_EZ = 0.65  # Efficiency of electrolyzer
K_EZ = 8.5
T_EZ = 10000
M_EZ = 0.003
Q_EZ = 0.15
G_EZ_MAX = P_EZ_MAX * N_EZ / LHV

# Fuel Cell (FC) Parameters
P_FC_MAX = 400  # Maximum power output of fuel cell (kW)
P_FC_MIN = 25  # Minimum power output of fuel cell (kW)
R_FC = 300  # Ramp rate for fuel cell (kW/h)
N_FC = 0.77  # Efficiency of fuel cell
K_FC = 32
T_FC = 10000
M_FC = 0.003
Q_FC = 0.02
G_FC_MAX = P_FC_MAX / (N_FC * LHV)

# FCEV Refueling Station Parameters
N_COMP = 0.8  # Compressor efficiency
Z_COMP = 2.7  # Compressor factor
PHI_FCEV = 4  # Loss factor for FCEV refueling

# Hydrogen Storage System (HSS) Parameters
SOP_HSS_MAX = 10  # Maximum state of pressure in HSS
SOP_HSS_MIN = 2  # Minimum state of pressure in HSS
DT_HSS = 0.006 / 100  # Time step change for HSS
SOP_HSS_SETPOINT = SOP_HSS_MAX  # Reference state of pressure for HSS

Y_HSS = (0.0001 * 8.314 * 313) / (200 * 0.002)  # Hydrogen factor for HSS

SOP_HSS_THRESHOLD = (1 + DT_HSS) * (SOP_HSS_MAX - DELTA_T * Y_HSS * P_EZ_MAX * N_EZ / LHV)

def create_directories():
    """
    Ensure that all necessary directories exist. If not, create them.
    """
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(GENERATED_DATA_DIR, exist_ok=True)

def print_config():
    """
    Utility function to print the current configuration settings.
    Useful for debugging and verification.
    """
    print("Hydrogen Refueling Station Configuration Settings:")
    print(f"Time horizon: {T_NUM} hours")
    print(f"Time step: {DELTA_T} hours")
    print(f"Grid purchase power: {P_GRID_PUR_MAX} kW")
    print(f"Grid export power: {P_GRID_EXP_MAX} kW")
    print(f"Max SOC for HSS: {SOP_HSS_MAX}")
    print(f"Min SOC for HSS: {SOP_HSS_MIN}")
    # Add more configuration details as needed

# Automatically create directories when the module is imported
create_directories()

if __name__ == "__main__":
    print_config()