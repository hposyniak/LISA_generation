import pandas as pd
import numpy as np
import scipy.signal
import h5py
import os
import re
import pickle

from scipy.signal import firwin, lfilter
from lisainstrument import Instrument
from pytdi import Data
from pytdi.michelson import X2, Y2, Z2
from lisaorbits import EqualArmlengthOrbits
from ldc.lisa.noise import get_noise_model
from ldc.waveform.waveform import HpHc
from lisagwresponse import ReadStrain


simlen = 5000 # simulation length, amount of time steps we want the simulation to have
skip_steps = 300 # number of steps to skip at the beginning of the simulation to allow the system to settle
samples_num = 3 # number of samples to generate, used for the seeds

fs = 4.0  # sampling rate (Hz), used for the antialiasing filter
cutoff_hz = 1.5 # cutoff frequency for the antialiasing filter (Hz)
num_taps = 103 # number of taps for the FIR filter 
nyquist = fs / 2.0 # nyquist frequency
filter_taps = firwin(num_taps, cutoff_hz / nyquist, window='hamming') # designs the FIR filter

# specifies all the output directories
output_dir_noise = "output/simulated_noise"
output_dir_gw = "output/simulated_gw"
os.makedirs(output_dir_noise, exist_ok=True)
os.makedirs(output_dir_gw, exist_ok=True)

# reads data from a csv file containg information on 15 massive black hole binaries 
df = pd.read_csv("data.csv")

orbits = EqualArmlengthOrbits() #generates the equal arm length orbits using the lisaorbits library
orbits.write("orbits.h5", mode = 'w') # generates an orbits file for the simulation

seeds = list(range(1, samples_num+1)) # list of seeds for the simulations, number of samples to generate

def create_pickle_file(h5_folder: str, output_pickle: str, dataset_key: str):
    """
    :param h5_folder: Directory containing .h5 files.
    :param output_pickle: Path where the aggregated data will be saved as a pickle file.
    :param dataset_key: Key within each .h5 file to extract data from (default is 'X').
    :return: Path to the saved pickle file.

    Creates a pickle file out of .h5 files
    """

   
    # extracts the numerical suffix from filenames
    def extract_number(filename):
        match = re.search(r'(\d+)', filename)
        return int(match.group(1)) if match else -1

    # gets a sorted list of .h5 files based on their numerical suffix
    h5_filenames = sorted(
        [f for f in os.listdir(h5_folder) if f.endswith('.h5')],
        key=extract_number
    )

    aggregated_data = []

    # loads the specified dataset from each file and appends it
    for fname in h5_filenames:
        file_path = os.path.join(h5_folder, fname)
        with h5py.File(file_path, 'r') as f:
            dataset = f[dataset_key][:]
            aggregated_data.append(dataset)

    # stacks all datasets into a single NumPy array
    final_array = np.stack(aggregated_data, axis=0)

    # saves the aggregated data as a pickle file
    with open(output_pickle, 'wb') as f:
        pickle.dump(final_array, f)

    print(f"Saved to {output_pickle}")


# --- Simulates LISA noise-only data (without gravitational wave signals) ---
for seed in seeds:
    instr = Instrument(
        size=simlen,
        lock='six',
        orbits='orbits.h5',
        seed=seed
    )
    instr.simulate() # runs the simulation

    temp_path = f"temp_noise_{seed:03}.h5"
    instr.write(temp_path, mode='w') # saves temporary simulation output

    noisedata = Data.from_instrument(temp_path) # loads the simulated data
    X_noise = X2.build(**noisedata.args)(noisedata.measurements) / instr.central_freq # builds and normalizes the TDI observable X

    X_trimmed = X_noise[skip_steps:] # skips the stabilization samples

    # saves the trimmed noise data to HDF5
    save_path = os.path.join(output_dir_noise, f"LISA_noise_{seed:03}.h5")
    with h5py.File(save_path, 'w') as hdf5:
        hdf5.create_dataset('X', data=X_trimmed)

# creates a pickle file from the noise-only simulation results
create_pickle_file(output_dir_noise, "output/simulated_noise.pkl", dataset_key='X')


# --- Simulates LISA data with injected gravitational wave signals ---
pMBHB_instances = df.to_dict(orient='records') # converts dataframe to list of dictionaries
hphc = HpHc.type("Test source", "MBHB", "IMRPhenomD")
hphc.set_param(pMBHB_instances[6]) # selects the 7th MBHB source from the dataset

# creates the time array for waveform generation
t_min = 0
t_max = pMBHB_instances[6]["CoalescenceTime"] + 1000
dt = 0.25
t = np.arange(t_min, t_max, dt)

# computes the plus and cross polarizations of the gravitational wave
hp, hc = hphc.compute_hphc_td(t)

# trims the strain to the simulation length
t_sim = t[-simlen:]
strain = ReadStrain(
    t_sim,
    hp[-simlen:],
    hc[-simlen:],
    gw_beta=pMBHB_instances[6]["EclipticLatitude"],
    gw_lambda=pMBHB_instances[6]["EclipticLongitude"],
    orbits='orbits.h5'
)

# saves the strain to an HDF5 file
strain.write(
    "gw.h5",
    t0=t_sim[0],
    dt=dt,
    size=len(t_sim),
    mode='w'
)

# simulates LISA data with the gravitational wave signal injected
for seed in seeds:
    instr = Instrument(
        size=simlen,
        t0=t_sim[0],
        orbits='orbits.h5',
        gws='gw.h5',
        lock='six',
        seed=seed
    )
    instr.simulate() # runs the simulation

    temp_path = f"temp_gws_{seed:03}.h5"
    instr.write(temp_path, mode='w') # saves the simulation output

    noisedata = Data.from_instrument(temp_path)
    measurements = noisedata.measurements

    # builds and normalizes the TDI observable X
    X_noise = X2.build(**noisedata.args)(noisedata.measurements) / instr.central_freq
    X_trimmed = X_noise[skip_steps:]

    # saves the trimmed data to HDF5
    save_path = os.path.join(output_dir_gw, f"LISA_gw_{seed:03}.h5")
    with h5py.File(save_path, 'w') as hdf5:
        hdf5.create_dataset('X', data=X_trimmed)

# creates a pickle file from the GW-injected simulation results
create_pickle_file(output_dir_gw, "output/simulated_gw.pkl", dataset_key='X')