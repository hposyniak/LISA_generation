from lisainstrument.instrument import Instrument
import h5py


# Create a simple simulation (5 minutes @ 1 Hz)
instrument = Instrument(
    size=1200,            
    dt=1/4,               
    t0="orbits",          
    orbits="static",      
    lock="N1-12",           
    fplan="static",       
    laser_asds=30, 
    clock_asds=6.32e-14 
)

instrument.simulate()


# Example: print or plot one signal
import matplotlib.pyplot as plt

instrument.plot_offsets()
instrument.plot_fluctuations()
instrument.plot_totals()
instrument.plot_mprs()
instrument.plot_dws()



instrument.write("my-file.h5")
