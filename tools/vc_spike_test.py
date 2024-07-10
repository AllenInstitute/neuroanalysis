import numpy as np
import scipy.ndimage as ndi
import pyqtgraph as pg
from neuroanalysis.ui.plot_grid import PlotGrid
from neuroanalysis.spike_detection import detect_vc_evoked_spikes
from neuroanalysis.data import TSeries


# Load test data
data = np.load('test_data/evoked_spikes/vc_evoked_spikes.npz')['arr_0']
dt = 20e-6

# gaussian filtering constant
sigma = 20e-6 / dt

# Initialize Qt
pg.mkQApp()
pg.dbg()

# Create a window with a grid of plots (N rows, 1 column)
win = PlotGrid()
win.set_shape(data.shape[0], 1)
win.show()

# Loop over all 10 channels
for i in range(data.shape[0]):
    # select the data for this channel
    trace = data[i, :, 0]
    stim = data[i, :, 1]

    # select the plot we will use for this trace
    plot = win[i, 0]

    # link all x-axes together
    plot.setXLink(win[0, 0])
    xaxis = plot.getAxis('bottom')
    if i == data.shape[0]-1:
        xaxis.setLabel('Time', 's')
    else:
        xaxis.hide()

    # use stimulus to find pulse edges
    diff = np.diff(stim)   # np.diff() gives first derivative
    on_times = np.argwhere(diff > 0)[:,0]
    off_times = np.argwhere(diff < 0)[:,0]

    # decide on the region of the trace to focus on
    start = on_times[1] - 1000
    stop = off_times[8] + 1000
    chunk = trace[start:stop]

    # plot the selected chunk
    t = np.arange(chunk.shape[0]) * dt
    plot.plot(t[:-1], np.diff(ndi.gaussian_filter(chunk, sigma)), pen=0.5)
    plot.plot(t, chunk)

    # detect spike times
    peak_inds = []
    rise_inds = []
    for j in range(8):  # loop over pulses
        pstart = on_times[j+1] - start
        pstop = off_times[j+1] - start
        spike_info = detect_vc_evoked_spikes(TSeries(chunk, dt=dt), pulse_edges=(pstart, pstop))
        if spike_info is not None:
            peak_inds.append(spike_info['peak_index'])
            rise_inds.append(spike_info['rise_index'])

    # display spike rise and peak times as ticks
    pticks = pg.VTickGroup(np.array(peak_inds) * dt, yrange=[0, 0.3], pen='r')
    rticks = pg.VTickGroup(np.array(rise_inds) * dt, yrange=[0, 0.3], pen='y')
    plot.addItem(pticks)
    plot.addItem(rticks)
    
    
    
