import numpy as np
import sounddevice as sd
import threading
from numpy.fft import rfft, rfftfreq
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

DURATION = 10. # ms
NUM_PLOTS = 4
fs = 44100
sd.default.samplerate = fs
WINDOW_SECONDS = 0.2
window_size = int(fs * WINDOW_SECONDS)
data_lock = threading.Lock()
plot_data = np.zeros((NUM_PLOTS, int(window_size)))
print(plot_data[3])

file_path = "bodysnatchers.mp3"

def callback(indata, outdata, frames, time, status):
    global plot_data
    if status:
        print(status)
    outdata[:] = indata
    fft_data = rfft(indata[:, 0]) # this returns the raw FFT naturally, which is an array of complex numbers
    freq_space = rfftfreq(frames, d=1/fs) 
    magnitudes = np.abs(fft_data) # get the "amplitudes" of each of the curves
    max_args = np.argsort(magnitudes)[-3:] # get the indices of the maximum amplitudes
    freqs = [freq_space[max_args[i]] for i in range(3)] # get the corresponding frequencies
    amplitudes = [magnitudes[max_args[i]] for i in range(3)] # get the corresponding amplitudes
    phases = [np.angle(fft_data[max_args[i]]) for i in range(3)] # get the corresponding phase shifts
    t = np.arange(window_size) / fs
    with data_lock:
        n = frames
        plot_data[0] = np.roll(plot_data[0], -n)
        plot_data[0, -n:] = indata[:, 0]
        for i in range(1,4):
            plot_data[i] = amplitudes[i - 1] * np.sin(2 * np.pi * freqs[i - 1] * t + phases[i - 1]) * 2 / len(freq_space)
        

app = QtWidgets.QApplication([])
win = pg.GraphicsLayoutWidget(title="bs")
curves = []

audio_plot = win.addPlot(row=0, col=0) #type: ignore
audio_plot.setYRange(-0.1, 0.1)
audio_plot.setXRange(0, 200)
curves.append(audio_plot.plot(pen='g'))

colors = ['r', 'y', 'c']
for i in range(1, NUM_PLOTS):
    p = win.addPlot(row=i, col=0) #type: ignore
    p.setYRange(-0.1, 0.1)
    p.setXRange(0, 200)
    p.setLabel('top')
    curves.append(p.plot(pen=colors[i - 1]))

def update():
    with data_lock:
        snapshot = plot_data.copy()
    for i, curve in enumerate(curves):
        curve.setData(snapshot[i])

timer = QtCore.QTimer() # this is simply a digital timer object
timer.timeout.connect(update) # the idea is that the update function gets called every time the timer runs out it calls update and then the timer resets
timer.start(30) # ~60 fps

stream = sd.Stream(channels=(1, 1), samplerate=fs, callback=callback, blocksize=1000)
stream.start()

win.show()
app.exec()
