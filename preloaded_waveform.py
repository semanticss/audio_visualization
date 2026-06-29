import numpy as np
import sounddevice as sd
import threading
from numpy.fft import rfft, rfftfreq
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
import librosa
from PyQt6.QtGui import QVector3D
import time
import sys
import pyqtgraph.opengl as gl

data_lock = threading.Lock()
class Modeler:
    def __init__(self):
        self.num_plots = 4
        self.window_seconds = 0.02
        self.file_path = "music/cecilia.mp3"
        self.audio_array, self.fs = librosa.load(self.file_path, sr=None) # sr=None keeps the sample rate of the original file
        self.window_size = int(self.fs * self.window_seconds)
        self.position = 0
        self.plot_data = np.zeros((self.num_plots, self.window_size))

    def process_chunk(self, chunk, n):

        fft_data = rfft(chunk)
        freq_space = rfftfreq(n, d=1 / self.fs)
        magnitudes = np.abs(fft_data)
        max_args = np.argsort(magnitudes)[-3:]
        freqs = [freq_space[max_args[i]] for i in range(3)]
        amplitudes = [magnitudes[max_args[i]] for i in range(3)]
        phases = [np.angle(fft_data[max_args[i]]) for i in range(3)]
        t = np.arange(self.window_size) / self.fs

        with data_lock:
            self.plot_data[0] = np.roll(self.plot_data[0], -n)
            self.plot_data[0, -n:] = chunk
            for i in range(1, self.num_plots):
                self.plot_data[i] = (
                    amplitudes[i - 1]
                    * np.sin(2 * np.pi * freqs[i - 1] * t + phases[i - 1])
                    * 2 / len(freq_space)
                )

    def callback(self, outdata, frames, time, status):
        if status:
            print(status)
        
        chunk = self.audio_array[self.position: self.position + frames]
        n = len(chunk)

        if n < frames:
            outdata[:n, 0] = chunk
            outdata[n:, 0] = 0
            self.process_chunk(chunk, n)
            self.position += n
            raise sd.CallbackStop
        
        outdata[:, 0] = outdata[:, 1] = chunk
        self.process_chunk(chunk, n)
        self.position += frames

    def start(self):
        self.stream = sd.OutputStream(callback=self.callback) # can specify blocksize here which would be electric
        self.stream.start()


class Visualizer: 
    def __init__(self, m: Modeler):
        self.traces = dict() # keys are integers entries are curves
        self.app = QtWidgets.QApplication(sys.argv)
        self.w = gl.GLViewWidget() # equivalent of a window, just a handle to display whatever u wanna display
        self.w.opts['distance'] = 1000 #type: ignore
        self.w.opts['center'] = QVector3D(220, 150, 0) #type: ignore
        self.w.opts['showAxes'] = True #type: ignore
        self.w.setWindowTitle('bs')
        self.w.setGeometry(500, 500, 1920, 1080) # x, y width of window, height of window
        self.w.show() # make it visible
        self.num_plots = m.num_plots
        self.axes = gl.GLAxisItem()
        self.axes.setSize(x=10, y=10, z=10)
        self.w.addItem(self.axes)
        

    def start(self):
            if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
                QtWidgets.QApplication.instance().exec() #type: ignore
        
    def set_plotdata(self, name, points, color, width):
            self.traces[name].setData(pos=points, color=color, width=width)
        
    def update(self):
            stime = time.time()
            if not hasattr(self, 'labels'):
                self.labels = dict()
            with data_lock:
                 snapshot = m.plot_data.copy()
            for i in range(self.num_plots):
                x = np.arange(snapshot.shape[1])
                y = np.full_like(x, i * 100)
                z = snapshot[i] * 25
                pts = np.vstack([x,y,z]).T

                if i not in self.traces:
                    self.traces[i] = gl.GLLinePlotItem()
                    self.w.addItem(self.traces[i])

                color = pg.glColor((i, self.num_plots * 1.3))

                self.set_plotdata( 
                name=i, 
                points=pts, 
                color=color,
                width=5
            ) 
                
                # why does this not work
                if i not in self.labels:
                     self.labels[i] = gl.GLTextItem()
                     self.w.addItem(self.labels[i])

                label_text = "Raw Chunk" if i == 0 else f"Freq Component {i}"
                self.labels[i].setData(text=label_text, color = color)
                self.labels[i].setData(pos=[-60., i * 100., 0.])


    def animation(self):
            timer = QtCore.QTimer()
            timer.timeout.connect(self.update)
            timer.start(10)
            self.start()

if __name__ == '__main__':
    m = Modeler()
    sd.default.samplerate = m.fs
    m.start()
    v = Visualizer(m)
    v.animation()