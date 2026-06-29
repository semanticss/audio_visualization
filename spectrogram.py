import numpy as np
import sounddevice as sd
import pyqtgraph.opengl as gl
from pyqtgraph.opengl.shaders import ShaderProgram, VertexShader, FragmentShader
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtGui import QVector3D
import sys
import librosa
import threading

AUDIO_FILEPATH = "music/weird_fishes.mp3"
STFT_WINDOWSIZE = 512
BLOCKSIZE = 2048  
OVERLAP = 0.5
HISTORY_CHUNKS = 200 
DATA_LOCK = threading.Lock()

# chat wrote this idek how this works
custom_vertex_shader = VertexShader("""
    uniform mat4 u_mvp;
    attribute vec4 a_position;
    varying float v_height;
    void main() {
        v_height = a_position.z; // Pass the raw Z height down to fragment processor
        gl_Position = u_mvp * a_position;
    }
""")

custom_fragment_shader = FragmentShader("""
    #ifdef GL_ES
    precision mediump float;
    #endif
    varying float v_height;
    void main() {
        // Map height bounds dynamically (0.0 to 35.0) to a normalized factor (0.0 to 1.0)
        float norm = clamp(v_height / 35.0, 0.0, 1.0);
        
        // Dynamic procedural color gradient: Blue (Silence) -> Green -> Red (Peaks)
        float r = smoothstep(0.4, 0.8, norm);
        float g = smoothstep(0.0, 0.5, norm) - smoothstep(0.6, 1.0, norm);
        float b = 1.0 - smoothstep(0.0, 0.4, norm);
        
        gl_FragColor = vec4(r, g, b, 1.0);
    }
""")

audio_height_shader = ShaderProgram('audioHeightShader', [custom_vertex_shader, custom_fragment_shader])

class Modeler():
    def __init__(self):
        self.filepath = AUDIO_FILEPATH
        self.audio_array, self.fs = librosa.load(self.filepath, sr=22050, mono=True)
        self.position = 0
        
        self.window = np.hamming(STFT_WINDOWSIZE)
        self.hop = int(STFT_WINDOWSIZE * OVERLAP)
        
        self.n_freq = STFT_WINDOWSIZE // 2 + 1
        self.history_chunks = HISTORY_CHUNKS
        
        self.freqs = np.linspace(0, self.fs / 2, self.n_freq)
        self.times = np.arange(self.history_chunks)
        self.buffer = np.zeros((self.n_freq, self.history_chunks))

    def process_chunk(self, chunk):
        if len(chunk) < STFT_WINDOWSIZE:
            return  
            
        stft_res = librosa.stft(chunk, n_fft=STFT_WINDOWSIZE, hop_length=self.hop, window='hamming')
        db = 20 * np.log10(np.abs(stft_res) + 1e-8)
        db_frame = np.mean(db, axis=1)

        with DATA_LOCK:
            self.buffer = np.roll(self.buffer, -1, axis=1)
            self.buffer[:, -1] = db_frame

    def callback(self, outdata, frames, t, status):
        if status:
            print(status)
            
        chunk = self.audio_array[self.position: self.position + frames]
        if len(chunk) < frames:
            outdata[:len(chunk), 0] = outdata[:len(chunk), 1] = chunk
            outdata[len(chunk):, :] = 0
            raise sd.CallbackStop
            
        outdata[:, 0] = outdata[:, 1] = chunk
        self.process_chunk(chunk)
        self.position += frames

    def start(self):
        self.stream = sd.OutputStream(callback=self.callback, blocksize=BLOCKSIZE, samplerate=self.fs, channels=2)
        self.stream.start()


class Renderer():
    def __init__(self, m: Modeler):
        self.m = m
        self.app = QtWidgets.QApplication(sys.argv)
        self.w = gl.GLViewWidget()
        self.axes = gl.GLAxisItem()
        self.axes.setSize(x=500, y=500, z=500)
        self.w.addItem(self.axes)
        self.w.setWindowTitle('bs')
        
        freq_len, history_len = m.buffer.shape
        self.w.opts['center'] = QVector3D(freq_len / 2, history_len / 2, 10) #type: ignore
        self.w.opts['distance'] = 200 #type: ignore
        self.w.opts['elevation'] = 50 #type: ignore
        self.w.opts['azimuth'] = 135 #type: ignore
        self.w.show()

        self.x = np.arange(freq_len)
        self.y = np.arange(history_len)
        z = np.zeros((freq_len, history_len))
        
        self.surface = gl.GLSurfacePlotItem(
            x=self.x, 
            y=self.y, 
            z=z, 
            computeNormals=False, 
            shader=audio_height_shader, 
            smooth=False
        )
        self.w.addItem(self.surface)

    def start(self):
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            sys.exit(self.app.exec())

    def update(self):
        with DATA_LOCK:
            z = self.m.buffer.copy()

        z = (z + 20) * .7
        self.surface.setData(z=z)

    def animation(self):
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(5)  
        self.start()


if __name__ == '__main__':
    m = Modeler()
    m.start()
    r = Renderer(m)
    r.animation()
