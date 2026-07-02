import PySide6
from PySide6.QtCore import Qt, QTimer, QElapsedTimer, QThread, Signal, Slot, QSize
from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QMainWindow, QApplication, QGraphicsPixmapItem, QGraphicsObject
from PySide6.QtGui import QImage, QMouseEvent, QPixmap

import sys
import numpy as np

class Params():
    def __init__(self):
        self.c = 30
        self.dx = .1


class Obstacle(QGraphicsObject):
    def __init__(self):
        super().__init__()



class FDTD(QThread):

    x = Signal(np.ndarray)

    def __init__(self, width, height):
        super().__init__()
        self.width = width
        self.height = height
        self.grid_dim = self.width * self.height
        self.running = False
        self.prev_grid = np.zeros(self.grid_dim)
        self.curr_grid = np.zeros(self.grid_dim)
        self.next_grid = np.zeros(self.grid_dim)
        mask = np.ones((self.height, self.width), dtype=bool)
        mask[0, :] = False
        mask[-1, :] = False
        mask[:, 0] = False
        mask[:, -1] = False
        self.mask = mask.ravel()


        self.speed = 1 # speed scaling factor: the higher the speed, the worse the approximation becomes
        self.dt = np.sqrt(0.5) * self.speed

    def add_perturbation(self, row, col):
        idx = row * self.width + col
        self.curr_grid[idx] = 15.

    def run(self):
        self.running = True
        w = self.width
        idx = np.where(self.mask)[0]

        while self.running:
            temp = (
                self.curr_grid[idx - 1]
                + self.curr_grid[idx + 1]
                + self.curr_grid[idx - w]
                + self.curr_grid[idx + w]
            )

            self.next_grid[idx] = temp * .5 - self.prev_grid[idx]
            self.next_grid *= .99

            self.prev_grid[:] = self.curr_grid
            self.curr_grid[:] = self.next_grid

            self.x.emit(self.curr_grid.copy())
            self.msleep(16)
        

    def stop(self):
        self.running = False
        self.wait()
        

class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("wave in a room")
        self.sim_width = 480
        self.sim_height = 320
        
        self.fdtd_thread = FDTD(self.sim_width, self.sim_height)
        self.scene = QGraphicsScene(0, 0, self.sim_width, self.sim_height)
        self.view = View(self.scene, self.fdtd_thread)
        
        self.setCentralWidget(self.view)
        self.resize(self.sim_width + 2, self.sim_height + 2)
        
        self.fdtd_thread.start()

    def closeEvent(self, event):
        self.fdtd_thread.stop()
        event.accept()
        

class View(QGraphicsView):
    def __init__(self, scene, fdtd: FDTD):
        super().__init__(scene)

        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.pixmap = QGraphicsPixmapItem()
        self.fdtd_thread = fdtd
        self.fdtd_thread.x.connect(self.update_view)

        self.pixmap = QGraphicsPixmapItem()
        self.scene().addItem(self.pixmap)

    def wheelEvent(self, event):
        event.accept()

    def mousePressEvent(self, event: QMouseEvent):
        view_pos = event.pos()
        scene_pos = self.mapToScene(view_pos)

        col = int(scene_pos.x())
        row = int(scene_pos.y())

        if 0 <= col < self.fdtd_thread.width and 0 <= row < self.fdtd_thread.height:
            self.fdtd_thread.add_perturbation(row, col)

    @Slot(np.ndarray)
    def update_view(self, grid_data: np.ndarray):
        max = 2.
        min = -max
        normalized = (255 * (grid_data - min) / (max - min)).astype(np.uint8)
        grid_2d = grid_data.reshape((self.fdtd_thread.height, self.fdtd_thread.width))
        height, width = grid_2d.shape
        bpl = width

        q_image = QImage(
            normalized.data, 
            width, 
            height, 
            bpl, 
            QImage.Format.Format_Grayscale8
        )

        p = QPixmap.fromImage(q_image)
        self.pixmap.setPixmap(p)


if __name__ == "__main__":
    app = QApplication()
    window = Window()
    window.show()
    sys.exit(app.exec())