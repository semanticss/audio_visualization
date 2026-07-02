import PySide6
from PySide6.QtCore import Qt, QTimer, QElapsedTimer, QThread, Signal, Slot, QSize
from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QMainWindow, QApplication, QGraphicsPixmapItem, QGraphicsObject
from PySide6.QtGui import QImage, QKeyEvent, QMouseEvent, QPixmap, QCursor, QKeySequence

import sys
import numpy as np

class FDTD(QThread):

    x = Signal(np.ndarray, np.ndarray)

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
        self.border_mask = mask.ravel()

        self.obstacle_mask = np.ones(self.grid_dim, dtype=bool)

        self.mask = self.border_mask & self.obstacle_mask

        self.speed = 1 # speed scaling factor: the higher the speed, the worse the approximation becomes
        self.dt = np.sqrt(0.5) * self.speed

    def add_perturbation(self, row, col):
        press_idx = row * self.width + col
        if self.mask[press_idx]:
            self.curr_grid[press_idx] = 15.

    def add_obstacle(self, row, col, thickness):
        press_idx = row * self.width + col
        top_left = press_idx - thickness - (self.width * thickness)
        j = 0
        for i in range(0, thickness ** 2):
            self.obstacle_mask[top_left + i + j] = False
            self.mask[top_left + i + j] = False
            self.prev_grid[top_left + i + j] = 0.
            self.curr_grid[top_left + i + j] = 0.
            self.next_grid[top_left + i + j] = 0.
            if i != 0 and i % thickness == 0:
                j += self.width - thickness # i tried this and it worked idk why it works rly
            

    def clear_obstacles(self):
        self.obstacle_mask[:] = True

    def clear_waves(self):
        waves = self.mask | self.obstacle_mask
        self.prev_grid[waves] = 0.
        self.curr_grid[waves] = 0.
        self.next_grid[waves] = 0.

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
            self.next_grid *= .9999

            self.next_grid[~self.obstacle_mask] = 0.
            self.next_grid[~self.mask] -= 0.

            self.prev_grid[:] = self.curr_grid
            self.curr_grid[:] = self.next_grid

            self.x.emit(self.curr_grid.copy(), self.obstacle_mask.copy())
            self.msleep(10)
        

    def stop(self):
        self.running = False
        self.wait()
        

class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("wave in a room")
        self.sim_width = 480 * 2
        self.sim_height = 320 * 2
        
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
        self.drawing = False
        self.setMouseTracking(True)
        self.obstacle_drawing_thickness = 5

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

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_D:
            self.drawing = True

        if event.key() == Qt.Key.Key_C:
            self.fdtd_thread.clear_obstacles()

        if event.key() == Qt.Key.Key_Q:
            self.fdtd_thread.clear_waves()

    def keyReleaseEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_D:
            self.drawing = False

    def mouseMoveEvent(self, event: QMouseEvent):
        view_pos = event.pos()
        scene_pos = self.mapToScene(view_pos)
        col = int(scene_pos.x())
        row = int(scene_pos.y())
        if self.drawing:
            self.fdtd_thread.add_obstacle(row, col, self.obstacle_drawing_thickness)
    

    @Slot(np.ndarray, np.ndarray)
    def update_view(self, grid_data: np.ndarray, obstacle_mask: np.ndarray):
        max = 10.
        min = -max
        normalized = (255 * (grid_data - min) / (max - min)).astype(np.uint8)
        normalized = np.clip(normalized, 0, 255).astype(np.uint8) 
        grid_2d = grid_data.reshape((self.fdtd_thread.height, self.fdtd_thread.width))
        height, width = grid_2d.shape
        rgb_data = np.zeros((height, width, 4), dtype=np.uint8) # screen has 2 dimensions, color has 4 where 4th is opacity or smth

        waves = normalized.reshape((height, width))
        obstacles = obstacle_mask.reshape((height, width))

        # set background color to whatever
        rgb_data[..., 0] = 255
        rgb_data[..., 1] = 255
        rgb_data[..., 2] = 255
        rgb_data[..., 3] = 255

        wave_mask = waves > 127 # not 0 because of the normalization
        rgb_data[wave_mask, 0] = 128
        rgb_data[wave_mask, 1] = 128
        rgb_data[wave_mask, 2] = 0
        rgb_data[wave_mask, 3] = 0
        rgb_data[~obstacles] = [255, 0, 0, 255]
        bpl = width * 4

        q_image = QImage(
            rgb_data.data,
            width, 
            height, 
            bpl, 
            QImage.Format.Format_RGBA8888
        )

        p = QPixmap.fromImage(q_image)
        self.pixmap.setPixmap(p)


if __name__ == "__main__":
    app = QApplication()
    window = Window()
    window.show()
    sys.exit(app.exec())