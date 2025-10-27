import os
import math
from PyQt5.QtCore import (
    QObject, pyqtSignal, pyqtSlot, QThread, QTimer,
    QUrl, QCoreApplication, QMetaObject
)
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtNetwork import QNetworkAccessManager, QNetworkRequest, QNetworkReply

# ---------------------------------------------------------------------------
# TileFetcher: lives entirely in its own QThread
# ---------------------------------------------------------------------------

class TileFetcher(QObject):
    '''TMS tile retreiver 

    Remarks
    -------
    - Listens for current aimpoint (tile index)
    - Sorts pending tiles by distnce from current aimpoint
    - Uses an on-disk cache of tiles
    - Emits tileReady when a tile is loaded
    '''

    tileReady = pyqtSignal(int, int, int, bytes)  # z, x, y, image data

    def __init__(self, cache_dir="cache", parent=None):
        super().__init__(parent)
        self.nam = QNetworkAccessManager()
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.pending = []
        self.active = {}
        self.requested = set()
        self.aimpoint = (0, 0, 0)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._dispatch_next)
        self.timer.start(50)  # regulate requests

        self.user_agent = b"Mozilla/5.0 (TileFetcher PyQt Example)"

    # ------------------------ slots (thread-safe) ------------------------

    @pyqtSlot()
    def shutdown(self)->None:
        """Cleanly stop timers and pending operations."""
        self.timer.stop()
        for reply in list(self.active.values()):
            reply.abort()
        self.active.clear()
        self.pending.clear()
        self.requested.clear()

    @pyqtSlot(int, int, int)
    def setAimpoint(self, zoom:int, x:int, y:int) -> None:
        """Set the current aimpoint (center tile)."""
        self.aimpoint = (zoom, x, y)
        # reprioritize queue
        self.pending.sort(key=lambda item: self._tile_distance(item, self.aimpoint))

    @pyqtSlot()
    def reset(self):
        """Clear pending and active requests."""
        for reply in list(self.active.values()):
            reply.abort()
        self.pending.clear()
        self.active.clear()
        self.requested.clear()

    @pyqtSlot(int, int, int, str)
    def requestTile(self, z, x, y, url_template):
        """Queue or start a tile request."""
        cache_path = os.path.join(self.cache_dir, str(z), str(x), f"{y}.png")

        # already on disk?
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                data = f.read()
            self.tileReady.emit(z, x, y, data)
            return

        if (z, x, y) in self.requested:
            return

        self.requested.add((z, x, y))
        self.pending.append((z, x, y, url_template))
        self.pending.sort(key=lambda item: self._tile_distance(item, self.aimpoint))

    # ------------------------ private helpers ------------------------

    def _tile_distance(self, item:tuple[int,int,int,int], aim: tuple[int,int,int])->int:
        '''Calculate distance from an aimpoint tile to current tile'''
        z, x, y, _ = item
        az, ax, ay = aim
        return abs(ax - x) + abs(ay - y) + (abs(az - z) * 4)

    def _dispatch_next(self)->None:
        """If any downloaders are available, start a download"""
        if len(self.active) >= 4 or not self.pending:
            return

        z, x, y, url_template = self.pending.pop(0)
        url = url_template.format(z=z, x=x, y=y)
        #print (f"get: {url}")
        req = QNetworkRequest(QUrl(url))
        req.setRawHeader(b"User-Agent", self.user_agent)
        reply = self.nam.get(req)
        reply.finished.connect(lambda: self._on_finished(reply, z, x, y))
        self.active[(z, x, y)] = reply

    def _on_finished(self, reply: QNetworkReply, z:int, x:int, y:int)->None:
        """Handle a response from tile server - cache the tile, emit tileReady"""
        reply.deleteLater()
        self.active.pop((z, x, y), None)

        if reply.error() == QNetworkReply.NetworkError.NoError:
            data = reply.readAll().data()
            cache_path = os.path.join(self.cache_dir, str(z), str(x))
            os.makedirs(cache_path, exist_ok=True)
            with open(os.path.join(cache_path, f"{y}.png"), "wb") as f:
                f.write(data)
            self.tileReady.emit(z, x, y, data)
        else:
            print(f"Tile {z}/{x}/{y} failed:", reply.errorString())

# ---------------------------------------------------------------------------
# GUI: demonstrates interaction
# ---------------------------------------------------------------------------

class TileViewer(QWidget):
    requestTile = pyqtSignal(int, int, int, str)
    setAimpoint = pyqtSignal(int, int, int)
    resetFetcher = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("TileFetcher Demo")
        layout = QVBoxLayout(self)
        self.label = QLabel("No tile yet")
        self.button = QPushButton("Change Aimpoint")
        layout.addWidget(self.label)
        layout.addWidget(self.button)

        self.zoom = 3
        self.tile_x = 2
        self.tile_y = 3

        self.button.clicked.connect(self.changeAimpoint)

    def changeAimpoint(self):
        """Simulate moving to a different tile center."""
        self.tile_x += 1
        self.tile_y += 1
        print(f"New aimpoint: {self.zoom}, {self.tile_x}, {self.tile_y}")
        self.setAimpoint.emit(self.zoom, self.tile_x, self.tile_y)
        self.resetFetcher.emit()
        # Request the 3×3 surrounding tiles
        url_template = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                self.requestTile.emit(
                    self.zoom,
                    self.tile_x + dx,
                    self.tile_y + dy,
                    url_template,
                )

    @pyqtSlot(int, int, int, bytes)
    def onTileReady(self, z, x, y, data):
        print(f"Tile ready: {z}/{x}/{y}")
        pix = QPixmap()
        pix.loadFromData(data)
        self.label.setPixmap(pix.scaled(256, 256))

# ---------------------------------------------------------------------------
# Main setup
# ---------------------------------------------------------------------------

def main():
    app = QApplication([])

    # Create fetcher + thread
    fetcher_thread = QThread()
    fetcher = TileFetcher()
    fetcher.moveToThread(fetcher_thread)
    fetcher_thread.start()

    # Create GUI
    viewer = TileViewer()
    viewer.resize(300, 350)
    viewer.show()

    # Connect signals safely (thread-aware)
    viewer.requestTile.connect(fetcher.requestTile)
    viewer.setAimpoint.connect(fetcher.setAimpoint)
    viewer.resetFetcher.connect(fetcher.reset)
    fetcher.tileReady.connect(viewer.onTileReady)

    app.aboutToQuit.connect(fetcher_thread.quit)
    app.aboutToQuit.connect(fetcher_thread.wait)
    app.exec_()

if __name__ == "__main__":
    main()

