import os
import math
import time
from PySide6.QtCore import (
    QObject, Signal, Slot, QThread, QTimer,
    QUrl, QCoreApplication, QMetaObject
)
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QComboBox
)
from PySide6.QtGui import QPixmap
from PySide6.QtNetwork import QNetworkAccessManager, QNetworkRequest, QNetworkReply

# ---------------------------------------------------------------------------
# TileFetcher: lives entirely in its own QThread
# ---------------------------------------------------------------------------

class TileFetcher(QObject):
    '''TMS caching tile retreiver 

    Remarks
    -------
    - Listens for current aimpoint (tile index)
    - Sorts pending tiles by distnce from current aimpoint
    - Uses an on-disk cache of tiles
    - Emits tileReady when a tile is loaded
    '''

    tileReady = Signal(int, int, int, bytes)

    def __init__(self, cache_dir:str = "cache", url_template:str='', parent=None):
        '''
        Parameters
        ----------
        cache_dir : str 
            Path to a directory for tile storage
        '''
        super().__init__(parent)
        self.cache_dir = cache_dir
        self.url_template = url_template
        os.makedirs(self.cache_dir, exist_ok=True)
        self.pending = []
        self.active = {}
        self.requested = set()
        self.aimpoint = (0, 0, 0)
        self.running = False

        self.timer = None
        self.nam = None

        self.user_agent = b"Mozilla/5.0 (TileFetcher PyQt Example)"

    @Slot()
    def start(self)->None:
        '''Initializes child objects, including dispatch timer and NetworkAccessManager'''
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._dispatch_next)
        self.timer.start(50)  # regulate requests
        self.nam = QNetworkAccessManager()
        self.running = True

    @Slot()
    def shutdown(self)->None:
        """Cleanly stop timers and pending operations."""
        print('hi brian')
        self.timer.stop()
        print ('blather')
        for reply in list(self.active.values()):
            reply.abort()
        self.active.clear()
        self.pending.clear()
        self.requested.clear()
        self.running = False

    @Slot(int, int, int)
    def setAimpoint(self, zoom:int, x:int, y:int) -> None:
        """Set the current aimpoint (center tile) for download prioritization.

        Parameters
        ----------
        zoom : int
            TMS Zoom level
        x : int
            TMS x
        y : int
            TMS y

        """
        self.aimpoint = (zoom, x, y)
        # reprioritize queue
        self.pending.sort(key=lambda item: self._tile_distance(item, self.aimpoint))

    @Slot()
    def reset(self):
        """Clear pending and active requests."""
        for reply in list(self.active.values()):
            reply.abort()
        self.pending.clear()
        self.active.clear()
        self.requested.clear()

    @Slot(int, int, int)
    def requestTile(self, z:int, x:int, y:int) -> None:
        """Queue or start a tile request.

        Parameters
        ----------
        z : int
            TMS z
        x : int
            TMS x
        y : int
            TMS y
        url_template : str
            URL to download tiles from
        """
        print (f"tile requested: {z}, {x}, {y}")
        cache_path = os.path.join(self.cache_dir, str(z), str(x), f"{y}.png")

        # already on disk?
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                data = f.read()
            self.tileReady.emit(z, x, y, data)
            return

        if (z, x, y) in self.requested:
            return

        print (f'self.url_template = {self.url_template}')

        self.requested.add((z, x, y))
        self.pending.append((z, x, y, self.url_template))
        self.pending.sort(key=lambda item: self._tile_distance(item, self.aimpoint))

    # ------------------------ private helpers ------------------------

    def _tile_distance(self, tile:tuple[int,int,int,int], aim: tuple[int,int,int])->int:
        '''Calculate distance from an aimpoint tile to current tile (for sorting)

        Parameters
        ----------
            tile: tuple[int,int,int,int]
                Tile in to test : [TMS Z, X, Y, template]
            aim : tuple[int,int,int]
                Current aimpoint [ TMS Z, X, Y ]

        Returns
        -------
            distance : int
                Distance from tile to aim
        '''
        z, x, y, _ = tile
        az, ax, ay = aim
        return abs(ax - x) + abs(ay - y) + (abs(az - z) * 4)

    @Slot()
    def _dispatch_next(self)->None:
        """If any downloaders are available, start a download"""
        #print ("_dipatch next")
        if len(self.active) >= 4 or not self.pending:
            #print (f"nope: {len(self.active)}, {self.pending}")
            return

        z, x, y, url_template = self.pending.pop(0)
        url = url_template.format(z=z, x=x, y=y)
        req = QNetworkRequest(QUrl(url))
        req.setRawHeader(b"User-Agent", self.user_agent)
        # This line
        reply = self.nam.get(req)
        reply.finished.connect(lambda: self._on_finished(reply, z, x, y))
        self.active[(z, x, y)] = reply

    def _on_finished(self, reply: QNetworkReply, z:int, x:int, y:int)->None:
        """Handle a response from tile server - cache the tile, emit tileReady

        Parameters
        ----------
        reply : QNetworkReply
            Object containing response to web request
        z : int
            TMS Z of requested tile
        x : int
            TMS X of requested tile
        y : int
            TMS y of requested tile
        """
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
        reply.deleteLater()


# ---------------------------------------------------------------------------
# Tile Manager Wrapper
# ---------------------------------------------------------------------------
class TileManager(QObject):
    """
    Manages a TileFetcher instance and its QThread, ensuring QNetworkAccessManager
    is created inside the worker thread and shutdown is clean.
    """
    tileReady = Signal(int, int, int, bytes)
    sigSetAimpoint = Signal(int, int, int)
    sigReset = Signal()
    sigRequestTile = Signal(int, int, int)
    sigStartFetcher = Signal()
    sigShutdown = Signal()

    def __init__(self, cache_dir : str = '', url_template : str = ''):
        super().__init__()
        self._thread = None
        self._fetcher = None

        self.set_fetcher(cache_dir, url_template)


    def set_fetcher(self, cache_dir, url_template):
        if self._thread is not None:
            print ("ERROR: already started")
            return

        # Thread and worker
        self._thread = QThread(self)

        self._fetcher = TileFetcher(cache_dir=cache_dir, url_template=url_template)


        # Move worker to thread (it will live in that thread after the thread starts)
        self._fetcher.moveToThread(self._thread)

        # Tile Ready Signal
        self._fetcher.tileReady.connect(self.tileReady)

        # Tile Request Signals
        self.sigSetAimpoint.connect(self._fetcher.setAimpoint)
        self.sigReset.connect(self._fetcher.reset)
        self.sigRequestTile.connect(self._fetcher.requestTile)

        # START AND STOP SIGNALS
        self.sigStartFetcher.connect(self._fetcher.start)
        self.sigShutdown.connect(self._fetcher.shutdown)

        # Thread lifecycle wiring
        self._thread.started.connect(self._on_thread_started)
        self._thread.finished.connect(self._on_thread_finished)


    @Slot()
    def _on_thread_started(self):
        """
        Ensure worker-owned objects (QNetworkAccessManager) are created inside the worker thread.
        We call the worker's initNetwork via a QueuedConnection so it runs inside the worker thread.
        """
        self.sigStartFetcher.emit()

    @Slot()
    def _on_thread_finished(self):
        """
        Cleanup when thread finished.
        """
        print("TileFetcherManager: thread finished; cleaning up worker")
        # Worker is still a QObject; schedule deletion
        self._fetcher.deleteLater()

    # Public API
    def start(self):
        """Start the thread and initialize the worker."""
        if not self._thread.isRunning():
            self._thread.start()
            print("TileFetcherManager: thread.start() called")

    def stop(self):
        """Stop the thread cleanly and wait for it to finish."""
        print ("tile manager stop")
        self.sigShutdown.emit()

        # Timed wait
        now = time.time() 
        while self._fetcher is not None and self._fetcher.running:
            time.sleep(0.1)
            if time.time() - now > 1:
                break

        if self._thread.isRunning():
            self._thread.quit()
            self._thread.wait()
            print("TileFetcherManager: thread.quit() / wait() completed")

    def sendReset(self):
        self.sigReset.emit()

    def setAimpoint(self, zoom, x, y):
        self.sigSetAimpoint.emit(zoom, x, y)

    def requestTile(self, zoom, x, y):
        self.sigRequestTile.emit(zoom, x, y)

# ---------------------------------------------------------------------------
# DEMO GUI FOR TEST
# ---------------------------------------------------------------------------
class TileViewer(QWidget):
    requestTile = Signal(int, int, int)
    setAimpoint = Signal(int, int, int)
    resetFetcher = Signal()

    def __init__(self, cache_dir, url_template):
        super().__init__()
        self.setWindowTitle("TileFetcher Demo")
        self.tile_manager = None
        layout = QVBoxLayout(self)
        self.label = QLabel("No tile yet")
        self.button = QPushButton('request tiles')
        self.map_combo = QComboBox()
        self.map_combo.addItems(
            ["OSM", "Blue Marble" ]
        )
        self.map_combo.currentIndexChanged.connect(self.on_map_combo)

        layout.addWidget(self.label)
        layout.addWidget(self.button)
        layout.addWidget(self.map_combo)


        # Set up the tile manager
        self.init_tile_manager(cache_dir, url_template)

        self.zoom = 3
        self.tile_x = 2
        self.tile_y = 3

        self.button.clicked.connect(self.changeAimpoint)

    def on_map_combo(self):
        txt = self.map_combo.currentText()
        if txt == 'OSM':
            cache_dir = 'osm'
            url_template = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
        else:
            cache_dir = 'bluemarble'
            url_template = "http://s3.amazonaws.com/com.modestmaps.bluemarble/{z}-r{y}-c{x}.jpg"
        self.init_tile_manager(cache_dir, url_template)

    def init_tile_manager(self, cache_dir, url_template):
        if self.tile_manager is not None:
            self.tile_manager.stop()
            del self.tile_manager

        self.tile_manager = TileManager(cache_dir, url_template)
        self.tile_manager.tileReady.connect(self.onTileReady)
        self.tile_manager.start()

    def changeAimpoint(self):
        """Simulate moving to a different tile center."""
        self.tile_x += 1
        self.tile_y += 1
        self.tile_manager.sendReset()
        self.tile_manager.setAimpoint(self.zoom, self.tile_x, self.tile_y)
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                self.tile_manager.requestTile(
                    self.zoom,
                    self.tile_x + dx,
                    self.tile_y + dy
                )

    @Slot(int, int, int, bytes)
    def onTileReady(self, z, x, y, data):
        print(f"Tile ready: {z}/{x}/{y}")
        pix = QPixmap()
        pix.loadFromData(data)
        self.label.setPixmap(pix.scaled(256, 256))

    def closeEvent(self, evt):
        print ('close event')
        self.tile_manager.stop()


def main():
    app = QApplication([])

    # Create GUI
    url_template = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
    cache_dir = 'cache'
    viewer = TileViewer(cache_dir, url_template)
    viewer.resize(300, 350)
    viewer.show()

    app.exec()

if __name__ == "__main__":
    main()
