# main.py
import sys
from PyQt6 import QtWidgets, QtGui, QtCore
from music_player import MusicPlayer  # Importez la classe depuis le fichier music_player.py
import os

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    # Configuration alternative pour la haute résolution
    app.setHighDpiScaleFactorRoundingPolicy(QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    icon_path = os.path.join(os.path.dirname(__file__), "ICON", "app_icon.png")
    if os.path.exists(icon_path):
        app.setWindowIcon(QtGui.QIcon(icon_path))
    music_player = MusicPlayer()
    music_player.show()
    sys.exit(app.exec())
