# Ce projet est sous licence MIT.
# Copyright (c) 2024 Nicolas Q.


import numpy as np
from scipy.fft import rfft
from scipy.io import wavfile
from PyQt6 import QtCore, QtGui, QtWidgets
from pygame import mixer
import wave
from PyQt6.QtWidgets import QLabel
import os
from PyQt6.QtCore import QPropertyAnimation
import settings_manager
from settings_manager import save_settings, load_settings

class SpectrumWorker(QtCore.QObject):
    spectrum_updated = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, audio_data, sample_rate, chunk_size, bars, window, freq_indices, freq_diffs, smoothing_factor, scale_factor, sync_delay=20):
        super().__init__()
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.bars = bars
        self.window = window
        self.freq_indices = freq_indices
        self.freq_diffs = freq_diffs
        self.smoothing_factor = smoothing_factor
        self.scale_factor = scale_factor
        self.is_running = True
        self.spectrum_data = np.zeros(self.bars)
        self.sync_delay = sync_delay  # Ajout du délai de synchronisation

    def run(self):
        if self.chunk_size is None:
            print("Erreur : la taille du chunk n'est pas définie.")
            return  # Sort si chunk_size est toujours None
        
        # Ajoute le délai initial pour synchroniser le spectre avec l'audio
        QtCore.QThread.msleep(self.sync_delay)
        while self.is_running:
            current_pos_ms = mixer.music.get_pos()
            if current_pos_ms == -1:
                QtCore.QThread.msleep(4)
                continue

            current_pos = current_pos_ms / 1000.0  # Conversion en secondes
            pos_samples = int(current_pos * self.sample_rate)
            if pos_samples >= len(self.audio_data):
                QtCore.QThread.msleep(4)
                continue

            chunk = self.audio_data[pos_samples:pos_samples + self.chunk_size]
            if len(chunk) == 0:
                QtCore.QThread.msleep(4)
                continue

            if len(chunk) < self.chunk_size:
                chunk = np.pad(chunk, (0, self.chunk_size - len(chunk)))

            new_spectrum = self.process_audio_data(chunk)
            self.spectrum_data = (
                self.spectrum_data * (1 - self.smoothing_factor) + new_spectrum * self.smoothing_factor
            )
            self.spectrum_updated.emit(self.spectrum_data)
            QtCore.QThread.msleep(4)  # Mise à jour toutes les 4 ms

    def stop(self):
        self.is_running = False

    def process_audio_data(self, data):
        windowed_data = data * self.window
        fft_data = np.abs(rfft(windowed_data))

        max_val = np.max(fft_data)
        if max_val > 0:
            fft_data /= max_val

        spectrum = np.add.reduceat(fft_data, self.freq_indices[:-1]) / self.freq_diffs
        spectrum = np.log10(spectrum * 9 + 1)
        spectrum = np.clip(spectrum * self.scale_factor, 0, 1)
        return spectrum

class AudioSpectrum(QtWidgets.QWidget):
    shared_gradient_cache = None

    def __init__(self, parent=None, chunk_size=16384):
        super().__init__(parent)
        self.setMinimumHeight(100)

        # Initialisation des paramètres
        self.settings = load_settings()  # Charge les paramètres depuis le fichier JSON

        # Initialisation des paramètres de spectre
        self.bars = 150
        self.bar_spacing = 2
        self.chunk_size = chunk_size
        self.window = np.hanning(self.chunk_size)
        self.scale_factor = 1
        self.smoothing_factor = 0.5
        self.spectrum_color = QtGui.QColor(self.settings.get("spectrum_color", "#FF0000"))  # Couleur par défaut rouge
        self.background_color = QtGui.QColor(self.settings.get("background_color", "#2A2A3A"))  # Couleur par défaut gris

        # Créer un QLabel pour la notification
        self.volume_notification = QLabel(self)
        self.volume_notification.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.volume_notification.setStyleSheet("""
            QLabel {
                background-color: rgba(0, 0, 0, 0.8);
                color: white;
                font-size: 16px;
                font-weight: bold;
                border-radius: 8px;
                padding: 10px;
            }
        """)
        self.volume_notification.setFixedSize(200, 50)
        self.volume_notification.setVisible(False)

        # Positionner au centre de la fenêtre
        self.volume_notification.move(
            self.width() // 2 - self.volume_notification.width() // 2,
            self.height() // 2 - self.volume_notification.height() // 2
        )

        # Données audio et de spectre
        self.spectrum_data = np.zeros(self.bars)
        self.sound_array = None
        self.sample_rate = None
        self.is_paused = True

        # Paramètres de visualisation
        self.bar_positions = None
        self.bar_widths = None
        self.update_gradient_cache()

        # Gestion des threads
        self.spectrum_worker_thread = None
        self.spectrum_worker = None

    def reset_spectrum_data(self):
        """Réinitialise les données du spectre à zéro."""
        self.spectrum_data = np.zeros(self.bars)
        self.update()  # Force le redessin immédiat

    def show_volume_notification(self, message):
        """Affiche une notification de volume avec une animation de zoom avant."""
        self.volume_notification.setText(message)

        # Repositionner dynamiquement la notification
        self.volume_notification.move(
            self.width() // 2 - self.volume_notification.width() // 2,
            self.height() // 2 - self.volume_notification.height() // 2
        )
        self.volume_notification.setVisible(True)

        # Animation de zoom avant
        if hasattr(self, 'volume_zoom_animation'):
            self.volume_zoom_animation.stop()
        self.volume_zoom_animation = QtCore.QPropertyAnimation(self.volume_notification, b"geometry")
        self.volume_zoom_animation.setDuration(300)
        self.volume_zoom_animation.setStartValue(self.volume_notification.geometry().adjusted(
            20, 10, -20, -10  # Réduit légèrement pour l'effet initial
        ))
        self.volume_zoom_animation.setEndValue(self.volume_notification.geometry())
        self.volume_zoom_animation.setEasingCurve(QtCore.QEasingCurve.Type.OutBack)
        self.volume_zoom_animation.start()

        # Planification pour cacher la notification après 2,5 secondes avec animation de disparition
        QtCore.QTimer.singleShot(2500, self.hide_volume_notification)

    def hide_volume_notification(self):
        """Cache la notification de volume avec une animation de zoom arrière."""
        if hasattr(self, 'volume_hide_animation'):
            self.volume_hide_animation.stop()
        self.volume_hide_animation = QtCore.QPropertyAnimation(self.volume_notification, b"geometry")
        self.volume_hide_animation.setDuration(300)
        self.volume_hide_animation.setStartValue(self.volume_notification.geometry())
        self.volume_hide_animation.setEndValue(self.volume_notification.geometry().adjusted(
            20, 10, -20, -10  # Réduit légèrement pour l'effet final
        ))
        self.volume_hide_animation.setEasingCurve(QtCore.QEasingCurve.Type.InBack)
        self.volume_hide_animation.start()
        self.volume_hide_animation.finished.connect(lambda: self.volume_notification.setVisible(False))
        
    def change_spectrum_color(self):
        """Ouvre une boîte de dialogue pour choisir la couleur du spectre."""
        if not hasattr(self, 'settings'):
            print("Erreur : 'settings' non défini.")
            return

        color_dialog = QtWidgets.QColorDialog(self)
        color_dialog.setWindowTitle("Choisissez la couleur du spectre")

        color_dialog.setCurrentColor(QtGui.QColor(self.settings.get("spectrum_color", "#FF0000")))
        if color_dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            color = color_dialog.selectedColor()
            if not isinstance(color, QtGui.QColor):
                print("Erreur : Couleur sélectionnée invalide.")
                return

            self.set_spectrum_color(color)
            self.settings['spectrum_color'] = color.name()  # Sauvegarde sous forme de chaîne
            try:
                save_settings(self.settings)
                print("Paramètres sauvegardés avec succès.")
            except Exception as e:
                print(f"Erreur de sauvegarde des paramètres : {e}")

    def update_gradient_cache(self):
        """Met à jour le cache de gradient partagé pour optimiser le dessin."""
        gradients = [
            QtGui.QLinearGradient(0, 0, 0, 1).setCoordinateMode(QtGui.QGradient.CoordinateMode.ObjectBoundingMode)
            for _ in range(self.bars)
        ]
        for i, gradient in enumerate(gradients):
            gradient.setColorAt(0, self.spectrum_color.lighter(120))
            gradient.setColorAt(1, self.spectrum_color.darker(120))
        AudioSpectrum.shared_gradient_cache = gradients

    def load_audio_file(self, file_path):
        """Charge le fichier audio et initialise le SpectrumWorker."""
        try:
            with wave.open(file_path, 'rb') as wav_file:
                self.sample_rate, data = wavfile.read(file_path)
            self.sound_array = self._process_audio_data(data)
            self.setup_frequency_bins()
            self.restart_spectrum_worker()
        except Exception as e:
            print(f"Erreur lors du chargement du fichier audio : {e}")

    def _process_audio_data(self, data):
        """Convertit les données audio en un tableau de float normalisé."""
        if data.ndim > 1:
            data = data[:, 0]
        return data.astype(np.float32) / (np.iinfo(data.dtype).max + 1.0)

    def setup_frequency_bins(self):
        """Initialise les bandes de fréquence pour le spectre."""
        min_freq = 20
        max_freq = self.sample_rate / 2
        epsilon = 1e-6

        self.freq_bins = np.geomspace(min_freq + epsilon, max_freq + epsilon, self.bars + 1)
        freqs = np.fft.rfftfreq(self.chunk_size, 1 / self.sample_rate)
        self.freq_indices = np.clip(np.searchsorted(freqs, self.freq_bins), 0, len(freqs) - 1)
        self.freq_diffs = np.diff(self.freq_indices)
        self.freq_diffs[self.freq_diffs == 0] = 1

    def restart_spectrum_worker(self):
        """Redémarre le SpectrumWorker avec les paramètres actuels sans perturber l'affichage."""
        # Sauvegarder l'état actuel
        current_spectrum = self.spectrum_data.copy()
        
        # Arrêter l'ancien worker
        self.stop_spectrum_worker()

        # Restaurer l'état visuel
        self.spectrum_data = current_spectrum
        self.update()

        # Démarrage d'un nouveau worker pour le spectre
        self.spectrum_worker_thread = QtCore.QThread()
        self.spectrum_worker = SpectrumWorker(
            audio_data=self.sound_array,
            sample_rate=self.sample_rate,
            chunk_size=self.chunk_size,
            bars=self.bars,
            window=self.window,
            freq_indices=self.freq_indices,
            freq_diffs=self.freq_diffs,
            smoothing_factor=self.smoothing_factor,
            scale_factor=self.scale_factor
        )
        
        # Configuration et démarrage du nouveau worker avec l'état préservé
        self.spectrum_worker.spectrum_data = current_spectrum  # Initialiser avec l'état actuel
        self.spectrum_worker.spectrum_updated.connect(self.update_spectrum_data)
        self.spectrum_worker.moveToThread(self.spectrum_worker_thread)
        self.spectrum_worker_thread.started.connect(self.spectrum_worker.run)
        self.spectrum_worker_thread.start()

    def stop_spectrum_worker(self):
        """Arrête le thread du SpectrumWorker."""
        if self.spectrum_worker:
            self.spectrum_worker.stop()
        if self.spectrum_worker_thread:
            self.spectrum_worker_thread.quit()
            self.spectrum_worker_thread.wait()
        self.spectrum_worker_thread = None
        self.spectrum_worker = None

    def update_spectrum_data(self, spectrum_data):
        self.spectrum_data = spectrum_data
        self.update()

    def set_paused(self, paused):
        """Définit l'état de pause pour le spectre audio."""
        self.paused = paused
        if not paused:
            self.restart_spectrum_worker()  # Reprend le calcul
        else:
            # Arrête seulement les calculs, pas l'affichage
            self.worker_running = False  # Arrêter le thread de calcul
    
    def resizeEvent(self, event):
        self._configure_bars()
        super().resizeEvent(event)

    def _configure_bars(self):
        """Configure les positions et largeurs des barres en fonction de la largeur du widget."""
        width = self.width()
        available_width = width - (self.bars - 1) * self.bar_spacing
        bar_width = max(1, available_width // self.bars)
        
        # Ajustement des pixels restants
        remaining_pixels = available_width - (bar_width * self.bars)
        self.bar_widths = np.full(self.bars, bar_width, dtype=int)
        self.bar_widths[:remaining_pixels] += 1

        # Position des barres
        self.bar_positions = np.cumsum(np.r_[0, self.bar_widths[:-1] + self.bar_spacing])
        
    def update_gradient_cache(self):
        """Met à jour le cache de gradient partagé pour optimiser le dessin."""
        gradients = []
        for _ in range(self.bars):
            gradient = QtGui.QLinearGradient(0, 0, 0, 1)
            gradient.setCoordinateMode(QtGui.QGradient.CoordinateMode.ObjectBoundingMode)
            gradient.setColorAt(0, self.spectrum_color.lighter(120))  # Couleur en haut de la barre
            gradient.setColorAt(1, self.spectrum_color.darker(120))  # Couleur en bas de la barre
            gradients.append(gradient)
        AudioSpectrum.shared_gradient_cache = gradients

    def update_chunk_size(self, chunk_size):
        """Met à jour la taille du chunk et redémarre le SpectrumWorker."""
        self.chunk_size = int(chunk_size)  # Conversion en entier pour s'assurer que c'est valide
        self.window = np.hanning(self.chunk_size)  # Régénérer la fenêtre avec le nouveau chunk_size

        if self.sample_rate is not None:
            self.setup_frequency_bins()

        # Redémarrer le SpectrumWorker avec la nouvelle valeur
        if self.spectrum_worker:
            self.spectrum_worker.stop()
            self.restart_spectrum_worker()

    def set_spectrum_color(self, color):
        """Met à jour la couleur du spectre et actualise le cache de gradient."""
        self.spectrum_color = color
        try:
            settings_manager.save_spectrum_color(color.name())  # Sauvegarde uniquement le nom
        except Exception as e:
            print(f"Erreur de sauvegarde de la couleur du spectre : {e}")
        self.update_gradient_cache()
        self.update()

    def set_background_color(self, color):
        """Met à jour la couleur de fond et rafraîchit l'affichage."""
        self.background_color = color
        self.update()

    def paintEvent(self, event):
        if self.bar_positions is None or self.bar_widths is None:
            print("Bar positions or widths not configured correctly. Configuring now.")
            self._configure_bars()

        painter = QtGui.QPainter(self)

        # Créer un chemin avec un rectangle arrondi pour toute la zone du spectre
        rounded_rect_path = QtGui.QPainterPath()
        corner_radius = 15  # Rayon des coins arrondis pour tout le widget
        rounded_rect_path.addRoundedRect(QtCore.QRectF(self.rect()), corner_radius, corner_radius)

        # Remplir le fond avec la couleur de fond en utilisant le chemin arrondi
        painter.fillPath(rounded_rect_path, self.background_color)

        # Appliquer un clip path pour restreindre le dessin à l'intérieur du rectangle arrondi
        painter.setClipPath(rounded_rect_path)

        if self.bar_positions is not None and self.bar_widths is not None:
            # Dessiner les barres du spectre
            for i, amplitude in enumerate(self.spectrum_data):
                if i >= len(self.bar_positions) or i >= len(self.bar_widths):
                    print(f"Index out of bounds: i={i}, len(bar_positions)={len(self.bar_positions)}, len(bar_widths)={len(self.bar_widths)}")
                    break

                bar_h = amplitude * self.height()
                y = self.height() - bar_h

                rect = QtCore.QRectF(self.bar_positions[i], y, self.bar_widths[i], bar_h)
                painter.fillRect(rect, AudioSpectrum.shared_gradient_cache[i])

class FullScreenSpectrum(QtWidgets.QWidget):
    """Fenêtre pour afficher le spectre en plein écran avec animation améliorée."""

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.setWindowTitle("Mode Plein Écran")
        self.setWindowFlags(QtCore.Qt.WindowType.Window)
        self.setStyleSheet("background-color: black;")

        # Configuration du layout principal
        main_layout = QtWidgets.QVBoxLayout(self)

        # Déplacer temporairement le spectre principal dans cette fenêtre
        self.spectrum = parent.spectrum
        main_layout.addWidget(self.spectrum)

        self.setLayout(main_layout)

        # Créer les layouts pour les boutons
        controls_layout = QtWidgets.QHBoxLayout()
        settings_layout = QtWidgets.QHBoxLayout()

        # Boutons de contrôle
        self.previous_button = QtWidgets.QPushButton()
        self.previous_button.setIcon(QtGui.QIcon(os.path.join(parent.icon_folder, "backward_icon.png")))
        self.previous_button.setIconSize(QtCore.QSize(50, 50))
        self.previous_button.clicked.connect(parent.music_manager.play_previous_music)
        controls_layout.addWidget(self.previous_button)

        self.play_button = QtWidgets.QPushButton()
        self.update_play_button_icon()
        self.play_button.setIconSize(QtCore.QSize(50, 50))
        self.play_button.clicked.connect(self.toggle_play_pause_in_fullscreen)
        controls_layout.addWidget(self.play_button)

        self.next_button = QtWidgets.QPushButton()
        self.next_button.setIcon(QtGui.QIcon(os.path.join(parent.icon_folder, "forward_icon.png")))
        self.next_button.setIconSize(QtCore.QSize(50, 50))
        self.next_button.clicked.connect(parent.music_manager.play_next_music)
        controls_layout.addWidget(self.next_button)

        # Boutons des paramètres
        self.color_button = QtWidgets.QPushButton("Couleur Spectre")
        self.color_button.clicked.connect(self.spectrum.change_spectrum_color)
        settings_layout.addWidget(self.color_button)

        self.background_button = QtWidgets.QPushButton("Couleur Fond")
        self.background_button.clicked.connect(parent.change_background_color)
        settings_layout.addWidget(self.background_button)

        self.chunk_size_button = QtWidgets.QPushButton("Taille Chunk")
        self.chunk_size_button.clicked.connect(self.open_chunk_size_menu)
        settings_layout.addWidget(self.chunk_size_button)

        self.exit_button = QtWidgets.QPushButton("Quitter")
        self.exit_button.clicked.connect(self.animate_exit)
        settings_layout.addWidget(self.exit_button)

        # Ajouter les layouts
        main_layout.addLayout(controls_layout)
        main_layout.addLayout(settings_layout)

        # Configuration de l'animation
        self.animation = QPropertyAnimation(self, b"geometry")
        self.animation.setDuration(700)  # Durée augmentée pour plus de fluidité
        self.animation.setEasingCurve(QtCore.QEasingCurve.Type.InOutQuad)

        # Lancer l'animation d'entrée
        self.animate_enter()

    def toggle_play_pause_in_fullscreen(self):
        """Appelle la méthode principale pour gérer Lecture/Pause et met à jour l'icône."""
        self.parent.toggle_play_pause()
        self.update_play_button_icon()

    def update_play_button_icon(self):
        """Mise à jour de l'icône du bouton Lecture/Pause."""
        if self.parent.playing and not self.parent.paused:
            icon_path = os.path.join(self.parent.icon_folder, "pause_icon.png")
        else:
            icon_path = os.path.join(self.parent.icon_folder, "play_icon.png")
        self.play_button.setIcon(QtGui.QIcon(icon_path))

    def open_chunk_size_menu(self):
        """Affiche un menu pour sélectionner la taille des chunks."""
        menu = QtWidgets.QMenu(self)
        for size in [4096, 8192, 16384, 32768]:
            action = QtGui.QAction(f"{size}", self)
            action.setCheckable(True)
            action.setChecked(self.parent.chunk_size == size)
            action.triggered.connect(lambda _, s=size: self.parent.set_chunk_size(s))
            menu.addAction(action)

        menu.exec(self.chunk_size_button.mapToGlobal(QtCore.QPoint(0, self.chunk_size_button.height())))

    def animate_enter(self):
        """Anime l'entrée en mode plein écran avec un zoom progressif."""
        # Cacher la fenêtre parent
        self.parent.hide()
        
        # Rendre la fenêtre transparente au début
        self.setWindowOpacity(0.0)
        
        # Calcul des géométries
        start_geometry = self.parent.geometry()
        end_geometry = QtWidgets.QApplication.primaryScreen().availableGeometry()

        # Configuration de l'animation de géométrie
        self.geometry_animation = QPropertyAnimation(self, b"geometry")
        self.geometry_animation.setDuration(700)
        self.geometry_animation.setEasingCurve(QtCore.QEasingCurve.Type.InOutQuad)
        self.geometry_animation.setStartValue(start_geometry)
        self.geometry_animation.setEndValue(end_geometry)

        # Configuration de l'animation d'opacité
        self.opacity_animation = QPropertyAnimation(self, b"windowOpacity")
        self.opacity_animation.setDuration(700)
        self.opacity_animation.setStartValue(0.0)
        self.opacity_animation.setEndValue(1.0)

        # Groupe d'animation
        self.animation_group = QtCore.QParallelAnimationGroup()
        self.animation_group.addAnimation(self.geometry_animation)
        self.animation_group.addAnimation(self.opacity_animation)

        # Définir la géométrie initiale et afficher la fenêtre
        self.setGeometry(start_geometry)
        self.show()

        # Démarrer les animations
        self.animation_group.start()

        # Actions après l'animation
        self.animation_group.finished.connect(self.raise_)
        self.animation_group.finished.connect(self.activateWindow)

    def animate_exit(self):
        """Anime la sortie du mode plein écran avec un zoom progressif."""
        # Calcul de la position centrale pour revenir au mode initial
        end_geometry = self.parent.geometry()

        # Configuration de l'animation de géométrie
        self.geometry_animation = QPropertyAnimation(self, b"geometry")
        self.geometry_animation.setDuration(700)
        self.geometry_animation.setEasingCurve(QtCore.QEasingCurve.Type.InOutQuad)
        self.geometry_animation.setStartValue(self.geometry())
        self.geometry_animation.setEndValue(end_geometry)

        # Configuration de l'animation d'opacité
        self.opacity_animation = QPropertyAnimation(self, b"windowOpacity")
        self.opacity_animation.setDuration(700)
        self.opacity_animation.setStartValue(1.0)
        self.opacity_animation.setEndValue(0.0)

        # Groupe d'animation
        self.animation_group = QtCore.QParallelAnimationGroup()
        self.animation_group.addAnimation(self.geometry_animation)
        self.animation_group.addAnimation(self.opacity_animation)

        # Connecter la fin de l'animation
        self.animation_group.finished.connect(self.close)
        self.animation_group.finished.connect(self.parent.show)  # Réafficher la fenêtre parent

        # Démarrer les animations
        self.animation_group.start()

    def closeEvent(self, event):
        """Restaure le spectre à sa position dans la fenêtre principale lors de la fermeture."""
        if self.parent and self.spectrum:
            self.parent.central_widget.layout().insertWidget(0, self.spectrum)
        event.accept()
