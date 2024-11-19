# Ce projet est sous licence MIT.
# Copyright (c) 2024 Nicolas Q.


import os
import random
from PyQt6 import QtWidgets, QtGui, QtCore
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QProgressBar, QFileDialog, QMessageBox, QColorDialog, QSlider, QLabel
from concurrent.futures import ThreadPoolExecutor
from pygame import mixer
from spectrum import AudioSpectrum, FullScreenSpectrum
from settings_manager import load_settings, save_settings, load_music_files_from_storage, load_stats, save_stats
import wave
import json
import shutil
import numpy as np
from animated_list_widget import AnimatedListWidget
from animated_list_widget_item import AnimatedListWidgetItem
from music_manager import MusicManager

class MusicPlayer(QtWidgets.QMainWindow):
    update_music_list_signal = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()

        # Charger les paramètres
        self.settings = load_settings()
        self.music_manager = MusicManager(self)  # Initialiser MusicManager avec une référence à MusicPlayer
        # Charger la taille du chunk depuis les paramètres et s'assurer qu'il s'agit bien d'un entier
        self.chunk_size = int(self.settings.get('chunk_size', 16384))
        self.paused_position = 0  # Position sauvegardée lors de la mise en pause

        # Vérification et message de débogage pour confirmer que chunk_size est bien un entier
        if not isinstance(self.chunk_size, int):
            print("Erreur : la taille du chunk n'est pas un entier. Valeur actuelle:", self.chunk_size)
            self.chunk_size = 16384
        else:
            print("Taille du chunk chargée correctement :", self.chunk_size)

        self.setWindowTitle("AcoustiqAV")
        self.setGeometry(100, 100, 1000, 800)
        self.setAcceptDrops(True)
        self.executor = None

        self.scroll_timer = QtCore.QTimer(self)
        self.scroll_timer.timeout.connect(self.scroll_text)

        # Initialisation du dossier des icônes
        self.icon_folder = os.path.join(os.path.dirname(__file__), "ICON")
        icon_path = os.path.join(self.icon_folder, "app_icon.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QtGui.QIcon(icon_path))

        try:
            mixer.init()
        except Exception as e:
            print(f"Erreur lors de l'initialisation de pygame.mixer : {e}")
            QMessageBox.critical(self, "Erreur", "Échec de l'initialisation de la bibliothèque audio.")
            return

        # Appliquer l'état 'toujours au premier plan' dès le démarrage
        always_on_top = self.settings.get('always_on_top', False)
        self.setWindowFlag(QtCore.Qt.WindowType.WindowStaysOnTopHint, always_on_top)
        
        # Initialiser les autres attributs
        self.music_data = load_music_files_from_storage()  # Charger les données de musique
        self.filtered_music_data = []
        self.deleted_music_backup = []
        self.current_track_index = -1
        self.paused = False
        self.playing = False
        self.sort_order = self.settings.get("sort_order", "Nom")
        self.shuffle_mode = self.settings.get("shuffle_mode", False)
        
        # Compteur de pistes
        self.track_counter = QtWidgets.QLabel("Total de musiques : 0")
        self.track_counter.setStyleSheet("""
        QLabel {
            border: 2px solid #FF5722;
            padding: 5px;
            border-radius: 8px;
            font-size: 14px;
        }
        """)

        # Créer une barre de progression avec un tracé coloré et un dégradé
        self.progress = QProgressBar(self)
        self.progress.setRange(0, 1000)  # Utiliser 1000 pour plus de précision
        self.progress.setValue(0)
        self.progress.setTextVisible(False)  # Masquer le texte si non nécessaire
        self.progress.setStyleSheet("""
            QProgressBar {
                border: 1px solid #4A4A4A;
                border-radius: 4px;
                background-color: #2A2A3A;
                height: 8px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                        stop:0 #3F51B5, stop:1 #FF0000);
                border-radius: 4px;
                border-top: 2px solid #FF0000;
            }
        """)

        # Ajouter la barre de progression au layout principal
        self.central_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.central_widget)
        layout = QtWidgets.QVBoxLayout(self.central_widget)
        layout.addWidget(self.progress)
        
        # Initialisation de l'affichage du spectre audio avec la taille de chunk chargée
        self.spectrum = AudioSpectrum(self)
        self.spectrum.update_chunk_size(self.chunk_size)  # Appliquer la taille du chunk au spectre audio

        # Ajouter le spectre audio au layout principal
        layout.addWidget(self.spectrum)
        
        # Initialiser le timer de progression pour la barre de progression
        self.progress_timer = QtCore.QTimer(self)
        self.progress_timer.setInterval(100)  # Mise à jour toutes les 100ms
        self.progress_timer.timeout.connect(self.update_progress_bar)

        # Ajouter un timer pour vérifier la fin de la musique
        self.end_check_timer = QtCore.QTimer(self)
        self.end_check_timer.setInterval(100)  # Vérifier toutes les 100ms
        self.end_check_timer.timeout.connect(self.music_manager.check_music_end)
        self.end_check_timer.start()
        
        # Initialiser l'interface utilisateur
        self.init_ui()

        # Appliquer la taille du chunk sauvegardée après l'initialisation de l'UI
        self.set_chunk_size(self.chunk_size)
        
        # Mettre à jour la liste de musique et le compteur dès le chargement
        self.music_manager.update_music_list()  # Actualiser la liste avec les données chargées
        self.music_manager.update_music_counter()  # Mettre à jour le compteur avec le nombre de musiques

        # Initialise l'historique des pistes jouées
        self.played_tracks_history = []

        # Initialise le volume avec une valeur par défaut.
        self.current_volume = round(mixer.music.get_volume() * 100)  # Stocke le volume actuel en pourcentage

        # Appliquer le tri selon le mode de tri sauvegardé
        self.apply_current_sort()

        # Restaurer et appliquer le mode de répétition initial
        self.repeat_mode = self.settings.get("repeat_mode", "none")
        self.apply_repeat_mode()

        # Initialiser les autres variables
        self.song_length = None
        self.executor = None

    def update_play_stats(self, track_name):
        """Met à jour les statistiques de lecture pour un morceau."""
        stats = load_stats()
        if track_name not in stats:
            stats[track_name] = {"plays": 0, "total_time": 0}
        stats[track_name]["plays"] += 1
        save_stats(stats)

    def toggle_fullscreen_spectrum(self):
        """Active ou désactive le mode plein écran pour le spectre."""
        if hasattr(self, 'fullscreen_window') and self.fullscreen_window and self.fullscreen_window.isVisible():
            # Sortie du mode plein écran avec une animation
            self.fullscreen_window.animate_exit()  # Utilise l'animation de sortie
        else:
            # Entrée en mode plein écran
            self.fullscreen_window = FullScreenSpectrum(self)  # Crée une fenêtre plein écran
            self.fullscreen_window.showFullScreen()

            # Ajout du raccourci F11 pour quitter le mode plein écran
            fullscreen_shortcut = QtGui.QShortcut(QtGui.QKeySequence("F11"), self.fullscreen_window)
            fullscreen_shortcut.activated.connect(self.fullscreen_window.animate_exit)

    def set_chunk_size(self, size):
        """Change la taille du chunk et met à jour la configuration en temps réel."""
        self.chunk_size = int(size)  # Assurez-vous que size est un entier
        self.settings['chunk_size'] = self.chunk_size  # Mettre à jour le paramètre de taille de chunk dans les settings
        save_settings(self.settings)  # Sauvegarde tous les paramètres, y compris la taille des chunks
        for action in self.chunk_size_menu.actions():
            action.setChecked(int(action.text()) == self.chunk_size)
        self.spectrum.update_chunk_size(self.chunk_size)

    def show_about_dialog(self):
        about_text = (
            "<h2>AcoustiqAV 🎶</h2>"
            "<p><strong>Développé par :</strong> Nicolas Q.</p>"
            "<p><strong>Version actuelle :</strong> 1.2.0</p>"
            "<p><strong>Licence :</strong> MIT</p>"
            "<p><strong>Compatibilité :</strong> Windows</p>"
            "<p><strong>Objectif :</strong> Offrir une expérience musicale immersive grâce à une interface moderne et intuitive, "
            "complétée par un spectre audio dynamique et des fonctionnalités avancées.</p>"
            "<h3>📖 Guide de démarrage rapide</h3>"
            "<ul>"
            "<li><strong>Charger une musique :</strong> Utilisez l'option <em>Ouvrir</em> dans le menu pour charger vos fichiers WAV.</li>"
            "<li><strong>Personnalisation :</strong> Accédez au menu <em>Visualisation</em> pour ajuster les couleurs et les options avancées.</li>"
            "</ul>"
            "<h3>🎮 Commandes</h3>"
            "<ul>"
            "<li><strong>Lecture/Pause :</strong> Utilisez le bouton <em>Lecture</em> ou <em>Pause</em> pour contrôler la musique.</li>"
            "<li><strong>Piste suivante/précédente :</strong> Passez à la piste suivante ou précédente avec les boutons dédiés.</li>"
            "<li><strong>Mode plein écran :</strong> Activez le mode plein écran pour une visualisation immersive en appuyant sur <strong>F11</strong> ou via le menu.</li>"
            "<li><strong>Raccourcis clavier :</strong>"
            "<ul>"
            "<li><strong>Espace :</strong> Lecture/Pause</li>"
            "<li><strong>Flèche haut :</strong> Augmenter le volume</li>"
            "<li><strong>Flèche bas :</strong> Diminuer le volume</li>"
            "<li><strong>F11 :</strong> Activer/Désactiver le mode plein écran</li>"
            "</ul>"
            "</li>"
            "</ul>"
            "<p><strong>Formats audio supportés :</strong> WAV uniquement</p>"
            "<h3>🛠️ Historique des Versions</h3>"
            "<ul>"
            "<li><strong>1.1.0 :</strong> Ajout du menu <em>Visualisation</em>, personnalisation des couleurs et gestion des chunks FFT.</li>"
            "<li><strong>1.0.1 :</strong> Correction de bugs mineurs, ajout de la fonction pause et stabilité améliorée.</li>"
            "<li><strong>1.0.0 :</strong> Version initiale avec prise en charge des fichiers WAV et spectre audio en temps réel.</li>"
            "</ul>"
            "<p><strong>🔗 Crédits :</strong> Ce projet utilise des bibliothèques open-source :</p>"
            "<ul>"
            "<li><em>PyQt6</em> : Interface graphique</li>"
            "<li><em>numpy</em> et <em>scipy</em> : Calculs spectraux</li>"
            "<li><em>pygame</em> : Gestion audio</li>"
            "</ul>"
            "<p style='color:red;'>Remarque importante : Ce lecteur supporte uniquement les fichiers audio au format WAV.</p>"
        )
        QMessageBox.about(self, "À propos", about_text)

    def show_updates_dialog(self):
        updates_text = (
            "<h2>🆕 Quoi de neuf ?</h2>"
            "<p><strong>Version actuelle :</strong> 1.2.0 (novembre 2024)</p>"
            "<h3>Dernières améliorations :</h3>"
            "<ul>"
            "<li><strong>🔄 Modes de répétition améliorés :</strong> Les modes de répétition (piste unique, playlist) fonctionnent maintenant parfaitement dès le lancement de l'application. L'expérience utilisateur est stable, qu'il s'agisse de lectures continues ou de retours manuels sur des pistes.</li>"
            "<li><strong>📊 Statistiques d'écoute corrigées :</strong> Les durées totales d'écoute sont désormais calculées et triées correctement, du plus long au plus court. Toutes les anomalies dans les calculs ont été résolues.</li>"
            "<li><strong>⏸️ Pause améliorée :</strong> La barre de progression et le spectre audio restent visibles et synchronisés après une pause prolongée, assurant une expérience utilisateur fluide.</li>"
            "<li><strong>⏱️ Synchronisation précise des temps :</strong> Le compteur de temps et la barre de progression sont maintenant réactifs et exacts, même après un changement manuel de piste.</li>"
            "<li><strong>🎵 Visualisation audio persistante :</strong> Le spectre audio reste actif après des pauses, des reprises ou des changements de pistes, offrant une continuité visuelle sans faille.</li>"
            "<li><strong>💻 Compatibilité renforcée :</strong> De nombreuses optimisations ont été réalisées pour garantir une expérience homogène sur Windows.</li>"
            "<li><strong>🚨 Gestion améliorée des erreurs :</strong> Les problèmes liés aux fichiers manquants ou endommagés affichent désormais des messages clairs pour aider l'utilisateur à résoudre facilement les situations.</li>"
            "<li><strong>✨ Améliorations visuelles :</strong> Des styles plus modernes et lisibles ont été appliqués aux menus et dialogues, notamment ceux des statistiques et des nouveautés.</li>"
            "</ul>"
            "<h3>📜 Historique des mises à jour :</h3>"
            "<ul>"
            "<li><strong>1.2.0 :</strong> Modes de répétition avancés, correction et tri des statistiques, synchronisation des temps et meilleure compatibilité système.</li>"
            "<li><strong>1.1.0 :</strong> Ajout du menu <em>Visualisation</em>, personnalisation des couleurs et gestion avancée des chunks FFT.</li>"
            "<li><strong>1.0.1 :</strong> Correction de bugs mineurs, fonction pause ajoutée, et compatibilité initiale Windows.</li>"
            "<li><strong>1.0.0 :</strong> Version initiale avec prise en charge des fichiers WAV et spectre audio en temps réel.</li>"
            "</ul>"
        )
        QMessageBox.about(self, "Nouveautés", updates_text)

    def init_executor(self):
        """Initialise le ThreadPoolExecutor pour les tâches en arrière-plan si nécessaire."""
        if self.executor is None:
            self.executor = ThreadPoolExecutor(max_workers=2)

    def init_ui(self):
        # Configuration de la fenêtre principale
        self.central_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.central_widget)

        # Création de la barre de menu
        menubar = self.menuBar()

        # Menu Paramètres
        settings_menu = menubar.addMenu("Paramètres")

        # Toujours au premier plan
        self.always_on_top_action = QAction("Toujours au premier plan", self)
        self.always_on_top_action.setCheckable(True)
        self.always_on_top_action.setChecked(self.settings.get("always_on_top", False))
        self.always_on_top_action.triggered.connect(self.toggle_always_on_top)
        settings_menu.addAction(self.always_on_top_action)

        # Actualiser la bibliothèque
        refresh_action = QAction("Actualiser la bibliothèque", self)
        refresh_action.triggered.connect(self.music_manager.refresh_music_folder)
        settings_menu.addAction(refresh_action)

        # Ajouter une action pour activer le mode plein écran
        fullscreen_action = QAction("Mode Plein Écran", self)
        fullscreen_action.triggered.connect(self.toggle_fullscreen_spectrum)
        settings_menu.addAction(fullscreen_action)  # Ajout du bouton dans le menu Paramètres

        # Ajouter le raccourci clavier pour basculer le mode plein écran
        self.fullscreen_shortcut = QtGui.QShortcut(QtGui.QKeySequence("F11"), self)
        self.fullscreen_shortcut.activated.connect(self.toggle_fullscreen_spectrum)

        # Menu Visualisation
        visualization_menu = menubar.addMenu("Visualisation")

        # Option pour changer la couleur du spectre
        self.change_color_action = QAction("Changer la couleur du spectre", self)
        self.change_color_action.triggered.connect(lambda: self.spectrum.change_spectrum_color())
        visualization_menu.addAction(self.change_color_action)

        # Option pour changer la couleur de fond
        self.change_background_color_action = QAction("Changer la couleur de fond", self)
        self.change_background_color_action.triggered.connect(lambda: self.change_background_color())
        visualization_menu.addAction(self.change_background_color_action)

        # Option pour sélectionner la taille du chunk
        self.chunk_size_menu = QtWidgets.QMenu("Taille du Chunk", self)
        for size in [4096, 8192, 16384, 32768]:
                action = QtGui.QAction(f"{size}", self)
                action.setCheckable(True)
                action.setChecked(self.chunk_size == size)
                action.triggered.connect(lambda _, s=size: self.set_chunk_size(s))
                self.chunk_size_menu.addAction(action)
        visualization_menu.addMenu(self.chunk_size_menu)

        # Menu Statistiques
        stats_action = QAction("Statistiques", self)
        stats_action.triggered.connect(self.show_stats_dialog)
        menubar.addAction(stats_action)

        # Menu Nouveautés
        updates_action = QAction("Nouveautés", self)
        updates_action.triggered.connect(self.show_updates_dialog)
        menubar.addAction(updates_action)

        # Menu À propos
        about_action = QAction("À propos", self)
        about_action.triggered.connect(self.show_about_dialog)
        menubar.addAction(about_action)

        # Couleurs du spectre et de fond à partir des paramètres
        spectrum_color = self.settings.get("spectrum_color", "#FF0000")
        background_color = self.settings.get("background_color", "#2A2A3A")

        # Ajouter les raccourcis clavier
        self.shortcut_play_pause = QtGui.QShortcut(QtGui.QKeySequence("Space"), self)
        self.shortcut_play_pause.activated.connect(self.toggle_play_pause)

        self.shortcut_next = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Right"), self)
        self.shortcut_next.activated.connect(self.music_manager.play_next_music)

        self.shortcut_previous = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Left"), self)
        self.shortcut_previous.activated.connect(self.music_manager.play_previous_music)

        self.shortcut_volume_up = QtGui.QShortcut(QtGui.QKeySequence("Up"), self)
        self.shortcut_volume_up.activated.connect(self.increase_volume)

        self.shortcut_volume_down = QtGui.QShortcut(QtGui.QKeySequence("Down"), self)
        self.shortcut_volume_down.activated.connect(self.decrease_volume)

        # Initialisation de la liste de pistes avec un style animé
        self.tracklist = AnimatedListWidget(self)
        self.tracklist.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.tracklist.itemDoubleClicked.connect(self.music_manager.play_music)

        # Créer la barre de progression
        self.progress = QSlider(QtCore.Qt.Orientation.Horizontal)
        self.progress.setRange(0, 1000)  # Utiliser 1000 pour plus de précision
        self.progress.setValue(0)

        # Chemin vers l'image du curseur
        cursor_icon_path = os.path.join(self.icon_folder, "cursor_image.png")

        # Appliquez le style pour le curseur de la barre de progression
        self.progress.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                border: 1px solid #999999;
                height: 8px;
                background: #4A4A4A;
                margin: 2px 0;
                border-radius: 4px;
            }}
            QSlider::handle:horizontal {{
                image: url("{cursor_icon_path.replace('\\', '/')}");
                width: 20px;
                height: 20px;
                margin: -6px 0;
            }}
            QSlider::handle:horizontal:hover {{
                image: url("{cursor_icon_path.replace('\\', '/')}");
            }}
        """)

        self.progress.setEnabled(False)

        # Ajouter les labels pour le temps
        self.time_label = QtWidgets.QLabel("0:00 / 0:00")
        self.time_label.setStyleSheet("""
            QLabel {{
                color: white;
                padding: 5px;
                font-size: 12px;
            }}
        """)

        # Créer un layout pour la barre de progression et le temps
        progress_layout = QtWidgets.QHBoxLayout()
        progress_layout.addWidget(self.time_label)
        progress_layout.addWidget(self.progress)

        # Barre de recherche
        self.search_var = QtWidgets.QLineEdit(self)
        self.search_var.setPlaceholderText("Rechercher")
        self.search_var.setFixedWidth(200)
        self.search_var.setStyleSheet("""
            QLineEdit {
                padding: 8px;
                border: 2px solid #CCCCCC;
                border-radius: 10px;
                background-color: #F5F5F5;
                font-size: 14px;
                color: #333333;
            }
            QLineEdit:focus {
                border-color: #3F51B5;
                background-color: #FFFFFF;
            }
        """)
        self.search_var.textChanged.connect(self.music_manager.update_music_list)

        # Compteur de pistes
        self.track_counter = QtWidgets.QLabel("Total de musiques : 0")
        self.track_counter.setStyleSheet("""
        QLabel {
            border: 2px solid #FF5722;
            padding: 5px;
            border-radius: 8px;
            font-size: 14px;
        }
        """)

        # Affichage du morceau actuel
        self.current_track_label = QtWidgets.QLabel("Aucun morceau à l'écoute")
        self.current_track_label.setFixedWidth(400)
        self.current_track_label.setFixedHeight(40)
        self.current_track_label.setStyleSheet("""
        QLabel {
            border: 2px solid #3F51B5;
            padding: 5px;
            border-radius: 8px;
            font-size: 14px;
            color: #FFFFFF;
            background-color: #333333;
        }
        """)
        self.current_track_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
    
        # Boutons de tri et de lecture aléatoire
        self.sort_button = QtWidgets.QPushButton("Trier")
        self.sort_button.setFixedSize(60, 30)
        self.sort_menu = QtWidgets.QMenu()
        self.sort_menu.addAction("Nom", self.sort_by_name)
        self.sort_menu.addAction("Date de modification", self.sort_by_date)
        self.sort_menu.addAction("Taille", self.sort_by_size)
        self.sort_button.setMenu(self.sort_menu)

        self.shuffle_button = QtWidgets.QPushButton("Aléatoire")
        self.shuffle_button.setFixedSize(70, 30)
        self.shuffle_button.setCheckable(True)
        self.shuffle_button.setChecked(self.settings.get("shuffle_mode", False))
        self.shuffle_button.clicked.connect(self.toggle_shuffle_mode)

        shuffle_color = self.settings.get("shuffle_button_color", "#FF5C5C")
        if self.shuffle_button.isChecked():
            self.shuffle_button.setStyleSheet(f"background-color: {shuffle_color};")
        else:
            self.shuffle_button.setStyleSheet("")

        # Style de base des boutons avec des dégradés et des coins arrondis
        button_style = """
        QPushButton {
            background-color: qlineargradient(
                x1: 0, y1: 0, x2: 1, y2: 1,
                stop: 0 #4CAF50, stop: 1 #66BB6A
            );  /* Couleur de dégradé vert */
            color: white;
            font-weight: bold;
            border: none;
            border-radius: 12px;  /* Plus arrondi */
            padding: 10px 20px;
        }
        QPushButton:hover {
            background-color: qlineargradient(
                x1: 0, y1: 0, x2: 1, y2: 1,
                stop: 0 #66BB6A, stop: 1 #81C784
            );  /* Changement de dégradé au survol */
        }
        QPushButton:disabled {
            background-color: #A5A5A5;
            color: #E0E0E0;
        }
        """

        delete_button_style = """
        QPushButton {
            background-color: qlineargradient(
                x1: 0, y1: 0, x2: 1, y2: 1,
                stop: 0 #E53935, stop: 1 #EF5350
            );  /* Couleur de dégradé rouge */
            color: white;
            font-weight: bold;
            border: none;
            border-radius: 12px;
            padding: 10px 20px;
        }
        QPushButton:hover {
            background-color: qlineargradient(
                x1: 0, y1: 0, x2: 1, y2: 1,
                stop: 0 #EF5350, stop: 1 #FF7043
            );
        }
        QPushButton:disabled {
            background-color: #A5A5A5;
            color: #E0E0E0;
        }
        """

        control_button_style = """
        QPushButton {
            background-color: qlineargradient(
                x1: 0, y1: 0, x2: 1, y2: 1,
                stop: 0 #8E24AA, stop: 1 #AB47BC
            );  /* Dégradé violet */
            color: white;
            border: none;
            border-radius: 15px;
            padding: 8px;
            font-size: 16px;
        }
        QPushButton:hover {
            background-color: qlineargradient(
                x1: 0, y1: 0, x2: 1, y2: 1,
                stop: 0 #AB47BC, stop: 1 #BA68C8
            );
        }
        QPushButton:disabled {
            background-color: #A5A5A5;
            color: #E0E0E0;
        }
        """

        # Application du style aux différents boutons de contrôle
        self.select_all_button = QtWidgets.QPushButton("Tout sélectionner")
        self.select_all_button.setStyleSheet(button_style)
        self.select_all_button.clicked.connect(self.music_manager.select_all_tracks)

        self.add_music_button = QtWidgets.QPushButton("Ajouter des musiques")
        self.add_music_button.setStyleSheet(button_style)
        self.add_music_button.clicked.connect(self.music_manager.add_music_files)

        self.play_button = QtWidgets.QPushButton()
        self.play_button.setIcon(QtGui.QIcon(os.path.join(self.icon_folder, "play_icon.png")))
        self.play_button.setIconSize(QtCore.QSize(30, 30))
        self.play_button.setStyleSheet(control_button_style)
        self.play_button.clicked.connect(self.toggle_play_pause)

        self.forward_button = QtWidgets.QPushButton()
        self.forward_button.setIcon(QtGui.QIcon(os.path.join(self.icon_folder, "forward_icon.png")))
        self.forward_button.setIconSize(QtCore.QSize(30, 30))
        self.forward_button.setStyleSheet(control_button_style)
        self.forward_button.clicked.connect(self.music_manager.play_next_music)

        self.backward_button = QtWidgets.QPushButton()
        self.backward_button.setIcon(QtGui.QIcon(os.path.join(self.icon_folder, "backward_icon.png")))
        self.backward_button.setIconSize(QtCore.QSize(30, 30))
        self.backward_button.setStyleSheet(control_button_style)
        self.backward_button.clicked.connect(self.music_manager.play_previous_music)

        self.delete_button = QtWidgets.QPushButton("Supprimer")
        self.delete_button.setStyleSheet(delete_button_style)
        self.delete_button.clicked.connect(self.music_manager.delete_selected_music)

        self.restore_button = QtWidgets.QPushButton("Restaurer")
        self.restore_button.setStyleSheet(button_style)
        self.restore_button.clicked.connect(self.music_manager.restore_music)
        self.restore_button.setEnabled(False)
        
        # Bouton de répétition
        repeat_button_style = button_style.replace("#4CAF50", "#FF9800").replace("#45A049", "#FB8C00")
        self.repeat_button = QtWidgets.QPushButton()
        self.repeat_button.setStyleSheet(repeat_button_style)
        self.repeat_button.setIcon(QtGui.QIcon(os.path.join(self.icon_folder, "repeat_icon.png")))
        self.repeat_button.setIconSize(QtCore.QSize(30, 30))
        self.repeat_button.clicked.connect(self.toggle_repeat)

        # Layouts pour organiser les widgets
        search_sort_layout = QtWidgets.QHBoxLayout()
        search_sort_layout.addWidget(self.track_counter)
        search_sort_layout.addStretch()
        search_sort_layout.addWidget(self.search_var)
        search_sort_layout.addWidget(self.sort_button)
        search_sort_layout.addWidget(self.shuffle_button)
        search_sort_layout.addStretch()
        search_sort_layout.addWidget(self.current_track_label)

        layout = QtWidgets.QVBoxLayout(self.central_widget)
        layout.addLayout(search_sort_layout)
        layout.addWidget(self.spectrum)
        layout.addLayout(progress_layout)  # Ajouter le layout de progression
        layout.addWidget(self.tracklist)
        
        control_layout = QtWidgets.QHBoxLayout()
        control_layout.addWidget(self.select_all_button)
        control_layout.addWidget(self.add_music_button)  # Ajout du bouton "Ajouter des musiques"
        control_layout.addWidget(self.delete_button)
        control_layout.addWidget(self.restore_button)
        control_layout.addWidget(self.backward_button)
        control_layout.addWidget(self.play_button)
        control_layout.addWidget(self.forward_button)
        control_layout.addWidget(self.repeat_button)

        layout.addLayout(control_layout)
        self.update_music_list_signal.connect(self.music_manager.update_music_list)
    
    def increase_volume(self):
        """Augmente le volume par incréments fixes de 5 % et affiche une notification."""
        # On calcule le nouveau volume en respectant les limites
        self.current_volume = min(100, self.current_volume + 5)
        # On applique le nouveau volume
        mixer.music.set_volume(self.current_volume / 100.0)
        # Affiche la notification
        self.spectrum.show_volume_notification(f"Volume : {self.current_volume}%")

    def decrease_volume(self):
        """Diminue le volume par incréments fixes de 5 % et affiche une notification."""
        # On calcule le nouveau volume en respectant les limites
        self.current_volume = max(0, self.current_volume - 5)
        # On applique le nouveau volume
        mixer.music.set_volume(self.current_volume / 100.0)
        # Affiche la notification
        self.spectrum.show_volume_notification(f"Volume : {self.current_volume}%")
        
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            files = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile() and url.toLocalFile().endswith('.wav')]
            if files:
                self.init_executor()
                self.executor.submit(self.add_files_async, files)
                event.acceptProposedAction()
        else:
            event.ignore()

    def change_background_color(self):
        """Ouvre une boîte de dialogue pour choisir la couleur de fond."""
        color_dialog = QtWidgets.QColorDialog(self)
        color_dialog.setWindowTitle("Choisissez la couleur de fond")

        # Définir la couleur actuelle de fond comme couleur initiale
        color_dialog.setCurrentColor(QtGui.QColor(self.settings.get("background_color", "#2A2A3A")))
        if color_dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            # Appliquer et sauvegarder la couleur sélectionnée
            color = color_dialog.selectedColor()
            self.spectrum.set_background_color(color)
            self.settings['background_color'] = color.name()
            save_settings(self.settings)  # Utiliser la fonction importée pour sauvegarder les paramètres

    def update_scroll(self):
        """Démarre le défilement si le texte est trop long pour le label."""
        self.scroll_text_full = self.current_track_label.text()
        font_metrics = self.current_track_label.fontMetrics()
        text_width = font_metrics.horizontalAdvance(self.scroll_text_full)
        if text_width > self.current_track_label.width() - 20:  # Vérifie si le texte dépasse
            self.scroll_text_full += "     "  # Ajoute de l'espace pour une transition fluide
            if not self.scroll_timer.isActive():
                self.scroll_position = 0
                self.scroll_timer.start(50)  # Intervalle de défilement
        else:
            self.scroll_timer.stop()
            self.scroll_position = 0
            self.current_track_label.setText(self.scroll_text_full)

    def scroll_text(self):
        """Défile le texte de droite à gauche."""
        visible_text_length = self.current_track_label.width() // 10  # Estimation des caractères visibles
        display_text = (self.scroll_text_full + "     ") * 2  # Double le texte pour un défilement en continu
        if self.scroll_position < len(self.scroll_text_full):
            self.scroll_position += 1
        else:
            self.scroll_position = 0  # Réinitialiser la position pour un défilement fluide
        self.current_track_label.setText(display_text[self.scroll_position:self.scroll_position + visible_text_length])

    def toggle_always_on_top(self):
        """Active ou désactive l'option 'Toujours au premier plan'."""
        always_on_top = self.always_on_top_action.isChecked()
        self.setWindowFlag(QtCore.Qt.WindowType.WindowStaysOnTopHint, always_on_top)
        self.show()  # Appliquer le changement immédiatement
        self.settings['always_on_top'] = always_on_top
        save_settings(self.settings)  # Utiliser la fonction importée pour sauvegarder les paramètres

    def apply_repeat_mode(self):
        """Applique immédiatement le mode de répétition sélectionné."""
        if self.repeat_mode == "track":
            self.repeat_button.setIcon(QtGui.QIcon(os.path.join(self.icon_folder, "repeat_track_icon.png")))
            print("Mode de répétition : Répéter la piste actuelle.")
        elif self.repeat_mode == "playlist":
            self.repeat_button.setIcon(QtGui.QIcon(os.path.join(self.icon_folder, "repeat_playlist_icon.png")))
            print("Mode de répétition : Répéter la playlist.")
        else:
            self.repeat_button.setIcon(QtGui.QIcon(os.path.join(self.icon_folder, "repeat_icon.png")))
            print("Mode de répétition : Aucun.")

    def load_folders(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Sélectionner des fichiers de musique", "", "Fichiers WAV (*.wav)")
        if files:
            self.init_executor()
            self.executor.submit(self.add_files_async, files)

    def add_files_async(self, files):
        """Ajoute les fichiers de musique sélectionnés à la liste en arrière-plan et les copie dans le dossier Stockage_Musique."""
        
        # Chemin du dossier de stockage
        storage_path = os.path.join(os.path.expanduser("~"), "Music", "Stockage_Musique")
        
        # Vérifie si le dossier existe, sinon le crée
        if not os.path.exists(storage_path):
            os.makedirs(storage_path)
        
        # Récupère tous les noms de fichiers déjà présents dans `self.music_data` pour éviter les doublons
        existing_files = {music['display_name'] for music in self.music_data}

        for track in files:
            # Créer le chemin de destination dans `Stockage_Musique`
            destination_path = os.path.join(storage_path, os.path.basename(track))
            display_name = os.path.basename(track).replace('.wav', '').lower()

            # Vérifie si le fichier est déjà dans la liste des musiques
            if display_name in existing_files:
                print(f"Le fichier {display_name} est déjà présent. Il ne sera pas ajouté de nouveau.")
                continue

            # Copier le fichier dans le dossier `Stockage_Musique` s’il n'est pas déjà à cet emplacement
            if track != destination_path:
                shutil.copy(track, destination_path)

            try:
                # Validation du fichier .wav
                with wave.open(destination_path, 'rb') as wav_file:
                    if wav_file.getnchannels() not in [1, 2]:  # Vérifie que le fichier est mono ou stéréo
                        print(f"Le fichier {destination_path} a un format audio non supporté.")
                        continue
                    length = wav_file.getnframes() / wav_file.getframerate()

                # Récupération des informations du fichier
                size = os.path.getsize(destination_path)
                mtime = os.path.getmtime(destination_path)

                # Ajout du fichier à la liste des données musicales
                self.music_data.append({
                    "type": "file",
                    "name": destination_path,
                    "size": size,
                    "mtime": mtime,
                    "length": length,
                    "display_name": display_name
                })

                # Ajouter le nom du fichier dans `existing_files` pour éviter de l'ajouter encore
                existing_files.add(display_name)
            except Exception as e:
                print(f"Erreur lors de l'ajout du fichier {destination_path}: {e}")
                continue

        # Mise à jour de l'interface directement après l'ajout
        self.apply_current_sort()  # Tri en fonction des préférences utilisateur
        self.music_manager.update_music_list()  # Mise à jour de la liste des musiques affichée
        self.music_manager.update_music_counter()  # Mise à jour du compteur de musiques

    def toggle_play_pause(self):
        """Bascule entre lecture et pause avec conservation de la position."""
        if self.playing:
            if self.paused:
                # Reprendre la musique à partir de la position sauvegardée
                mixer.music.stop()  # Stopper proprement toute lecture en cours
                mixer.music.play(start=self.paused_position)  # Reprendre à la position sauvegardée
                self.start_time.restart()  # Redémarrer le timer en conservant la progression
                self.play_button.setIcon(QtGui.QIcon(os.path.join(self.icon_folder, "pause_icon.png")))
                self.paused = False
                self.spectrum.set_paused(False)

                # Redémarrer le SpectrumWorker uniquement si nécessaire
                if not (self.spectrum.spectrum_worker_thread and self.spectrum.spectrum_worker_thread.isRunning()):
                    self.spectrum.restart_spectrum_worker()

                self.progress_timer.start()  # Redémarrer le timer de progression
                print(f"Reprise à la position : {self.paused_position:.2f} secondes")
            else:
                # Mettre en pause et sauvegarder la position actuelle
                self.paused_position += mixer.music.get_pos() / 1000.0  # Ajouter la position actuelle en secondes
                mixer.music.stop()  # Utiliser stop pour arrêter proprement
                self.play_button.setIcon(QtGui.QIcon(os.path.join(self.icon_folder, "play_icon.png")))
                self.paused = True
                self.spectrum.set_paused(True)
                self.progress_timer.stop()  # Arrêter le timer de progression
                print(f"Pause à la position : {self.paused_position:.2f} secondes")
        else:
            # Appeler `play_music` via le gestionnaire de musique
            if 0 <= self.tracklist.currentRow() < len(self.filtered_music_data):
                self.music_manager.play_music()  # Utilise la méthode du gestionnaire
            else:
                if self.filtered_music_data:
                    self.current_track_index = random.randint(0, len(self.filtered_music_data) - 1)
                    self.tracklist.setCurrentRow(self.current_track_index)
                    self.music_manager.play_music()  # Utilise la méthode du gestionnaire
                else:
                    print("Aucune musique disponible pour la lecture.")

    def show_stats_dialog(self):
        """Affiche la fenêtre des statistiques d'écoute."""
        stats = load_stats()
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Statistiques d'écoute")
        dialog.setFixedSize(600, 400)  # Taille fixe de la fenêtre

        layout = QtWidgets.QVBoxLayout(dialog)

        header_label = QtWidgets.QLabel("<h2>Statistiques d'écoute</h2>")
        header_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header_label)

        stats_table = QtWidgets.QTableWidget(dialog)
        stats_table.setColumnCount(3)
        stats_table.setHorizontalHeaderLabels(["Titre", "Nombre d'écoutes", "Temps total"])
        stats_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)  # Empêche l'édition
        stats_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)  # Désactive la sélection
        stats_table.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)  # Retire le focus
        stats_table.horizontalHeader().setStretchLastSection(True)
        stats_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        stats_table.setStyleSheet("""
            QTableWidget {
                background-color: #2A2A3A;
                color: #FFFFFF;
                border: none;
                gridline-color: #444;
            }
            QHeaderView::section {
                background-color: #444;
                color: #FFF;
                padding: 4px;
                border: none;
            }
            QTableWidget::item {
                padding: 5px;
            }
            QTableWidget::item:hover {
                background-color: transparent; /* Supprime l'effet de survol */
            }
        """)

        if stats:
            # Trier les statistiques par temps total d'écoute (descendant)
            sorted_stats = sorted(stats.items(), key=lambda x: -x[1].get("total_time", 0))

            stats_table.setRowCount(len(sorted_stats))
            for row, (track, data) in enumerate(sorted_stats):
                plays = data.get("plays", 0)
                total_time_seconds = data.get("total_time", 0)
                formatted_time = self.format_total_time(total_time_seconds)

                stats_table.setItem(row, 0, QtWidgets.QTableWidgetItem(track))
                stats_table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(plays)))
                stats_table.setItem(row, 2, QtWidgets.QTableWidgetItem(formatted_time))
        else:
            stats_table.setRowCount(0)
            no_data_label = QtWidgets.QLabel("Aucune statistique disponible.")
            no_data_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(no_data_label)

        layout.addWidget(stats_table)

        # Ajouter les boutons
        button_layout = QtWidgets.QHBoxLayout()

        # Bouton pour réinitialiser les statistiques
        reset_button = QtWidgets.QPushButton("Réinitialiser les statistiques", dialog)
        reset_button.setStyleSheet("""
            QPushButton {
                background-color: #E53935;
                color: white;
                font-weight: bold;
                padding: 8px 12px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #FF5C5C;
            }
        """)
        reset_button.clicked.connect(lambda: self.reset_stats(dialog))
        button_layout.addWidget(reset_button)

        # Bouton pour exporter les statistiques
        export_button = QtWidgets.QPushButton("Exporter les statistiques", dialog)
        export_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 8px 12px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #66BB6A;
            }
        """)
        export_button.clicked.connect(self.export_stats)
        button_layout.addWidget(export_button)

        layout.addLayout(button_layout)

        dialog.setLayout(layout)
        dialog.exec()

    def export_stats(self):
        """Exporte les statistiques dans un fichier texte."""
        stats = load_stats()
        if not stats:
            QMessageBox.information(self, "Exporter les statistiques", "Aucune statistique disponible à exporter.")
            return

        # Ouvrir une boîte de dialogue pour sélectionner le dossier de destination
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Exporter les statistiques",
            "",
            "Fichiers texte (*.txt)"
        )
        if not file_path:
            return  # L'utilisateur a annulé
        try:
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write("Statistiques d'écoute :\n")
                file.write("=" * 50 + "\n")
                sorted_stats = sorted(stats.items(), key=lambda x: x[1]["plays"], reverse=True)
                for track, data in sorted_stats:
                    plays = data.get("plays", 0)
                    total_time_seconds = data.get("total_time", 0)
                    formatted_time = self.format_total_time(total_time_seconds)
                    file.write(f"{track}\n  Nombre d'écoutes : {plays}\n  Temps total : {formatted_time}\n\n")
            QMessageBox.information(self, "Exporter les statistiques", f"Statistiques exportées avec succès dans {file_path}.")
        except Exception as e:
            QMessageBox.critical(self, "Erreur", f"Erreur lors de l'exportation : {str(e)}.")

    def reset_stats(self, dialog):
        """Réinitialise les statistiques et met à jour l'affichage."""
        confirm = QMessageBox.question(
            self,
            "Réinitialisation",
            "Voulez-vous réinitialiser toutes les statistiques ?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel
        )
        if confirm == QMessageBox.StandardButton.Yes:
            save_stats({})  # Sauvegarder un dictionnaire vide pour réinitialiser
            dialog.accept()
            QMessageBox.information(self, "Statistiques réinitialisées", "Toutes les statistiques ont été réinitialisées.")

    def format_total_time(self, total_seconds):
        """Formate le temps total en jours, heures, minutes et secondes."""
        days = int(total_seconds // 86400)
        hours = int((total_seconds % 86400) // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        formatted_time = []
        if days > 0:
            formatted_time.append(f"{days} {'jour' if days == 1 else 'jours'}")
        if hours > 0:
            formatted_time.append(f"{hours} {'heure' if hours == 1 else 'heures'}")
        if minutes > 0:
            formatted_time.append(f"{minutes} {'minute' if minutes == 1 else 'minutes'}")
        if seconds > 0 or not formatted_time:
            formatted_time.append(f"{seconds} {'seconde' if seconds == 1 else 'secondes'}")
        return ", ".join(formatted_time)

    def format_time(self, seconds):
        """Formate le temps en minutes:secondes"""
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes}:{seconds:02d}"

    def update_time_label(self, current_time):
        """Met à jour le label affichant le temps écoulé et total"""
        if self.song_length is not None:
            total_time = self.format_time(self.song_length)
            current_time_str = self.format_time(current_time)
            self.time_label.setText(f"{current_time_str} / {total_time}")

    def update_progress_bar(self):
        """Met à jour la barre de progression et le label de temps."""
        if self.playing and not self.paused and self.song_length:
            elapsed_time = self.paused_position + self.start_time.elapsed() / 1000.0  # Temps écoulé en secondes
            if elapsed_time >= self.song_length:
                self.music_manager.handle_track_end()
            else:
                progress_percentage = (elapsed_time / self.song_length) * 1000
                self.progress.setValue(int(progress_percentage))
                self.update_time_label(elapsed_time)  # Met à jour le label avec le temps écoulé

    def toggle_repeat(self):
        # Alterner entre les modes de répétition
        if self.repeat_mode == "none":
            self.repeat_mode = "track"
            icon_path = os.path.join(self.icon_folder, "repeat_track_icon.png")
        elif self.repeat_mode == "track":
            self.repeat_mode = "playlist"
            icon_path = os.path.join(self.icon_folder, "repeat_playlist_icon.png")
        else:
            self.repeat_mode = "none"
            icon_path = os.path.join(self.icon_folder, "repeat_icon.png")
        if os.path.exists(icon_path):
            self.repeat_button.setIcon(QtGui.QIcon(icon_path))
        else:
            print(f"Erreur : l'icône pour le mode '{self.repeat_mode}' est introuvable.")
        self.settings['repeat_mode'] = self.repeat_mode
        save_settings(self.settings)  # Sauvegarde dans les paramètres
        self.apply_repeat_mode()

    def toggle_shuffle_mode(self):
        """Active ou désactive le mode de lecture aléatoire."""
        self.shuffle_mode = not self.shuffle_mode
        self.shuffle_button.setChecked(self.shuffle_mode)
        
        # Mise à jour du style en fonction de l'état
        if self.shuffle_mode:
            self.shuffle_button.setStyleSheet("background-color: #FF5C5C;")  # Couleur activée
        else:
            self.shuffle_button.setStyleSheet("background-color: #D3D3D3;")  # Couleur désactivée

        # Sauvegarder le mode de lecture aléatoire dans les paramètres
        self.settings['shuffle_mode'] = self.shuffle_mode
        save_settings(self.settings)  # Utiliser la fonction importée pour sauvegarder les paramètres

    def sort_by_name(self):
        """Trie les musiques par nom et sauvegarde le mode de tri."""
        self.music_data.sort(key=lambda x: x['display_name'])
        self.settings["sort_order"] = "Nom"
        save_settings(self.settings)  # Sauvegarde le mode de tri
        self.music_manager.update_music_list()

    def sort_by_date(self):
        """Trie les musiques par date de modification et sauvegarde le mode de tri."""
        self.music_data.sort(key=lambda x: x['mtime'])
        self.settings["sort_order"] = "Date de modification"
        save_settings(self.settings)  # Sauvegarde le mode de tri
        self.music_manager.update_music_list()

    def sort_by_size(self):
        """Trie les musiques par taille et sauvegarde le mode de tri."""
        self.music_data.sort(key=lambda x: x['size'])
        self.settings["sort_order"] = "Taille"
        save_settings(self.settings)  # Sauvegarde le mode de tri
        self.music_manager.update_music_list()

    def apply_current_sort(self):
        """Applique le tri selon le mode de tri sélectionné."""
        sort_order = self.settings.get("sort_order", "Nom")  # Défaut : tri par nom
        if sort_order == "Nom":
            self.sort_by_name()
        elif sort_order == "Date de modification":
            self.sort_by_date()
        elif sort_order == "Taille":
            self.sort_by_size()
        else:
            print(f"Mode de tri inconnu : {sort_order}")
       
    def closeEvent(self, event):
        """Gestion de la fermeture de l'application."""
        try:
            if self.playing:
                mixer.music.stop()
            if hasattr(self, 'end_check_timer'):
                self.end_check_timer.stop()
            if hasattr(self, 'spectrum') and self.spectrum.spectrum_worker is not None:
                self.spectrum.spectrum_worker.stop()
            save_settings(self.settings)  # Sauvegarder les paramètres
            if self.executor:
                self.executor.shutdown(wait=True)
            if mixer.get_init():
                mixer.quit()
            event.accept()
        except Exception as e:
            print(f"Erreur lors de la fermeture: {e}")
            event.accept()
