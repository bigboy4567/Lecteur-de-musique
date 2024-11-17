import os
import random
from PyQt6 import QtWidgets, QtGui, QtCore
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QProgressBar, QFileDialog, QMessageBox, QColorDialog, QSlider, QLabel
from concurrent.futures import ThreadPoolExecutor
from pygame import mixer
from spectrum import AudioSpectrum
from settings_manager import load_settings, save_settings, load_music_files_from_storage, load_stats, save_stats
import wave
import json
import shutil
import numpy as np
from animated_list_widget import AnimatedListWidget
from animated_list_widget_item import AnimatedListWidgetItem

class MusicPlayer(QtWidgets.QMainWindow):
    update_music_list_signal = QtCore.pyqtSignal()
    
    def __init__(self):
        super().__init__()
        
        # Charger les paramètres
        self.settings = load_settings()
        
        # Charger la taille du chunk depuis les paramètres et s'assurer qu'il s'agit bien d'un entier
        self.chunk_size = int(self.settings.get('chunk_size', 16384))
        self.paused_position = 0  # Position sauvegardée lors de la mise en pause

        # Vérification et message de débogage pour confirmer que chunk_size est bien un entier
        if not isinstance(self.chunk_size, int):
            print("Erreur : la taille du chunk n'est pas un entier. Valeur actuelle:", self.chunk_size)
            self.chunk_size = 16384
        else:
            print("Taille du chunk chargée correctement :", self.chunk_size)

        self.setWindowTitle("Acoustiq")
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
        print("Initialisation du spectre audio avec chunk_size:", self.chunk_size)  # Débogage
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
        self.end_check_timer.timeout.connect(self.check_music_end)
        self.end_check_timer.start()
        
        # Initialiser l'interface utilisateur
        self.init_ui()

        # Appliquer la taille du chunk sauvegardée après l'initialisation de l'UI
        self.set_chunk_size(self.chunk_size)
        
        # Mettre à jour la liste de musique et le compteur dès le chargement
        self.update_music_list()  # Actualiser la liste avec les données chargées
        self.update_music_counter()  # Mettre à jour le compteur avec le nombre de musiques

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

    def set_chunk_size(self, size):
        """Change la taille du chunk et met à jour la configuration en temps réel."""
        self.chunk_size = int(size)  # Assurez-vous que size est un entier
        print("Mise à jour de la taille du chunk à :", self.chunk_size)  # Débogage
        self.settings['chunk_size'] = self.chunk_size  # Mettre à jour le paramètre de taille de chunk dans les settings
        save_settings(self.settings)  # Sauvegarde tous les paramètres, y compris la taille des chunks

        # Désélectionner toutes les autres tailles de chunk
        for action in self.chunk_size_menu.actions():
            action.setChecked(int(action.text()) == self.chunk_size)

        # Mettre à jour la taille du chunk dans le spectre audio
        self.spectrum.update_chunk_size(self.chunk_size)

    def refresh_music_folder(self):
        """Recherche et ajoute de nouvelles musiques dans le dossier Stockage_Musique ( !!! NE PAS SUPPRIMER !!!)."""
        # Charger les musiques actuelles depuis le dossier
        new_music_data = load_music_files_from_storage()
        
        # Compter le nombre total de musiques dans le dossier
        total_tracks_found = len(new_music_data)
        
        # Ajouter uniquement les nouvelles musiques qui ne sont pas déjà dans la liste
        existing_files = {music['name'] for music in self.music_data}
        added_tracks = 0
        for track in new_music_data:
            if track['name'] not in existing_files:
                self.music_data.append(track)
                added_tracks += 1
        
        # Mise à jour de la liste et affichage d'un message
        self.update_music_list()
        self.update_music_counter()
        
        # Afficher un message avec le nombre total et le nombre de nouvelles musiques ajoutées
        QMessageBox.information(
            self,
            "Actualisation",
            f"{total_tracks_found} musiques trouvées dans le dossier.\n{added_tracks} nouvelles musiques ajoutées à la bibliothèque."
        )

    def show_about_dialog(self):
        about_text = (
            "<h2>Acoustiq</h2>"
            "<p><strong>Développé par :</strong> Nicolas Q.</p>"
            "<p><strong>Version :</strong> 1.1.0 - Dernière mise à jour : novembre 2024</p>"
            "<p><strong>Licence :</strong> MIT</p>"
            "<p><strong>Compatibilité :</strong> Windows et macOS</p>"
            "<p><strong>Objectif :</strong> Ce projet vise à offrir une expérience musicale immersive grâce à une interface intuitive, "
            "un spectre audio dynamique, et de nouvelles fonctionnalités avancées.</p>"
            "<h3>Guide de démarrage rapide</h3>"
            "<ul>"
            "<li>Charger un fichier WAV en utilisant l'option <em>Ouvrir</em> dans le menu.</li>"
            "<li>Utiliser les commandes de lecture, pause et arrêt pour contrôler la musique.</li>"
            "<li>Accéder au menu <em>Visualisation</em> pour personnaliser les couleurs, la taille des chunks, et bien plus.</li>"
            "</ul>"
            "<p><strong>Formats audio supportés :</strong> WAV uniquement</p>"
            "<h3>Historique des Versions</h3>"
            "<ul>"
            "<li><strong>1.1.0 :</strong> Ajout du menu <em>Visualisation</em>, réorganisation des paramètres, "
            "et optimisation des options de personnalisation du spectre audio.</li>"
            "<li><strong>1.0.1 :</strong> Correction de bugs mineurs, ajout de la fonction pause.</li>"
            "<li><strong>1.0.0 :</strong> Version initiale avec prise en charge des fichiers WAV et spectre audio en temps réel.</li>"
            "</ul>"
            "<h3>Nouveautés :</h3>"
            "<ul>"
            "<li>Ajout d'un menu dédié <em>Visualisation</em> pour personnaliser les couleurs et gérer les options avancées.</li>"
            "<li>Possibilité de changer la couleur du fond et du spectre.</li>"
            "<li>Support pour la gestion avancée de la taille des chunks FFT.</li>"
            "</ul>"
            "<p><strong>Crédits :</strong> Ce projet utilise des bibliothèques open-source telles que <em>PyQt6</em> pour l'interface graphique, "
            "<em>numpy</em> et <em>scipy</em> pour les calculs spectraux, et <em>pygame</em> pour la gestion de l'audio.</p>"
            "<p style='color:red;'>Remarque : Ce lecteur accepte uniquement les fichiers audio au format WAV.</p>"
        )
        QMessageBox.about(self, "À propos", about_text)

    def show_updates_dialog(self):
        updates_text = (
            "<h2>Quoi de neuf ?</h2>"
            "<p><strong>Version actuelle :</strong> 1.2.0 (novembre 2024)</p>"
            "<h3>Dernières mises à jour :</h3>"
            "<ul>"
            "<li><strong>Modes de répétition améliorés :</strong> Les modes de répétition (piste unique, playlist) fonctionnent maintenant de manière fiable dès le lancement de l'application. Le comportement est stable et intuitif lors de la lecture continue ou du retour manuel à une piste précédente.</li>"
            "<li><strong>Correction des statistiques :</strong> Les statistiques d'écoute affichent désormais correctement la durée totale d'écoute pour chaque piste. Les données sont triées automatiquement en fonction du temps total d'écoute (de la plus longue à la plus courte). Les anomalies dans les calculs de durée ont été corrigées.</li>"
            "<li><strong>Amélioration de la pause :</strong> La barre de progression et le spectre audio restent visibles et synchronisés, même après une pause prolongée. Cela garantit une expérience fluide et cohérente.</li>"
            "<li><strong>Synchronisation des temps :</strong> Le compteur de temps associé à la barre de progression est désormais précis et réactif, même lorsque vous changez manuellement de piste via la liste des musiques. Plus aucune déviation entre l'état réel et affiché.</li>"
            "<li><strong>Visualisation persistante :</strong> Le spectre audio reste activé et en fonctionnement après des pauses, des reprises ou des modifications dans la lecture, offrant une continuité visuelle parfaite.</li>"
            "<li><strong>Compatibilité renforcée :</strong> De nombreux ajustements ont été apportés pour améliorer les performances sur macOS et Windows. Les menus, icônes et comportements spécifiques aux systèmes sont désormais homogènes.</li>"
            "<li><strong>Gestion des erreurs :</strong> Les erreurs liées aux fichiers manquants ou endommagés sont maintenant mieux gérées, avec des messages explicites pour guider l'utilisateur.</li>"
            "<li><strong>Mises à jour esthétiques :</strong> Amélioration des styles pour une meilleure lisibilité, y compris dans les menus et dialogues comme celui des statistiques et des nouveautés.</li>"
            "</ul>"
            "<h3>Historique des mises à jour :</h3>"
            "<ul>"
            "<li><strong>1.2.0 :</strong> Modes de répétition avancés, correction et tri des statistiques, synchronisation améliorée des temps, stabilité accrue.</li>"
            "<li><strong>1.1.0 :</strong> Ajout du menu <em>Visualisation</em>, personnalisation des couleurs et gestion avancée des chunks FFT.</li>"
            "<li><strong>1.0.1 :</strong> Correction de bugs mineurs, ajout de la fonction pause, et compatibilité initiale macOS/Windows.</li>"
            "<li><strong>1.0.0 :</strong> Version initiale avec prise en charge des fichiers WAV et spectre audio en temps réel.</li>"
            "</ul>"
        )
        QMessageBox.about(self, "Nouveautés", updates_text)

    def init_executor(self):
        """Initialise le ThreadPoolExecutor pour les tâches en arrière-plan si nécessaire."""
        if self.executor is None:
            self.executor = ThreadPoolExecutor(max_workers=2)

    def add_music_files(self):
        """Ouvre une boîte de dialogue pour ajouter manuellement des fichiers de musique .wav et les ajoute à la liste."""
        # Ouvre une boîte de dialogue pour sélectionner des fichiers .wav
        files, _ = QFileDialog.getOpenFileNames(self, "Sélectionner des fichiers de musique", "", "Fichiers WAV (*.wav)")
 
        if files:  # Si des fichiers sont sélectionnés
            # Chemin du dossier de stockage
            storage_path = os.path.join(os.path.expanduser("~"), "Music", "Stockage_Musique ( !!! NE PAS SUPPRIMER !!!)")
 
            # Vérifie si le dossier existe, sinon le crée
            if not os.path.exists(storage_path):
                os.makedirs(storage_path)
 
            # Copie les fichiers vers le dossier de stockage et les ajoute à la liste
            for file_path in files:
                destination_path = os.path.join(storage_path, os.path.basename(file_path))
                if not os.path.exists(destination_path):
                    shutil.copy(file_path, destination_path)
                self.init_executor()  # Initialiser l'exécuteur si nécessaire
                self.executor.submit(self.add_files_async, [destination_path])  # Ajoute le fichier dans la liste
 
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
        refresh_action.triggered.connect(self.refresh_music_folder)
        settings_menu.addAction(refresh_action)

        # Menu Visualisation
        visualization_menu = menubar.addMenu("Visualisation")

        # Option pour changer la couleur du spectre
        self.change_color_action = QAction("Changer la couleur du spectre", self)
        self.change_color_action.triggered.connect(self.change_spectrum_color)
        visualization_menu.addAction(self.change_color_action)

        self.played_tracks_history = []  # Historique des indices des pistes jouées

        # Option pour changer la couleur de fond
        self.change_background_color_action = QAction("Changer la couleur de fond", self)
        self.change_background_color_action.triggered.connect(self.change_background_color)
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

        # Menu Nouveautés
        updates_menu = QAction("Nouveautés", self)
        updates_menu.triggered.connect(self.show_updates_dialog)
        menubar.addAction(updates_menu)

        # Menu À propos
        about_action = QAction("À propos", self)
        about_action.triggered.connect(self.show_about_dialog)
        menubar.addAction(about_action)

        # Couleurs du spectre et de fond à partir des paramètres
        spectrum_color = self.settings.get("spectrum_color", "#FF0000")
        background_color = self.settings.get("background_color", "#2A2A3A")

        # Initialisation de l'affichage du spectre
        self.spectrum = AudioSpectrum(self)
        self.spectrum.set_spectrum_color(QtGui.QColor(spectrum_color))
        self.spectrum.set_background_color(QtGui.QColor(background_color))

        # Ajouter les raccourcis clavier
        self.shortcut_play_pause = QtGui.QShortcut(QtGui.QKeySequence("Space"), self)
        self.shortcut_play_pause.activated.connect(self.toggle_play_pause)

        self.shortcut_next = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Right"), self)
        self.shortcut_next.activated.connect(self.play_next_music)

        self.shortcut_previous = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Left"), self)
        self.shortcut_previous.activated.connect(self.play_previous_music)

        self.shortcut_volume_up = QtGui.QShortcut(QtGui.QKeySequence("Up"), self)
        self.shortcut_volume_up.activated.connect(self.increase_volume)

        self.shortcut_volume_down = QtGui.QShortcut(QtGui.QKeySequence("Down"), self)
        self.shortcut_volume_down.activated.connect(self.decrease_volume)

        # Initialisation de la liste de pistes avec un style animé
        self.tracklist = AnimatedListWidget(self)
        self.tracklist.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.tracklist.itemDoubleClicked.connect(self.play_music)

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
        self.search_var.textChanged.connect(self.update_music_list)

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
        self.select_all_button.clicked.connect(self.select_all_tracks)

        self.add_music_button = QtWidgets.QPushButton("Ajouter des musiques")
        self.add_music_button.setStyleSheet(button_style)
        self.add_music_button.clicked.connect(self.add_music_files)

        self.delete_button = QtWidgets.QPushButton("Supprimer")
        self.delete_button.setStyleSheet(delete_button_style)
        self.delete_button.clicked.connect(self.delete_selected_music)

        self.restore_button = QtWidgets.QPushButton("Restaurer")
        self.restore_button.setStyleSheet(button_style)
        self.restore_button.clicked.connect(self.restore_music)
        self.restore_button.setEnabled(False)

        self.play_button = QtWidgets.QPushButton()
        self.play_button.setIcon(QtGui.QIcon(os.path.join(self.icon_folder, "play_icon.png")))
        self.play_button.setIconSize(QtCore.QSize(30, 30))
        self.play_button.setStyleSheet(control_button_style)
        self.play_button.clicked.connect(self.toggle_play_pause)

        self.forward_button = QtWidgets.QPushButton()
        self.forward_button.setIcon(QtGui.QIcon(os.path.join(self.icon_folder, "forward_icon.png")))
        self.forward_button.setIconSize(QtCore.QSize(30, 30))
        self.forward_button.setStyleSheet(control_button_style)
        self.forward_button.clicked.connect(self.play_next_music)

        self.backward_button = QtWidgets.QPushButton()
        self.backward_button.setIcon(QtGui.QIcon(os.path.join(self.icon_folder, "backward_icon.png")))
        self.backward_button.setIconSize(QtCore.QSize(30, 30))
        self.backward_button.setStyleSheet(control_button_style)
        self.backward_button.clicked.connect(self.play_previous_music)
        
        # Bouton de répétition
        repeat_button_style = button_style.replace("#4CAF50", "#FF9800").replace("#45A049", "#FB8C00")
        self.repeat_button = QtWidgets.QPushButton()
        self.repeat_button.setStyleSheet(repeat_button_style)
        self.repeat_button.setIcon(QtGui.QIcon(os.path.join(self.icon_folder, "repeat_icon.png")))
        self.repeat_button.setIconSize(QtCore.QSize(30, 30))
        self.repeat_button.clicked.connect(self.toggle_repeat)

        # Menu Statistiques
        stats_menu = QAction("Statistiques", self)
        stats_menu.triggered.connect(self.show_stats_dialog)
        menubar.addAction(stats_menu)

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
        self.update_music_list_signal.connect(self.update_music_list)
    
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

    def change_spectrum_color(self):
        """Ouvre un dialogue de couleur pour sélectionner la couleur du spectre."""
        # Créer un sélecteur de couleur en tant que boîte de dialogue modale
        color_dialog = QColorDialog(self)
        color_dialog.setOption(QColorDialog.ColorDialogOption.ShowAlphaChannel, False)  # Masquer l'option d'opacité si non nécessaire
        color_dialog.setWindowTitle("Choisissez la couleur du spectre")
        color_dialog.setCurrentColor(QtGui.QColor(self.settings.get("spectrum_color", "#FF0000")))

        if color_dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            # Si une couleur est sélectionnée, appliquer la couleur au spectre
            color = color_dialog.selectedColor()
            self.spectrum.set_spectrum_color(color)
            self.settings['spectrum_color'] = color.name()
            save_settings(self.settings)  # Utiliser la fonction importée pour sauvegarder les paramètres

    def select_all_tracks(self):
        """Sélectionne tous les morceaux de la liste, même si le mode de sélection est unique."""
        # Passer en mode sélection multiple temporairement
        self.tracklist.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.MultiSelection)
        for i in range(self.tracklist.count()):
            item = self.tracklist.item(i)
            item.setSelected(True)
    
        # Rétablir le mode de sélection unique
        self.tracklist.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)

    def delete_selected_music(self):
        """Supprime les morceaux sélectionnés."""
        selected_items = self.tracklist.selectedItems()
        if not selected_items:
            QMessageBox.information(self, "Suppression", "Aucun morceau sélectionné pour suppression.")
            return
        confirm = QMessageBox.question(
            self,
            "Supprimer",
            f"Voulez-vous supprimer les {len(selected_items)} morceaux sélectionnés ?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel
        )
        if confirm == QMessageBox.StandardButton.Yes:
            for item in selected_items:
                # Trouver l'élément `music_item` par son nom de fichier (`name`)
                music_item = next(
                    (m for m in self.music_data if m['name'] == next(
                        (f['name'] for f in self.filtered_music_data if f['display_name'] == item.text()), None)
                    ), None
                )
                if music_item:
                    # Supprimer l'élément trouvé
                    self.deleted_music_backup.insert(0, music_item)
                    self.music_data.remove(music_item)

            # Mettre à jour l'affichage sans sauvegarde
            self.update_music_list()
            self.restore_button.setEnabled(bool(self.deleted_music_backup))
            self.update_music_counter()

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

        # Redémarrer les timers ou ajuster les comportements si nécessaire
        if not self.playing:
            print("La lecture n'est pas en cours. Les modifications seront appliquées lors de la prochaine lecture.")

    def delete_music(self):
        """Supprime la piste sélectionnée de la liste."""
        current_row = self.tracklist.currentRow()
        if current_row >= 0:
            confirm = QMessageBox.question(
                self,
                "Supprimer",
                "Voulez-vous supprimer la piste sélectionnée ?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel
            )
            if confirm == QMessageBox.StandardButton.Yes:
                # Obtenir l'index réel dans self.music_data
                music_item = self.filtered_music_data[current_row]
                index_in_music_data = self.music_data.index(music_item)
                self.deleted_music_backup.insert(0, self.music_data[index_in_music_data])
                if len(self.deleted_music_backup) > 40:  # Limite le nombre de pistes sauvegardées
                    self.deleted_music_backup.pop()
                del self.music_data[index_in_music_data]
                self.update_music_list()
                self.restore_button.setEnabled(True)
                self.update_music_counter()

    def restore_music(self):
        """Restaure la dernière piste supprimée."""
        if self.deleted_music_backup:
            self.music_data.append(self.deleted_music_backup.pop(0))
            self.update_music_list()
            self.update_music_counter()
            self.restore_button.setEnabled(bool(self.deleted_music_backup))

    def load_folders(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Sélectionner des fichiers de musique", "", "Fichiers WAV (*.wav)")
        if files:
            self.init_executor()
            self.executor.submit(self.add_files_async, files)

    def add_files_async(self, files):
        """Ajoute les fichiers de musique sélectionnés à la liste en arrière-plan et les copie dans le dossier Stockage_Musique ( !!! NE PAS SUPPRIMER !!!), en évitant les doublons."""
        
        # Chemin du dossier de stockage
        storage_path = os.path.join(os.path.expanduser("~"), "Music", "Stockage_Musique ( !!! NE PAS SUPPRIMER !!!)")
        
        # Vérifie si le dossier existe, sinon le crée
        if not os.path.exists(storage_path):
            os.makedirs(storage_path)
        
        # Récupère tous les noms de fichiers déjà présents dans `self.music_data` pour éviter les doublons
        existing_files = {music['display_name'] for music in self.music_data}

        for track in files:
            # Créer le chemin de destination dans `Stockage_Musique ( !!! NE PAS SUPPRIMER !!!)`
            destination_path = os.path.join(storage_path, os.path.basename(track))
            display_name = os.path.basename(track).replace('.wav', '').lower()

            # Vérifie si le fichier est déjà dans la liste des musiques
            if display_name in existing_files:
                print(f"Le fichier {display_name} est déjà présent. Il ne sera pas ajouté de nouveau.")
                continue

            # Copier le fichier dans le dossier `Stockage_Musique ( !!! NE PAS SUPPRIMER !!!)` s’il n'est pas déjà à cet emplacement
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

        # Appliquer le tri selon le mode de tri actuel
        self.apply_current_sort()

        # Mise à jour de l'interface
        self.update_music_counter()
        self.update_music_list_signal.emit()  # Mise à jour de la liste des musiques affichée

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
            if 0 <= self.tracklist.currentRow() < len(self.filtered_music_data):
                self.play_music()  # Lancer la lecture si rien ne joue actuellement
            else:
                if self.filtered_music_data:
                    self.current_track_index = random.randint(0, len(self.filtered_music_data) - 1)
                    self.tracklist.setCurrentRow(self.current_track_index)
                    self.play_music()
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

    def play_music(self):
        try:
            # Vérifiez si une piste est sélectionnée dans la liste
            if 0 <= self.tracklist.currentRow() < len(self.filtered_music_data):
                # Ajouter l'indice actuel à l'historique si ce n'est pas déjà fait
                if self.current_track_index != -1 and (not self.played_tracks_history or self.played_tracks_history[-1] != self.current_track_index):
                    self.played_tracks_history.append(self.current_track_index)
                
                # Arrêter le suivi du temps pour l'ancien morceau, s'il existe
                if self.playing:
                    self.stop_tracking_time()

                # Mettre à jour l'index de la piste actuelle
                self.current_track_index = self.tracklist.currentRow()
                
                # Récupération des informations sur la piste
                selected_track = self.filtered_music_data[self.current_track_index]
                track_name = selected_track.get("name")
                display_name = selected_track.get("display_name", "Inconnu")  # Utiliser "Inconnu" comme valeur par défaut
                
                # Vérification de l'existence et de la validité du fichier
                if not track_name or not os.path.exists(track_name):
                    QMessageBox.warning(self, "Erreur", f"Le fichier {track_name} est introuvable ou invalide.")
                    return
                
                try:
                    # Calculer la durée totale de la piste
                    with wave.open(track_name, 'rb') as wav_file:
                        frame_rate = wav_file.getframerate()
                        num_frames = wav_file.getnframes()
                        self.song_length = num_frames / frame_rate
                except wave.Error:
                    QMessageBox.warning(self, "Erreur", "Le fichier sélectionné n'est pas un fichier audio valide.")
                    return
                
                # Charger et lire le fichier audio
                mixer.music.load(track_name)
                if self.paused and self.paused_position > 0:
                    # Reprendre depuis la position sauvegardée
                    mixer.music.play(start=self.paused_position)
                    print(f"Reprise à partir de la position : {self.paused_position:.2f} secondes.")
                else:
                    # Lecture depuis le début
                    mixer.music.play()
                    self.paused_position = 0  # Réinitialiser la position si une nouvelle lecture commence

                self.spectrum.load_audio_file(track_name)

                # Initialisation de l'état de lecture
                self.start_time = QtCore.QElapsedTimer()
                self.start_time.start()
                self.progress_timer.start()
                self.update_play_stats(display_name)
                self.current_track_label.setText(display_name)
                self.play_button.setIcon(QtGui.QIcon(os.path.join(self.icon_folder, "pause_icon.png")))
                self.playing = True
                self.paused = False
                self.spectrum.set_paused(False)
                
                # Mettre à jour le temps total pour l'affichage dès que la piste change
                self.update_time_label(0)  # Met à jour le label de temps à 0
                
                # Démarrer le suivi du temps d'écoute
                self.start_tracking_time(display_name)

                print(f"Lecture de {display_name} ({self.song_length:.2f} secondes).")
            else:
                print("Aucune piste sélectionnée ou liste vide.")
        except Exception as e:
            print(f"Erreur lors de la lecture du fichier : {e}")
            self.playing = False
            self.paused = False
            self.play_button.setIcon(QtGui.QIcon(os.path.join(self.icon_folder, "play_icon.png")))
            self.progress_timer.stop()

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
                self.handle_track_end()
            else:
                progress_percentage = (elapsed_time / self.song_length) * 1000
                self.progress.setValue(int(progress_percentage))
                self.update_time_label(elapsed_time)  # Met à jour le label avec le temps écoulé

    def handle_track_end(self):
        """Gère la fin de la piste en fonction du mode de répétition."""
        # Arrête le suivi du temps pour la piste terminée
        self.stop_tracking_time()

        if self.repeat_mode == "track":
            # Rejoue la même piste
            if 0 <= self.current_track_index < len(self.filtered_music_data):
                self.tracklist.setCurrentRow(self.current_track_index)  # Assure que la bonne piste est sélectionnée
                self.play_music()  # Relance la musique actuelle
            else:
                print("Aucune piste sélectionnée ou liste vide.")
        elif self.repeat_mode == "playlist":
            # Passe à la piste suivante ou revient au début si nécessaire
            self.play_next_music()
        else:
            # Mode normal : arrête si c'est la dernière piste
            if self.current_track_index < len(self.filtered_music_data) - 1:
                self.play_next_music()
            else:
                # Fin de la lecture
                self.play_button.setIcon(QtGui.QIcon(os.path.join(self.icon_folder, "play_icon.png")))
                self.playing = False
                self.progress.setValue(0)
                self.progress_timer.stop()
                print("Lecture terminée.")

    def start_tracking_time(self, track_name):
        """Démarre ou reprend le suivi du temps d'écoute pour une piste."""
        # Vérifiez si current_track_name existe et si le morceau change
        if getattr(self, 'current_track_name', None) != track_name:
            # Si une nouvelle piste est jouée, réinitialiser le temps cumulé
            self.track_cumulative_time = 0

        # Assigner le nom du morceau actuel
        self.current_track_name = track_name
        
        # Initialiser et démarrer le timer
        self.track_play_start_time = QtCore.QElapsedTimer()
        self.track_play_start_time.start()

        print(f"Suivi du temps démarré pour la piste : {track_name}")

    def stop_tracking_time(self):
        """Arrête le suivi du temps d'écoute et met à jour les statistiques."""
        if not self.track_play_start_time or not self.current_track_name:
            return  # Assurez-vous que tout est initialisé correctement

        # Calculer le temps depuis le dernier démarrage
        elapsed_time = self.track_play_start_time.elapsed() / 1000.0  # Temps écoulé en secondes
        
        # Empêcher les temps négatifs ou anormaux
        if elapsed_time <= 0 or elapsed_time > self.song_length * 2:  # Par sécurité, limite à 2 fois la durée du morceau
            print(f"Temps anormal détecté : {elapsed_time}s. Ignoré.")
            return
        
        # Charger les statistiques actuelles
        stats = load_stats()

        # Ajouter la piste aux statistiques si elle n'existe pas encore
        if self.current_track_name not in stats:
            stats[self.current_track_name] = {"plays": 0, "total_time": 0}

        # Ajouter le temps d'écoute uniquement si la piste est en cours de lecture
        stats[self.current_track_name]["total_time"] += elapsed_time
        stats[self.current_track_name]["plays"] += 1

        # Sauvegarder les statistiques
        save_stats(stats)

        # Réinitialiser le suivi
        self.track_play_start_time = None
        self.current_track_name = None

        print(f"Temps d'écoute ajouté : {elapsed_time:.2f}s pour {self.current_track_name}")

    def check_music_end(self):
        """Vérifie si la musique est terminée et gère la fin de la piste."""
        if self.playing and not self.paused and not mixer.music.get_busy():
            self.handle_track_end()

    def play_next_music(self):
        """Passe à la musique suivante."""
        if not self.filtered_music_data:
            return

        # Si la lecture est active, arrêtez le suivi du temps pour la piste actuelle
        if self.playing:
            self.stop_tracking_time()

        # Passez à l'index suivant ou au début si en mode répétition de playlist
        if self.shuffle_mode:
            self.current_track_index = random.randint(0, len(self.filtered_music_data) - 1)
        else:
            self.current_track_index += 1
            if self.current_track_index >= len(self.filtered_music_data):
                if self.repeat_mode == "playlist":
                    self.current_track_index = 0
                else:
                    self.current_track_index = len(self.filtered_music_data) - 1
                    self.stop_music()
                    return

        # Sélectionnez la piste dans la liste et jouez-la
        self.tracklist.setCurrentRow(self.current_track_index)
        self.play_music()

    def stop_music(self):
        """Arrête la musique et met à jour les statistiques."""
        if self.playing:
            self.stop_tracking_time()

        # Arrêter la musique
        mixer.music.stop()
        self.playing = False
        self.paused = False
        self.progress_timer.stop()
        self.play_button.setIcon(QtGui.QIcon(os.path.join(self.icon_folder, "play_icon.png")))

    def play_previous_music(self):
        """Passe à la musique précédente."""
        if self.played_tracks_history:
            self.current_track_index = self.played_tracks_history.pop()
        else:
            self.current_track_index -= 1
            if self.current_track_index < 0:
                self.current_track_index = 0
                return

        self.tracklist.setCurrentRow(self.current_track_index)
        self.play_music()

    def toggle_repeat(self):
        print("toggle_repeat appelé")  # Vérifie si le bouton est cliqué

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

        # Appliquer l'icône correspondante
        if os.path.exists(icon_path):
            self.repeat_button.setIcon(QtGui.QIcon(icon_path))
        else:
            print(f"Erreur : l'icône pour le mode '{self.repeat_mode}' est introuvable.")

        # Sauvegarder le nouveau mode dans les paramètres
        self.settings['repeat_mode'] = self.repeat_mode
        save_settings(self.settings)  # Sauvegarde dans les paramètres

        # Appliquer immédiatement les changements du mode de répétition
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
        self.update_music_list()

    def sort_by_date(self):
        """Trie les musiques par date de modification et sauvegarde le mode de tri."""
        self.music_data.sort(key=lambda x: x['mtime'])
        self.settings["sort_order"] = "Date de modification"
        save_settings(self.settings)  # Sauvegarde le mode de tri
        self.update_music_list()

    def sort_by_size(self):
        """Trie les musiques par taille et sauvegarde le mode de tri."""
        self.music_data.sort(key=lambda x: x['size'])
        self.settings["sort_order"] = "Taille"
        save_settings(self.settings)  # Sauvegarde le mode de tri
        self.update_music_list()

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

    def update_music_list(self):
        """Met à jour la liste des pistes en fonction du texte de recherche."""
        search_text = self.search_var.text().lower()
        self.tracklist.clear()
        self.filtered_music_data = [
            item for item in self.music_data if search_text in item['display_name']
        ]
        for item in self.filtered_music_data:
            self.tracklist.addItem(item['display_name'])
            
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

    def update_music_counter(self):
        """Met à jour le compteur des pistes."""
        self.track_counter.setText(f"Total de musiques : {len(self.music_data)}")