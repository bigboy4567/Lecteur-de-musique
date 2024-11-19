# Ce projet est sous licence MIT.
# Copyright (c) 2024 Nicolas Q.


import os
import random
import shutil
import wave
from PyQt6 import QtCore, QtGui, QtWidgets
from pygame import mixer
from settings_manager import save_stats, load_stats, load_music_files_from_storage

class MusicManager:
    def __init__(self, parent):
        self.parent = parent  # Référence à l'instance du lecteur pour accéder à ses attributs

    def update_music_counter(self):
        """Met à jour le compteur des pistes."""
        self.parent.track_counter.setText(f"Total de musiques : {len(self.parent.music_data)}")

    def update_music_list(self):
        """Met à jour la liste des pistes en fonction du texte de recherche."""
        search_text = self.parent.search_var.text().lower()
        self.parent.tracklist.clear()
        self.parent.filtered_music_data = [
            item for item in self.parent.music_data if search_text in item['display_name']
        ]
        for item in self.parent.filtered_music_data:
            self.parent.tracklist.addItem(item['display_name'])

    def play_previous_music(self):
        """Passe à la musique précédente."""
        try:
            # Vérifie si l'historique des pistes existe et contient des éléments
            if hasattr(self.parent, 'played_tracks_history') and self.parent.played_tracks_history:
                self.parent.current_track_index = self.parent.played_tracks_history.pop()
            else:
                # Si l'historique est vide ou n'existe pas, passe à l'index précédent
                self.parent.current_track_index -= 1
                if self.parent.current_track_index < 0:
                    self.parent.current_track_index = 0
                    return

            # Met à jour la sélection de la liste de pistes
            self.parent.tracklist.setCurrentRow(self.parent.current_track_index)

            # Lance la lecture de la piste précédente
            self.play_music()

            # Débogage : affiche l'état de l'historique après modification
            print(f"Nouvel historique des pistes : {self.parent.played_tracks_history}")

        except AttributeError as e:
            # Gestion d'erreur si l'attribut n'existe pas
            print(f"Erreur : {e}. Vérifiez que 'played_tracks_history' est bien initialisé dans MusicPlayer.")
        except Exception as e:
            # Gestion d'autres exceptions éventuelles
            print(f"Une erreur inattendue s'est produite : {e}")

    def play_next_music(self):
        """Passe à la musique suivante."""
        if not self.parent.filtered_music_data:
            return

        if self.parent.playing:
            self.stop_tracking_time()

        if self.parent.shuffle_mode:
            self.parent.current_track_index = random.randint(0, len(self.parent.filtered_music_data) - 1)
        else:
            self.parent.current_track_index += 1
            if self.parent.current_track_index >= len(self.parent.filtered_music_data):
                if self.parent.repeat_mode == "playlist":
                    self.parent.current_track_index = 0
                else:
                    self.parent.current_track_index = len(self.parent.filtered_music_data) - 1
                    self.parent.stop_music()
                    return

        self.parent.tracklist.setCurrentRow(self.parent.current_track_index)
        self.play_music()

    def play_music(self, force_restart=False):
        """Joue la musique sélectionnée ou redémarre la piste actuelle si nécessaire."""
        try:
            # Vérifiez si une piste est sélectionnée dans la liste
            if 0 <= self.parent.tracklist.currentRow() < len(self.parent.filtered_music_data):
                # Initialiser played_tracks_history s'il n'existe pas
                if not hasattr(self.parent, 'played_tracks_history'):
                    self.parent.played_tracks_history = []

                # Vérifiez si on est en mode répétition de piste ou si une nouvelle piste est sélectionnée
                if self.parent.repeat_mode == "track" and not force_restart:
                    print("Mode répétition de piste activé. Redémarrage de la piste.")
                else:
                    # Mettre à jour l'index de la piste actuelle uniquement si ce n'est pas une répétition
                    self.parent.current_track_index = self.parent.tracklist.currentRow()
                    print(f"Piste sélectionnée : {self.parent.current_track_index}")

                # Ajouter l'indice actuel à l'historique si ce n'est pas déjà fait
                if (
                    self.parent.current_track_index != -1 and
                    (not self.parent.played_tracks_history or self.parent.played_tracks_history[-1] != self.parent.current_track_index)
                ):
                    self.parent.played_tracks_history.append(self.parent.current_track_index)

                # Arrêter le suivi du temps pour l'ancien morceau, s'il existe
                if self.parent.playing:
                    self.stop_tracking_time()

                # Récupération des informations sur la piste
                selected_track = self.parent.filtered_music_data[self.parent.current_track_index]
                track_name = selected_track.get("name")
                display_name = selected_track.get("display_name", "Inconnu")  # Utiliser "Inconnu" comme valeur par défaut

                # Vérification de l'existence et de la validité du fichier
                if not track_name or not os.path.exists(track_name):
                    QtWidgets.QMessageBox.warning(self.parent, "Erreur", f"Le fichier {track_name} est introuvable ou invalide.")
                    return

                try:
                    # Calculer la durée totale de la piste
                    with wave.open(track_name, 'rb') as wav_file:
                        frame_rate = wav_file.getframerate()
                        num_frames = wav_file.getnframes()
                        self.parent.song_length = num_frames / frame_rate
                except wave.Error:
                    QtWidgets.QMessageBox.warning(self.parent, "Erreur", "Le fichier sélectionné n'est pas un fichier audio valide.")
                    return

                # Charger et lire le fichier audio
                mixer.music.load(track_name)

                if self.parent.paused and self.parent.paused_position > 0:
                    mixer.music.play(start=self.parent.paused_position)
                else:
                    mixer.music.play()
                    self.parent.paused_position = 0

                # Initialisation de l'état de lecture
                self.parent.start_time = QtCore.QElapsedTimer()
                self.parent.start_time.start()
                self.parent.progress_timer.start()
                self.parent.update_play_stats(display_name)
                self.parent.current_track_label.setText(display_name)
                self.parent.play_button.setIcon(QtGui.QIcon(os.path.join(self.parent.icon_folder, "pause_icon.png")))
                self.parent.playing = True
                self.parent.paused = False

                # Réinitialiser les données du spectre avant de charger la nouvelle musique
                self.parent.spectrum.reset_spectrum_data()

                # Charger et démarrer le spectre audio avec la nouvelle musique
                QtCore.QTimer.singleShot(0, lambda: self.parent.spectrum.load_audio_file(track_name))
                QtCore.QTimer.singleShot(0, lambda: self.parent.spectrum.restart_spectrum_worker())

                # Mettre à jour le temps total pour l'affichage dès que la piste change
                self.parent.update_time_label(0)

                # Démarrer le suivi du temps d'écoute
                self.start_tracking_time(display_name)
            else:
                print("Aucune piste sélectionnée ou liste vide.")
        except AttributeError as e:
            print(f"Erreur d'attribut : {e}. Vérifiez l'initialisation des attributs nécessaires.")
        except Exception as e:
            print(f"Erreur lors de la lecture du fichier : {e}")
            self.parent.playing = False
            self.parent.paused = False
            self.parent.play_button.setIcon(QtGui.QIcon(os.path.join(self.parent.icon_folder, "play_icon.png")))
            self.parent.progress_timer.stop()

    def check_music_end(self):
        """Vérifie si la musique est terminée et gère la fin de la piste."""
        if self.parent.playing and not self.parent.paused and not mixer.music.get_busy():
            self.handle_track_end()

    def start_tracking_time(self, track_name):
        """Démarre ou reprend le suivi du temps d'écoute pour une piste."""
        if getattr(self.parent, 'current_track_name', None) != track_name:
            self.parent.track_cumulative_time = 0

        self.parent.current_track_name = track_name
        self.parent.track_play_start_time = QtCore.QElapsedTimer()
        self.parent.track_play_start_time.start()

    def stop_tracking_time(self):
        """Arrête le suivi du temps d'écoute et met à jour les statistiques."""
        if not self.parent.track_play_start_time or not self.parent.current_track_name:
            return

        elapsed_time = self.parent.track_play_start_time.elapsed() / 1000.0
        if elapsed_time <= 0 or elapsed_time > self.parent.song_length * 2:
            return

        stats = load_stats()
        if self.parent.current_track_name not in stats:
            stats[self.parent.current_track_name] = {"plays": 0, "total_time": 0}

        stats[self.parent.current_track_name]["total_time"] += elapsed_time
        stats[self.parent.current_track_name]["plays"] += 1

        save_stats(stats)

        self.parent.track_play_start_time = None
        self.parent.current_track_name = None

    def handle_track_end(self):
        """Gère la fin de la piste en fonction du mode de répétition."""
        # Arrête le suivi du temps pour la piste terminée
        self.stop_tracking_time()

        # Vérifie le mode de répétition
        if self.parent.repeat_mode == "track":
            # Rejoue la même piste
            print("Mode répétition de piste activé. Redémarrage de la piste actuelle.")
            if 0 <= self.parent.current_track_index < len(self.parent.filtered_music_data):
                self.parent.tracklist.setCurrentRow(self.parent.current_track_index)
                self.play_music(force_restart=True)
        elif self.parent.repeat_mode == "playlist":
            # Passe à la piste suivante ou revient au début si nécessaire
            self.play_next_music()
        else:
            # Aucun mode de répétition, arrêter la lecture
            print("Aucun mode de répétition actif. Arrêt de la lecture.")
            self.parent.play_button.setIcon(QtGui.QIcon(os.path.join(self.parent.icon_folder, "play_icon.png")))
            self.parent.playing = False
            self.parent.progress.setValue(0)
            self.parent.progress_timer.stop()

    def restore_music(self):
        """Restaure la dernière piste supprimée."""
        if self.parent.deleted_music_backup:
            self.parent.music_data.append(self.parent.deleted_music_backup.pop(0))
            self.update_music_list()
            self.update_music_counter()
            self.parent.restore_button.setEnabled(bool(self.parent.deleted_music_backup))

    def refresh_music_folder(self):
        """Recherche et ajoute de nouvelles musiques dans le dossier Stockage_Musique."""
        # Charger les musiques actuelles depuis le dossier
        new_music_data = load_music_files_from_storage()
        
        # Compter le nombre total de musiques dans le dossier
        total_tracks_found = len(new_music_data)
        
        # Ajouter uniquement les nouvelles musiques qui ne sont pas déjà dans la liste
        existing_files = {music['name'] for music in self.parent.music_data}
        added_tracks = 0
        for track in new_music_data:
            if track['name'] not in existing_files:
                self.parent.music_data.append(track)
                added_tracks += 1
        
        # Mise à jour de la liste et affichage d'un message
        self.update_music_list()
        self.update_music_counter()
        
        # Afficher un message avec le nombre total et le nombre de nouvelles musiques ajoutées
        QtWidgets.QMessageBox.information(
            self.parent,
            "Actualisation",
            f"{total_tracks_found} musiques trouvées dans le dossier.\n{added_tracks} nouvelles musiques ajoutées à la bibliothèque."
        )

    def delete_music(self):
        """Supprime la piste sélectionnée de la liste."""
        current_row = self.parent.tracklist.currentRow()
        if current_row >= 0:
            confirm = QtWidgets.QMessageBox.question(
                self.parent,
                "Supprimer",
                "Voulez-vous supprimer la piste sélectionnée ?",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.Cancel
            )
            if confirm == QtWidgets.QMessageBox.StandardButton.Yes:
                music_item = self.parent.filtered_music_data[current_row]
                index_in_music_data = self.parent.music_data.index(music_item)
                self.parent.deleted_music_backup.insert(0, self.parent.music_data[index_in_music_data])
                if len(self.parent.deleted_music_backup) > 40:
                    self.parent.deleted_music_backup.pop()
                del self.parent.music_data[index_in_music_data]
                self.update_music_list()
                self.parent.restore_button.setEnabled(True)
                self.update_music_counter()

    def delete_selected_music(self):
        """Supprime les morceaux sélectionnés."""
        selected_items = self.parent.tracklist.selectedItems()
        if not selected_items:
            QtWidgets.QMessageBox.information(
                self.parent, 
                "Suppression", 
                "Aucun morceau sélectionné pour suppression."
            )
            return

        confirm = QtWidgets.QMessageBox.question(
            self.parent,
            "Supprimer",
            f"Voulez-vous supprimer les {len(selected_items)} morceaux sélectionnés ?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.Cancel
        )
        if confirm == QtWidgets.QMessageBox.StandardButton.Yes:
            for item in selected_items:
                # Trouver l'élément à supprimer dans music_data
                music_item = next(
                    (m for m in self.parent.music_data if m['name'] == next(
                        (f['name'] for f in self.parent.filtered_music_data if f['display_name'] == item.text()), 
                        None
                    )),
                    None
                )
                if music_item:
                    # Ajouter à la sauvegarde et supprimer de la liste principale
                    self.parent.deleted_music_backup.insert(0, music_item)
                    self.parent.music_data.remove(music_item)

            # Mettre à jour l'interface
            self.update_music_list()
            self.parent.restore_button.setEnabled(bool(self.parent.deleted_music_backup))
            self.update_music_counter()

    def add_music_files(self):
        """Ouvre une boîte de dialogue pour ajouter manuellement des fichiers de musique .wav et les ajoute à la liste."""
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(self.parent, "Sélectionner des fichiers de musique", "", "Fichiers WAV (*.wav)")
        if files:
            storage_path = os.path.join(os.path.expanduser("~"), "Music", "Stockage_Musique")

            # Crée le dossier de stockage si nécessaire
            if not os.path.exists(storage_path):
                os.makedirs(storage_path)

            # Ajoute les fichiers au dossier de stockage et à la liste
            for file_path in files:
                destination_path = os.path.join(storage_path, os.path.basename(file_path))
                if not os.path.exists(destination_path):
                    shutil.copy(file_path, destination_path)

                # Ajouter les fichiers à la liste des musiques
                self.parent.music_data.append({
                    "type": "file",
                    "name": destination_path,
                    "size": os.path.getsize(destination_path),
                    "mtime": os.path.getmtime(destination_path),
                    "display_name": os.path.basename(destination_path).replace('.wav', '')
                })

            # Mettre à jour l'interface
            self.update_music_list()
            self.update_music_counter()

    def select_all_tracks(self):
        """Sélectionne tous les morceaux de la liste."""
        # Passer en mode sélection multiple temporairement
        self.parent.tracklist.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.MultiSelection)
        for i in range(self.parent.tracklist.count()):
            item = self.parent.tracklist.item(i)
            item.setSelected(True)

        # Rétablir le mode de sélection unique
        self.parent.tracklist.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
