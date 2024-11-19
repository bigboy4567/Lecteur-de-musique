import json
import os
import wave

def load_settings():
    try:
        with open('settings.json', 'r') as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return {
            'always_on_top': False,
            'repeat_mode': "none",
            'shuffle_mode': False,
            'spectrum_color': "#FF0000",
            'background_color': "#2A2A3A"
        }

def save_settings(settings):
    with open('settings.json', 'w') as file:
        json.dump(settings, file)

def load_music_files_from_storage():
    """Charge les musiques depuis le dossier Stockage_Musique ( !!! NE PAS SUPPRIMER !!!) et récupère leurs informations."""
    music_data = []

    # Chemin du dossier de stockage
    storage_path = os.path.join(os.path.expanduser("~"), "Music", "Stockage_Musique ( !!! NE PAS SUPPRIMER !!!)")
    
    # Vérifie si le dossier existe, sinon le crée
    if not os.path.exists(storage_path):
        os.makedirs(storage_path)
    
    # Charger les fichiers .wav depuis le dossier de stockage
    for filename in os.listdir(storage_path):
        if filename.endswith(".wav"):
            file_path = os.path.join(storage_path, filename)
            try:
                with wave.open(file_path, 'rb') as wav_file:
                    length = wav_file.getnframes() / wav_file.getframerate()
                size = os.path.getsize(file_path)
                mtime = os.path.getmtime(file_path)
                display_name = os.path.basename(file_path).replace('.wav', '').lower()
                
                # Ajouter le fichier à la liste des données musicales
                music_data.append({
                    "type": "file",
                    "name": file_path,
                    "size": size,
                    "mtime": mtime,
                    "length": length,
                    "display_name": display_name
                })
            except Exception as e:
                print(f"Erreur lors de l'ajout du fichier {file_path}: {e}")

    return music_data

# Nouvelle fonction pour charger les statistiques d'écoute
def load_stats():
    """Charge les statistiques d'écoute depuis un fichier JSON."""
    try:
        with open('stats.json', 'r') as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

# Nouvelle fonction pour sauvegarder les statistiques d'écoute
def save_stats(stats):
    """Sauvegarde les statistiques d'écoute dans un fichier JSON."""
    with open('stats.json', 'w') as file:
        json.dump(stats, file, indent=4)
