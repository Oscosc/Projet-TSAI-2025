"""Code permettant d'afficher une image échographique en B-mode"""

import os

import numpy as np
import matplotlib.pyplot as plt

def aff_res(image_complex, img_coord_x, img_coord_z, dynamic_range=60, title=None, is_load = False):
    """
    Fonction pour afficher une image échographique (B-mode).
    Convertit les données complexes en échelle logarithmique (dB).
    """

    if not is_load :
        image_complex = np.load(image_complex)
        img_coord_x = np.load(img_coord_x)
        img_coord_z = np.load(img_coord_z)


    # Calcul de l'enveloppe (module du signal complexe)
    # On ajoute une valeur (epsilon) pour éviter log(0)
    img_abs = np.abs(image_complex) + 1e-12

    # Conversion en décibels (Log-compression)
    bmode = 20 * np.log10(img_abs)

    # Normalisation : on met le maximum à 0 dB
    bmode = bmode - np.max(bmode)

    # Affichage
    # L'extent définit les limites physiques des axes [xmin, xmax, zmax, zmin]
    # zmax et zmin sont inversés car l'axe Z (profondeur) va vers le bas
    extent = [np.min(img_coord_x), np.max(img_coord_x),
              np.max(img_coord_z), np.min(img_coord_z)]

    kwargs = {'vmin': -dynamic_range,
              'vmax': 0, 
              'cmap': 'gray', 
              'extent': extent}

    _, ax = plt.subplots(figsize=(8, 6))
    # On transpose (.T) car imshow attend (Y, X) alors que les données sont souvent (X, Z)
    im = ax.imshow(bmode.T, **kwargs)

    ax.set_xlabel('Position latérale x [m]')
    ax.set_ylabel('Profondeur z [m]')

    if title:
        ax.set_title(title)

    # Barre de couleur pour voir l'échelle dB
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Amplitude [dB]')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Nom du dossier contenant les résultats (doit correspondre au script précédent)
    FOLDER_OUT = '../data/example_abdominal_wall_data_out'

    print(f"Chargement des données depuis '{FOLDER_OUT}'...")

    try:
        # Chargement des fichiers numpy
        img_out = np.load(os.path.join(FOLDER_OUT, 'data.npy'))
        x_coord_out = np.load(os.path.join(FOLDER_OUT, 'x_coord.npy'))
        z_coord_out = np.load(os.path.join(FOLDER_OUT, 'z_coord.npy'))

        print("Données chargées avec succès.")
        print(f"Dimensions de l'image : {img_out.shape}")
        print(f"Plage X : de {x_coord_out.min():.4f} à {x_coord_out.max():.4f} m")
        print(f"Plage Z : de {z_coord_out.min():.4f} à {z_coord_out.max():.4f} m")

        # Affichage
        aff_res(img_out, x_coord_out, z_coord_out, dynamic_range=60, 
                     title="Image Reconstruite (CPU)", is_load = True)

    except FileNotFoundError as e:
        print("\nERREUR : Impossible de trouver les fichiers de données.")
        print(f"Vérifie que le dossier '{FOLDER_OUT}' existe et contient data.npy,\
               x_coord.npy et z_coord.npy")
        print(f"Détail de l'erreur : {e}")
