import numpy as np
import matplotlib.pyplot as plt
import os

def bmode_simple(image_complex, img_coord_x, img_coord_z, dynamic_range=60, title=None):
    """
    Fonction pour afficher une image échographique (B-mode).
    Convertit les données complexes en échelle logarithmique (dB).
    """
    # 1. Calcul de l'enveloppe (module du signal complexe)
    # On ajoute une petite valeur (epsilon) pour éviter log(0)
    img_abs = np.abs(image_complex) + 1e-12

    # 2. Conversion en décibels (Log-compression)
    bmode = 20 * np.log10(img_abs)

    # 3. Normalisation : on met le maximum à 0 dB
    bmode = bmode - np.max(bmode)

    # 4. Affichage
    # L'extent définit les limites physiques des axes [xmin, xmax, zmax, zmin]
    # Note: zmax et zmin sont inversés car l'axe Z (profondeur) va vers le bas
    extent = [np.min(img_coord_x), np.max(img_coord_x),
              np.max(img_coord_z), np.min(img_coord_z)]

    kwargs = {'vmin': -dynamic_range, 
              'vmax': 0, 
              'cmap': 'gray', 
              'extent': extent}

    fig, ax = plt.subplots(figsize=(8, 6))
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
    folder_out = 'example_abdominal_wall_data_out'

    print(f"Chargement des données depuis '{folder_out}'...")

    try:
        # Chargement des fichiers numpy
        img_out = np.load(os.path.join(folder_out, 'data.npy'))
        x_coord_out = np.load(os.path.join(folder_out, 'x_coord.npy'))
        z_coord_out = np.load(os.path.join(folder_out, 'z_coord.npy'))

        print("Données chargées avec succès.")
        print(f"Dimensions de l'image : {img_out.shape}")
        print(f"Plage X : de {x_coord_out.min():.4f} à {x_coord_out.max():.4f} m")
        print(f"Plage Z : de {z_coord_out.min():.4f} à {z_coord_out.max():.4f} m")

        # Affichage
        bmode_simple(img_out, x_coord_out, z_coord_out, dynamic_range=60, title="Image Reconstruite (CPU)")

    except FileNotFoundError as e:
        print("\nERREUR : Impossible de trouver les fichiers de données.")
        print(f"Vérifie que le dossier '{folder_out}' existe et contient data.npy, x_coord.npy et z_coord.npy")
        print(f"Détail de l'erreur : {e}")