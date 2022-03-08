################################################
# Date 8 mars 2022
# Titre du fichier: main.py
# Auteurs: Paul du Réau de la Gaignonnière et Olivier Lortie
# CIP: durp0701 et loro0801
################################################

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from src.zplane import zplane


def plotFreqz(num, den, title, Fe, displayAngle=True):
    w, h = signal.freqz(num, den)
    fig, ax1 = plt.subplots()
    ax1.set_title(title)
    ax1.plot(w*Fe/(2*np.pi), 20 * np.log10(abs(h)), 'b')
    ax1.set_ylabel('Amplitude [dB]', color='b')
    ax1.set_xlabel('Frequence [Hz]')

    if displayAngle:
        ax2 = ax1.twinx()
        angles = np.unwrap(np.angle(h))
        ax2.plot(w*Fe/(2*np.pi), angles, 'g')
        ax2.set_ylabel('Angle (radians)', color='g')
        ax2.grid()
        ax2.axis('tight')

    return


def plotImage(image, title):
    plt.figure()
    plt.gray()
    plt.imshow(image)
    plt.title(title)


def aberrations(img_with_aberrations):
    print("===== Correction des aberrations =====")

    num = np.poly([0.9 * np.exp(1j * np.pi / 2), 0.9 * np.exp((-1) * 1j * np.pi / 2),
                          0.95 * np.exp(1j * np.pi / 8), 0.95 * np.exp((-1) * 1j * np.pi / 8)])
    den = np.poly([0, -0.99, -0.99, 0.8])

    plt.figure()
    zplane(num, den)

    image_without_aberrations = signal.lfilter(den, num, img_with_aberrations)

    return image_without_aberrations


def rotation(img_with_rotation):
    print("===== Rotation de 90 vers la droite =====")

    # Inverser la taille des x et y
    img_without_rotation = np.zeros((int(len(img_with_rotation[0])), int(len(img_with_rotation))))

    # Matrice de rotation
    rot_matrix = [[0, 1], [-1, 0]]
    for y in range(int(len(img_with_rotation))):
        for x in range(int(len(img_with_rotation[0]))):
            [[new_x], [new_y]] = np.matmul(rot_matrix, [[x], [y]])
            new_y += len(img_with_rotation) - 1
            img_without_rotation[new_x][new_y] = img_with_rotation[x][y]

    return img_without_rotation


def filtre_transformation_bilineaire(img_with_noise):
    print("===== Filtrage par transformation bilinéaire =====")

    num = [0.418, 0.837, 0.418]
    den = [1, 0.463, 0.21]

    plt.figure()
    zplane(num, den)

    plotFreqz(num, den, "Réponse en fréquence du filtre par transformation bilinéaire", 1600)

    return signal.lfilter(num, den, img_with_noise)


def filtre_python(img_with_noise):
    print("===== Filtrage avec les fonctions python =====")

    # filtre butterworth
    order_butter, wn_butter = signal.buttord(wp=500, ws=750, gpass=0.2, gstop=60, analog=False, fs=1600)
    print("Ordre du filtre butterworth: ", order_butter)

    # filtre chebyshev type 1
    order_cheby1, wn_cheby1 = signal.cheb1ord(wp=500, ws=750, gpass=0.2, gstop=60, analog=False, fs=1600)
    print("Ordre du filtre chebyshev type 1: ", order_cheby1)

    # filtre chebyshev type 2
    order_cheby2, wn_cheby2 = signal.cheb2ord(wp=500, ws=750, gpass=0.2, gstop=60, analog=False, fs=1600)
    print("Ordre du filtre chebyshev type 2: ", order_cheby2)

    # filtre elliptique
    order_ellip, wn_ellip = signal.ellipord(wp=500, ws=750, gpass=0.2, gstop=60, analog=False, fs=1600)
    print("Ordre du filtre elliptique: ", order_ellip)

    # Le filtre elliptique est celui ayant le plus petit ordre, on fait donc un filtre elliptique
    num, den = signal.ellip(N=order_ellip, rp=0.1, rs=60, Wn=500, btype='lowpass', analog=False, output='ba', fs=1600)

    plt.figure()
    zplane(num, den)

    plotFreqz(num, den, "Réponse en fréquence du filtre Elliptique", 1600)

    return signal.lfilter(num, den, img_with_noise)


def compression(image, percentage):
    print("===== Compression de l'image =====")

    # Matrice de covariance
    cov = np.cov(image)

    # Vecteurs propres
    values, vectors = np.linalg.eig(cov)

    # Matrice de passage
    p_matrix = np.transpose(vectors)
    p_matrix_inv = np.linalg.inv(p_matrix)

    # Compresser l'image
    compressed_image = np.matmul(p_matrix, image)

    # Fixer à zéro
    nb_lines = int(np.floor(percentage*len(image)))
    for i in range(nb_lines):
        compressed_image[len(compressed_image) - i - 1] = np.zeros(len(compressed_image[0]))

    # Décompresser l'image
    decompressed_image = np.matmul(p_matrix_inv, compressed_image)

    return decompressed_image, compressed_image


if __name__ == "__main__":
    # Séquence cas par cas

    # # Aberrations
    # img_with_aberrations = np.load("../goldhill_aberrations.npy")
    # img_without_aberrations = aberrations(img_with_aberrations)
    # plotImage(img_with_aberrations, 'Avec aberrations')
    # plotImage(img_without_aberrations, 'Sans aberrations')
    #
    # # Rotation
    # img_with_rotation = mpimg.imread('../goldhill_rotate.png')
    # img_with_rotation = np.mean(img_with_rotation, -1)  # Enlève la 3D
    # img_without_rotation = rotation(img_with_rotation)
    # plotImage(img_with_rotation, 'Sans rotation')
    # plotImage(img_without_rotation, 'Avec rotation de 90 degrés vers la droite')
    #
    # # Debruitage de l'image
    # img_with_noise = np.load('../goldhill_bruit.npy')
    # filtered1 = filtre_transformation_bilineaire(img_with_noise)
    # plotImage(filtered1, "Image filtrée avec la transformation bilinéaire")
    # filtered2 = filtre_python(img_with_noise)
    # plotImage(filtered2, "Image filtrée avec les fonctions Python")
    #
    # # Compression
    # image = mpimg.imread('../goldhill.png')
    # decompressed_image_50, compressed_image_50 = compression(image, 0.5)
    # decompressed_image_70, compressed_image_70 = compression(image, 0.7)
    # plotImage(decompressed_image_50, "Image décompressée avec 50%")
    # plotImage(decompressed_image_70, "Image décompressée avec 70%")

    # Séquence complète

    complete_image = np.load('../image_complete.npy')

    # Aberrations
    plotImage(complete_image, 'Avec aberrations')
    complete_image = aberrations(complete_image)
    plotImage(complete_image, 'Sans aberrations et sans rotation')

    # Rotation
    complete_image = rotation(complete_image)
    plotImage(complete_image, 'Avec rotation de 90 degrés vers la droite')

    # Débruitage
    complete_image_1 = complete_image
    complete_image_1 = filtre_transformation_bilineaire(complete_image_1)
    plotImage(complete_image_1, 'Image sans le bruit avec la transformation bilinéaire')
    complete_image = filtre_python(complete_image)
    plotImage(complete_image, 'Image sans le bruit avec les fonctions python')

    # Compression
    decompressed_image_50, compressed_image_50 = compression(complete_image, 0.5)
    decompressed_image_70, compressed_image_70 = compression(complete_image, 0.7)
    plotImage(decompressed_image_50, "Image décompressée 50%")
    plotImage(decompressed_image_70, "Image décompressée 70%")

    plt.show()
