import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from src.zplane import zplane


def aberrations():
    numerateur = np.poly([0.9 * np.exp(1j * np.pi / 2), 0.9 * np.exp((-1) * 1j * np.pi / 2),
                          0.95 * np.exp(1j * np.pi / 8), 0.95 * np.exp((-1) * 1j * np.pi / 8)])
    denominateur = np.poly([0, -0.99, 0.8])

    plt.figure()
    zplane(numerateur, denominateur)

    w, h = signal.freqz(numerateur, denominateur)

    plt.figure()
    plt.title('Filtre numérique pour les aberrations')
    plt.plot(w, 20 * np.log10(abs(h)))
    plt.ylabel('Amplitude [dB]')
    plt.xlabel('Frequency [rad/sample]')

    plt.gray()
    img_load = np.load("../goldhill_aberrations.npy")
    print(img_load)
    image_sans_aberrations = signal.lfilter(denominateur, numerateur, img_load)

    mpimg.imsave('../goldhill_avec_aberrations.png', img_load)
    mpimg.imsave('../goldhill_sans_aberrations.png', image_sans_aberrations)


def rotation():
    plt.gray()
    img_couleur = mpimg.imread('../goldhill_rotate.png')

    # Inverser la taille des x et y
    img_rotate = np.zeros((int(len(img_couleur[0])), int(len(img_couleur))))
    for y in range(int(len(img_couleur))):
        for x in range(int(len(img_couleur[0]))):
            img_rotate[x][int(len(img_couleur)-1-y)] = img_couleur[y][x][0]  # Car l'image est en 3D? Voir l'array...

    mpimg.imsave('../goldhill_avec_rotation.png', img_rotate)


if __name__ == "__main__":
    # aberrations()
    rotation()

    plt.show()
