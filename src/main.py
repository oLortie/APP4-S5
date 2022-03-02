import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

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
    plt.gray()
    plt.figure()
    plt.imshow(image)
    plt.title(title)


def filtre_transformation_bilineaire(original_image):
    print("===== Filtrage par transformation bilinéaire =====")

    # Aucune idée si ce résultat là est bon, à vérifier
    num = [0.139, 0.139]
    den = [1, 0.649, 0.099]

    plotFreqz(num, den, "Réponse en fréquence du filtre par transformation bilinéaire", 1600)

    # Filtrage de l'image
    filtered_image = []
    for i in range(len(original_image)):
        filtered_image.append(signal.lfilter(num, den, original_image[i]))

    return filtered_image


def filtre_python(original_image):
    print("===== Filtrage avec les fonction python =====")

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

    plotFreqz(num, den, "Réponse en fréquence du filtre Elliptique", 1600)

    #Filtrage de l'image
    filtered_image = []
    for i in range(len(original_image)):
        filtered_image.append(signal.lfilter(num, den, original_image[i]))

    return filtered_image


def compression(image, percentage):
    print("===== Compression de l'image =====")

    # Matrice de covariance
    cov = np.cov(image)

    # Vecteurs propres
    values, vectors = np.linalg.eig(cov)

    # Matrice de passage
    p_matrix = vectors
    p_matrix_inv = np.linalg.inv(p_matrix)

    # Compresser l'image
    compressed_image = np.dot(image, p_matrix)

    # Fixer à zéro
    nb_lines = int(np.floor(percentage*len(image)))
    for i in range(nb_lines):
        compressed_image[len(compressed_image) - i - 1] = np.zeros(len(compressed_image[0]))

    # Décompresser l'image
    decompressed_image = np.dot(compressed_image, p_matrix_inv)

    return decompressed_image


if __name__ == "__main__":
    print("Début du script...")

    goldhill_noise = np.load('../goldhill_bruit.npy')
    plotImage(goldhill_noise, "Image avec bruit")

    filtered1 = filtre_transformation_bilineaire(goldhill_noise)
    plotImage(filtered1, "Image filtrée avec la transformation bilinéaire")

    filtered2 = filtre_python(goldhill_noise)
    plotImage(filtered2, "Image filtrée avec les fonctions Python")

    decompressed_image = compression(filtered2, 0.7)
    plotImage(decompressed_image, "Image compressée et décompressée")

    plt.show()