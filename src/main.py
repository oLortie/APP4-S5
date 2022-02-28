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


def filtre_transformation_bilineaire():
    print("===== Filtrage par transformation bilinéaire =====")
    # Aucune idée si ce résultat là est bon, à vérifier
    num = [0.139, 0.139]
    den = [1, 0.649, 0.099]

    plotFreqz(num, den, "Réponse en fréquence du filtre par transformation bilinéaire", 1600)

    return


def filtre_python():
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

    return


if __name__ == "__main__":
    print("Début du script...")

    filtre_transformation_bilineaire()
    filtre_python()

    plt.show()