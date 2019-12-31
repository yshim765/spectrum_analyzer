import pandas as pd
import numpy as np
import wave
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import sys, argparse

def read_data(path):
    return pd.read_csv(path, header=None)

def plot_spec(i, data, sr, axes):
    """
    data: series, spectrum data
    """
    axes[0].cla()
    axes[1].cla()

    temp = data.iloc[i:i+100]

    fft_temp = np.fft.fft(temp)
    freq = np.fft.fftfreq(temp.shape[0])

    fft_temp = np.array([*fft_temp[int(len(fft_temp)/2):], *fft_temp[:int(len(fft_temp)/2)]])
    freq = np.array([*freq[int(len(freq)/2):], *freq[:int(len(freq)/2)]])


    axes[0].plot(temp)
    axes[1].plot(freq, np.sqrt(fft_temp.real**2 + fft_temp.imag**2))
    axes[0].set_title('i=' + str(i))

    return freq, fft_temp

def main(args):
    path = args.filepath
    sr = args.sr
    windowtype = args.windowtype
    windowsize = args.windowsize

    data = read_data(path)

    data = pd.Series([np.sin(2*np.pi*x/10) + np.sin(2*np.pi*x/100) for x in range(1, 131)] +\
                     [np.sin(2*np.pi*x/5) + np.sin(2*np.pi*x/20) for x in range(1, 131)] +\
                     [np.sin(2*np.pi*x/10) + np.sin(2*np.pi*x/100) for x in range(1, 131)] +\
                     [np.sin(2*np.pi*x/5) + np.sin(2*np.pi*x/20) for x in range(1, 131)])

    N = len(data)

    fig, axes = plt.subplots(2, 1, figsize=[12, 8])
    ani = animation.FuncAnimation(fig, plot_spec, fargs = (data, sr, axes), interval=16, frames=N-100)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="spectrum analyzer")
    parser.add_argument("filepath", help=".wav or .csv file")
    parser.add_argument("--sr", help="sampling rate. default=44100", type=int, default=44100)
    parser.add_argument("--windowtype", help="window type")
    parser.add_argument("--windowsize", help="window size", type=int, default=44100)
    args = parser.parse_args()

    main(args)