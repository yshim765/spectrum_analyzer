import pandas as pd
import numpy as np
#import librosa
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import sys, os, argparse

def read_data(path, sr):
    ext = os.path.splitext(path)[1][1:]
    
    if ext == "csv":
        data = pd.read_csv(path, header=None)
    elif ext == "wav":
        raise(Exception(".wav does not implemented"))
    else:
        raise(Exception("Unexpected extension"))

    return data.values.reshape(len(data))

def plot_spec(i, data, sr, windowtype, windowsize, axes):
    """
    data: series, spectrum data
    """

    window_dict = {
        "hamming":np.hamming,
        "hanning":np.hanning,
        "rectangule":(lambda x: np.ones(x))
    }

    window_func = window_dict[windowtype]

    axes[0].cla()
    axes[1].cla()

    temp = window_func(windowsize) * data[i:i+windowsize]

    fft_temp = np.fft.fft(temp)
    freq = np.fft.fftfreq(temp.shape[0], d=1/sr)

    fft_temp = np.array(fft_temp[0 <= freq])
    freq = np.array(freq[0 <= freq])

    axes[0].plot(temp)
    axes[1].plot(freq, np.sqrt(fft_temp.real**2 + fft_temp.imag**2))
    axes[0].set_title('i=' + str(i))

    return freq, fft_temp

def main(args):
    path = args.filepath
    
    if args.usesettings:
        pass
    else:
        sr = args.sr
        windowtype = args.windowtype
        windowsize = args.windowsize

    data = read_data(path, sr)

    N = len(data)

    fig, axes = plt.subplots(2, 1, figsize=[12, 8])
    ani = animation.FuncAnimation(fig, plot_spec, fargs = (data, sr, windowtype, windowsize, axes), interval=1000/16, frames=N-windowsize)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="spectrum analyzer")
    parser.add_argument("filepath", help=".wav or .csv file")
    parser.add_argument("--sr", type=int, default=44100, help="sampling rate. default=44100")
    parser.add_argument("--windowtype", default="hamming", choices=["hamming", "hanning", "rectangule"], help="window type. default=hamming. can use hamming, hanning, rectangule")
    parser.add_argument("--windowsize", type=int, default=100, help="window size. default=100")
    parser.add_argument("--usesettings", action='store_true', help="use saved setting file")
    parser.add_argument("--savesettings", action='store_true', help="save settings to file")
    
    args = parser.parse_args()

    main(args)