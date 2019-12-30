import pandas as pd
import numpy as np
import wave

import sys, argparse

def main(args):
    print(args.filepath)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", help=".wav or .csv file")
    args = parser.parse_args()

    main(args)