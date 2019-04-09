import argparse
import re
import time
from modules.plot import Plot
import os
from pathlib import Path

max_batch = 10
epoch_num = 10
DIR_REGEX = '\d*-\d*'
epoch_range = range(0, 25, 5)


def get_newest_h5_dir():
    all_subdirs = [d for d in os.listdir('.') if os.path.isdir(d) and re.search(DIR_REGEX, d)]
    return max(all_subdirs, key=os.path.getmtime)


def main(dir, batch_size):
    if not os.path.isabs(dir):
        dir = str(Path(os.getcwd())) + os.sep + dir + os.sep
    start = time.time()
    for epoch in epoch_range:
        Plot.create_plots(epoch, batch_size)
    end = time.time()
    print("create_plots: " + str(end - start))
    start = time.time()
    Plot.create_video(max_batch, epoch_num, dir)
    end = time.time()
    print("create_video: " + str(end - start))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create plot from h5 files')
    parser.add_argument('--dir', required=False, type=str, default=get_newest_h5_dir(),
                        help='Directory to folder containing h5 files')
    parser.add_argument('--batch_size', required=False, type=str, default=512,
                        help='Batch size')
    args = parser.parse_args()
    main(args.dir, args.batch_size)
