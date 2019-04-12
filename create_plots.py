import argparse
import re
import time

from Statistics import Statistics
from modules.plot import Plot
import os
from pathlib import Path

batch_range = range(10)
epoch_num = 10
DIR_REGEX = '\d*-\d*'
epoch_range = range(0, 360, 50)


def get_newest_h5_dir():
    all_subdirs = [d for d in os.listdir('.') if os.path.isdir(d) and re.search(DIR_REGEX, d)]
    return max(all_subdirs, key=os.path.getmtime)


def main(dir, batch_size):
    # if not os.path.isabs(dir):
    #     dir = str(Path(os.getcwd())) + os.sep + dir + os.sep
    # os.chdir(dir)
    # start = time.time()
    # for epoch in epoch_range:
    #     Plot.create_plots(epoch, batch_size)
    # end = time.time()
    # print("create plots: " + str(end - start))
    # start = time.time()
    # Plot.create_video(batch_range, epoch_num, dir)
    # end = time.time()
    # print("create video: " + str(end - start))
    start = time.time()
    stats = Statistics(dir)
    stats.calculate(epoch_range, batch_range)
    end = time.time()
    print("create statistics: " + str(end - start))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create plot from h5 files')
    parser.add_argument('--dir', required=False, type=str, default=get_newest_h5_dir(),
                        help='Directory to folder containing h5 files')
    parser.add_argument('--batch_size', required=False, type=str, default=512,
                        help='Batch size')
    args = parser.parse_args()
    args.dir = '/home/yael/Documents/Yael - Private/Workspace/emergent-language/2057-10042019/'
    args.batch_size = 15
    main(args.dir, args.batch_size)
