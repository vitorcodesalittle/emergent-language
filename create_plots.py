import argparse
import os
import re
from Statistics import Statistics
from modules.plot import Plot
from pathlib import Path

DIR_REGEX = '\d*-\d*'
def get_newest_dir():
    all_subdirs = [d for d in os.listdir('.') if os.path.isdir(d) and re.search(DIR_REGEX, d)]
    if len(all_subdirs) == 0:
        pass
    else:
        return max(all_subdirs, key=os.path.getmtime)

parser = argparse.ArgumentParser(description='Create plot from h5 files')
parser.add_argument('--dir', required=False, type=str, default=get_newest_dir(),help='Directory to folder containing h5 files')
parser.add_argument('--batch-size', required=False, type=int, default=5, help='Batch size')
parser.add_argument('--epoch-range', required=False, default = range(0,500,25), help='define how we sampeled the epochs for plotting')
parser.add_argument('--batch-range', required=False, default = range(10), help='' )


def main(args):
    dir = args.dir
    batch_size = args.batch_size
    epoch_range = args.epoch_range
    batch_range = args.batch_range
    if not os.path.isabs(dir):
         dir = str(Path(os.getcwd())) + os.sep + dir + os.sep
    os.chdir(dir)
    for epoch in epoch_range:
        Plot.create_plots(epoch, batch_size)
    # Plot.create_video(batch_range, epoch_num, dir) #fix
    stats = Statistics(dir)
    stats.calculate(epoch_range, batch_range)
    stats.calculate_goal_success(epoch_range)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)




