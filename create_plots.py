import time
from modules.plot import Plot
import os
max_batch = 10
epoch_num = 10
from pathlib import Path

def main():
    folder_name = '2111-05042019' #enter the folder name here
    folder_dir = str(Path(os.getcwd())) + os.sep + folder_name + os.sep
    os.chdir(folder_dir)
    start = time.time()
    for epoch in range(0,25,5):
        Plot.create_plots(epoch, 10)
    end = time.time()
    print("create_plots: " +str(end - start))
    start = time.time()
    Plot.create_video(max_batch, epoch_num, folder_dir)
    end = time.time()
    print("create_video: " + str(end - start))


if __name__ == "__main__":
    main()



