import time
from modules.plot import Plot


def main():
    start = time.time()
    Plot.create_plots(1, 30)
    end = time.time()
    print("create_plots: "+str(end - start))
    start = time.time()
    Plot.create_video()
    end = time.time()
    print("create_video: " + str(end - start))


if __name__ == "__main__":
    main()


