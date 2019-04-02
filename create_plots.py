import time
from modules.plot import Plot


def main():
    start = time.time()
    for epoch in range(0,1000,50):
        Plot.create_plots(epoch, 10)
    end = time.time()
    print("create_plots: "+str(end - start))
    start = time.time()
    # Plot.create_video()
    end = time.time()
    print("create_video: " + str(end - start))


if __name__ == "__main__":
    main()


