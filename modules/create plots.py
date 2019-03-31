import time

from modules.plot import Plot

start = time.time() #to calculate how long the code runs
Plot.create_plots()
end = time.time()
print(end - start)


