from modules.plot import Plot
import numpy as np

threshold = 0.05


class Statistics:
    def __init__(self, dir):
        self.dir = dir
        self.utterance_stat = np.zeros(shape=(20))
        self.counter = 0

    def calculate(self, epoch_range, batch_range):
        for epoch in epoch_range:
            locations, colors, shapes, num_agents, utterance, goals_by_landmark = Plot.extract_data(epoch, dir=self.dir)
            total_iterations = locations.shape[1]
            for batch in batch_range:
                for iteration in range(total_iterations):
                    self.counter+=1
                    cur_utter = utterance[batch, iteration, :, :]
                    for j in range(cur_utter.shape[0]):
                        for i in range(cur_utter.shape[1]):
                            if cur_utter[j, i] >= threshold:
                                self.utterance_stat[i] += 1
            np.true_divide(self.utterance_stat, self.counter)
            print(self.utterance_stat)
