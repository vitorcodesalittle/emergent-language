import os

import numpy as np
import torch
from modules.plot import Plot

threshold = 0.05
error_in_dist = 3.5 ##

class Statistics:
    def __init__(self, dir):
        self.dir = dir
        self.utterance_stat = np.zeros(shape=(20))
        self.counter = 0
        self.error_in_dist = error_in_dist


    def calculate(self, epoch_range, batch_range):
        for epoch in epoch_range:
            utterance = Plot.extract_data(epoch, dir=self.dir, calculate_utternace='ON')
            total_iterations = utterance.shape[1]
            for batch in batch_range:
                for iteration in range(total_iterations):
                    self.counter+=1
                    cur_utter = utterance[batch, iteration, :, :]
                    for j in range(cur_utter.shape[0]):
                        for i in range(cur_utter.shape[1]):
                            if cur_utter[j, i] >= threshold:
                                self.utterance_stat[i] += 1
        self.utterance_stat = np.true_divide(self.utterance_stat, self.counter)
        print(self.utterance_stat)

    def calculate_goal_success (self,epoch_range):
        if os.path.isfile('batch_succeed.txt'):
            pass
        else:
            with open('batch_succeed.txt', 'w+') as f:
                pass
        sucess_rate_epoch, sucess_rate_epoch_std = [], []
        error_in_dist = (self.error_in_dist ** 2 * 2) ** 0.5  # euclidean distance
        for epoch in epoch_range:
            dist_from_goal_per_agent = torch.tensor(Plot.extract_data(epoch, dir=self.dir + os.sep, calculate_dist='ON'), dtype=torch.float)
            sucess_rate_batch = torch.sum(dist_from_goal_per_agent <= error_in_dist, dim=1)/dist_from_goal_per_agent.shape[1]
            idx_of_batches_suceeded = (sucess_rate_batch == 1).non
            zero()
            with open('batch_succeed.txt', 'a+') as f:
                f.write('In Epoch {0} the Total num of batches whos agent succeeded is: {1}\n'.format
                        (epoch,idx_of_batches_suceeded.shape[0]))
                f.write ('The batchs whos agents succeeded are:/n')
                for batch in idx_of_batches_suceeded:
                    f.write(str(batch.item()))
                    f.write(',')
                f.write('\n')
            sucess_rate_batch = sucess_rate_batch.type(torch.FloatTensor)
            sucess_rate_epoch += [torch.mean(sucess_rate_batch)]
            sucess_rate_epoch_std += [torch.std(sucess_rate_batch)]
        Plot.create_sucees_reate_plot(sucess_rate_epoch, sucess_rate_epoch_std, epoch_range)

