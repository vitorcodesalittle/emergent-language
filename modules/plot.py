import datetime
import os

import numpy
import numpy as np
import matplotlib.pyplot as plt
import progressbar
from time import sleep
import subprocess
from pathlib import Path
import h5py
import torch
from torch import Tensor

dict_colors = {'[0.]': 'red', '[1.]': 'green', '[2.]': 'blue'}
dict_shapes = {'[0.]': 'o', '[1.]': 'v'}
epoch = -1


def save_dataset(file_name, datasetname, dataset, mode):
    with h5py.File(file_name, mode) as hf:
        epoch = len(list(hf.keys()))
        datasetname = datasetname + str(epoch)
        hf.create_dataset(datasetname, data=dataset)
    return epoch

def open_dataset(file_name):
    with h5py.File(file_name, 'r') as hf:
        utterance_file_name = list(hf.keys())[epoch]
        return np.array(hf[utterance_file_name])

class Plot:
    def __init__(self, batch_num, total_iteration, num_locations, location_dim, world_dim, num_agents, goals,
                 landmarks_location, folder_dir):
        self.batch_num = batch_num
        self.total_iteration = total_iteration + 1
        self.world_dim = world_dim
        self.num_agents = num_agents
        self.location_matrix = np.zeros(shape=(self.batch_num, self.total_iteration , num_locations, location_dim)) # total_iteration + 1 - so it will include the 'start',
        self.color_matrix = np.zeros(shape=(self.batch_num, num_locations, 1))
        self.shape_matrix = np.zeros(shape=(self.batch_num, num_locations, 1))
        self.goal_matrix = goals
        self.landmarks_location = landmarks_location
        self.folder_dir = folder_dir

    def save_goals(self):
        save_dataset(self.folder_dir + '.\goals.h5', 'goals', self.goal_matrix, 'w')
        save_dataset(self.folder_dir + '.\landmarks_location.h5', 'landmarks_location', self.landmarks_location, 'w')
        goals_by_landmark = torch.zeros(size=(self.batch_num, self.num_agents, 2), dtype=torch.int32)

        for batch in range(0, 512):
            for agent in range(self.num_agents):
                agent_goal_x = self.goal_matrix[batch, agent, 0]
                agent_goal_y = self.goal_matrix[batch, agent, 1]
                goal_on = self.goal_matrix[batch, agent, 2]
                # find close landmark
                for i in range(self.landmarks_location.shape[1]):
                    if numpy.isclose(self.landmarks_location[batch,i,0].item(), agent_goal_x, rtol=1e-01, atol=1e-01, equal_nan=False) \
                            and numpy.isclose(self.landmarks_location[batch,i,1].item(), agent_goal_y, rtol=1e-01, atol=1e-01, equal_nan=False):
                        goals_by_landmark[batch, agent, 1] = int(goal_on)
                        goals_by_landmark[batch, agent, 0] = i + self.num_agents + 1
        save_dataset(self.folder_dir + '.\goals_by_landmark.h5', 'goals_by_landmark', goals_by_landmark, 'w')

    def save_utterance_matrix(self,utterance, iteration):
        if iteration == 0:
            self.utterance_matrix = np.zeros(shape=(self.batch_num, self.total_iteration, self.num_agents, utterance.shape[2]))
        elif iteration < self.total_iteration - 2:
            self.utterance_matrix[:,iteration + 1, :, :] = utterance.detach().numpy()
        else:
            self.utterance_matrix[:,iteration + 1, :, :] = utterance.detach().numpy()
            if os.path.isfile(self.folder_dir + '.\sentence.h5'):
                # locations, colors, shapes, num_agents = Plot.extract_data_locations()
                save_dataset(self.folder_dir + '.\sentence.h5', 'sentence', self.utterance_matrix, 'a')

            else:
                save_dataset(self.folder_dir + '.\sentence.h5', 'sentence', self.utterance_matrix, 'w')

    def save_h5_file(self, mode):
        save_dataset(self.folder_dir + '.\locations.h5', 'location', self.location_matrix, mode)
        save_dataset(self.folder_dir + '.\colors.h5', 'colors', self.color_matrix, mode)
        save_dataset(self.folder_dir + '.\shape.h5', 'shape', self.shape_matrix, mode)
        save_dataset(self.folder_dir + '.\players.h5', 'players', self.num_agents, mode)

    def save_plot_matrix(self, iteration, locations, colors, shapes):
        if iteration == 'start':
            self.save_goals() # Curently only prints the goals, TODO: save the goals to first plot
            self.location_matrix[:,0,:,:] = locations
            self.color_matrix[:, :, :] = colors
            self.shape_matrix[:, :, :] = shapes
            self.goal_matrix[:, :, :] = self.goal_matrix

        elif iteration < self.total_iteration - 2:
            self.location_matrix[:, iteration + 1, :, :] = locations.detach().numpy()
        else:
            self.location_matrix[:, iteration + 1, :, :] = locations.detach().numpy()
            if os.path.isfile(self.folder_dir + os.sep + '.\locations.h5'):
                self.save_h5_file('a')
            else:
                self.save_h5_file('w')

    @staticmethod
    def create_video(max_batch, max_epoch, folder_dir):
        for epoch in range(max_epoch):
            for batch in range(max_batch):
                # create a video from all pictures in movies_dir of the format
                # 'batchnum_batchiter_{int}d.png'
                cmd = 'ffmpeg -f image2 -r 1/2 -i "'+ folder_dir + 'movies' + os.sep + 'epoch_' \
                      +str(epoch)+'batchnum_'+str(batch)+'iter_%d.png" -vcodec mpeg4 -y movie{:02d}_{:02d}.mp4'.format(epoch, batch)
                cmd = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE,
                                       shell=True,
                                       cwd=os.getcwd())
                out, err = cmd.communicate()
                if cmd.returncode == 0:
                    pass  # success
                else:
                    print("ERROR")
                    print(out)
                    print(err)




    @staticmethod
    def create_plots(epoch, batch_size):

        #extracting the matrices containing the data from the files
        locations, colors, shapes, num_agents, utterance, goals_by_landmark = Plot.extract_data(epoch)
        utterance_legand = np.zeros(shape=(locations.shape[2],utterance.shape[3])) #locations.shape[2] = num of entitels, utterance.shape[3] = vcob size

        #labels for the entitles, will be used in the plot
        text_label = Plot.creating_dot_label(locations.shape[2], num_agents)  #locations.shape[1] = num of entitels
        #opening a status bar
        bar = progressbar.ProgressBar(maxval=batch_size, \
                                      widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        #creating the plots

        total_iterations = locations.shape[1]
        for batch in range(batch_size):
            marker = np.array([dict_shapes[str(shape)] for shape in np.array(shapes[batch])])
            colors_plot = np.array([dict_colors[str(color)] for color in np.array(colors[batch])])
            title = ""
            for agent in range(num_agents):
                title += "the Goal of agent {0} is that agent {1} will reach LM {1}\n"\
                    .format(agent + 1, goals_by_landmark[batch, agent, 0], goals_by_landmark[batch, agent, 1])
            for iteration in range(total_iterations):
                plt.clf()
                fig, ax = plt.subplots()
                plt.axis([0, 16, 0, 16])
                locations_y = locations[batch, iteration, :, 1]
                locations_x = locations[batch, iteration, :, 0]
                utterance_legand[:num_agents] = utterance[batch, iteration, :, :]
                for obj in range(len(locations_y)):
                    ax.scatter(locations_x[obj], locations_y[obj], color = colors_plot[obj], marker = marker[obj],
                                    label = np.around(utterance_legand[obj], decimals = 3))
                    ax.annotate(text_label[obj], (locations_x[obj]+0.05, locations_y[obj]+0.05))
                    plt.draw()
                # Shrink current axis's height by 10% on the bottom ,  so the legand will not be over the plot
                box = ax.get_position()
                ax.set_position([box.x0, box.y0 + box.height * 0.05, box.width, box.height * 0.9])
                # Put a legend to the right of the current axis
                ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
                          fancybox=True, shadow=True, ncol=2 ,prop={'size': 5})
                plt.title(title)
                plt.savefig('plots' + os.sep + 'epoch_{0}batchnum_{1}iter_{2}.png'.format(epoch, batch, iteration))
            bar.update(batch + 1)
            sleep(0.1)
        bar.finish()

    @staticmethod
    def extract_data (epoch):
        #extracting the matrices containing the data from the file
        return open_dataset('.\locations.h5'), open_dataset('.\colors.h5'), \
               open_dataset('.\shape.h5'), open_dataset('.\players.h5'), \
               open_dataset('.\sentence.h5'), open_dataset('.\goals_by_landmark.h5')

    @staticmethod
    def creating_dot_label (entitle, num_agents):
        text_label = [None] * entitle
        text_label[:num_agents] = [str(num + 1) for num in range(num_agents)]
        for landmark in range(num_agents, entitle):
            text_label[landmark] = 'LM' + " " + str(landmark)
        return text_label




