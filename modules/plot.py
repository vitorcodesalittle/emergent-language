import os
import string
import subprocess
import torch

# import progressbar
# from time import sleep

import h5py
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from modules import data

dict_colors = {'[0.]': 'red', '[1.]': 'green', '[2.]': 'blue'}
dict_shapes = {'[0.]': 'o', '[1.]': 'v'}
epoch = -1
threshold = 0.05
ABC = list(string.ascii_uppercase)


def save_dataset(file_name, dataset_name, dataset, mode):
    with h5py.File(file_name, mode) as hf:
        epoch = len(list(hf.keys()))
        dataset_name = dataset_name + str(epoch)
        hf.create_dataset(dataset_name, data=dataset)
    return epoch


def open_dataset(file_name, epoch):
    try:
        with h5py.File(file_name, 'r') as hf:
            inner_file_name = list(hf.keys())[epoch]
            return np.array(hf[inner_file_name])
    except Exception as e:
        print(e)

class Plot:
    def __init__(self, batch_num, total_iteration, num_locations, location_dim, world_dim, num_agents, goals_by_landmark,
                 folder_dir):
        self.batch_num = batch_num
        self.total_iteration = total_iteration + 1
        self.world_dim = world_dim
        self.num_agents = num_agents
        self.location_matrix = np.zeros(shape=(self.batch_num, self.total_iteration , num_locations, location_dim)) # total_iteration + 1 - so it will include the 'start',
        self.color_matrix = np.zeros(shape=(self.batch_num, num_locations, 1))
        self.shape_matrix = np.zeros(shape=(self.batch_num, num_locations, 1))
        self.goals_by_landmark = goals_by_landmark
        self.filenames(folder_dir)

    def filenames(self, folder_dir):
        if not os.path.isabs(folder_dir):
            folder_dir = str(Path(os.getcwd())) + os.sep + folder_dir + os.sep
        self.location_file_name = folder_dir + 'locations.h5'
        self.colors_file_name = folder_dir + 'colors.h5'
        self.shape_file_name = folder_dir + 'shape.h5'
        self.players_file_name = folder_dir + 'players.h5'
        self.sentence_file_name = folder_dir + 'sentence.h5'
        self.goals_by_landmark_file_name = folder_dir + 'goals_by_landmark.h5'
        self.sentence_file_name_super = folder_dir + 'sentence_super.h5'

    def save_utterance_matrix(self, utterance, iteration, mode = None):
        if mode is None:
            if iteration == 0:
                self.utterance_matrix = np.zeros(shape=(self.batch_num, self.total_iteration, self.num_agents, 13))
            self.utterance_matrix[:,iteration+1, :, :] = utterance.detach().numpy()
            if iteration == self.total_iteration -2:
                if os.path.isfile(self.sentence_file_name):
                    self.save_h5_file('a', utterance='ON')
                else:
                    self.save_h5_file('w', utterance='ON')
        else:
            if iteration == 0:
                self.utterance_super_matrix = np.zeros(
                shape=(self.batch_num, self.total_iteration, self.num_agents, 13))
            self.utterance_super_matrix[:, iteration+1, :, :] = utterance.detach().numpy()
            if iteration == self.total_iteration-2:
                if os.path.isfile(self.sentence_file_name_super):
                    self.save_h5_file('a', utterance='ON',mode_utter='super')
                else:
                    self.save_h5_file('w', utterance='ON',mode_utter='super')

    def save_h5_file(self, mode, utterance=None, mode_utter=None):
        if utterance is None:
            save_dataset(self.location_file_name, 'location', self.location_matrix, mode)
            save_dataset(self.colors_file_name, 'colors', self.color_matrix, mode)
            save_dataset(self.shape_file_name, 'shape', self.shape_matrix, mode)
            save_dataset(self.players_file_name, 'players', self.num_agents, mode)
            save_dataset(self.goals_by_landmark_file_name, 'goals', self.goals_by_landmark, mode)
        elif utterance is not None and mode_utter is None:
            save_dataset(self.sentence_file_name, 'sentence', self.utterance_matrix, mode)
        elif utterance is not None and mode_utter is not None:
            save_dataset(self.sentence_file_name_super, 'sentence_super', self.utterance_super_matrix, mode)

    def save_plot_matrix(self, iteration, locations, colors, shapes):
        if iteration == 'start':
            self.location_matrix[:,0,:,:] = locations
            self.color_matrix[:, :, :] = colors
            self.shape_matrix[:, :, :] = shapes

        elif iteration < self.total_iteration - 2:
            self.location_matrix[:, iteration + 1, :, :] = locations.detach().numpy()
        else:
            self.location_matrix[:, iteration + 1, :, :] = locations.detach().numpy()
            if os.path.isfile(self.location_file_name):
                self.save_h5_file('a')
            else:
                self.save_h5_file('w')

    @staticmethod
    def create_video(batch_range, epoch_range, folder_dir):
        for epoch in epoch_range:
            for batch in batch_range:
                # create a video from all pictures in movies_dir of the format
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
    def create_plots(epoch, batch_size, dataset_dictionary):
        locations, colors, shapes, num_agents, utterance, goals_by_landmark = Plot.extract_data(epoch)
        text_label = Plot.creating_dot_label(locations.shape[2], num_agents)
        #creating the plots
        total_iterations = locations.shape[1]
        for batch in range(batch_size):
            marker = np.array([dict_shapes[str(shape)] for shape in np.array(shapes[batch])])
            colors_plot = np.array([dict_colors[str(color)] for color in np.array(colors[batch])])
            title = ""
            for agent in range(num_agents):
                title += "the Goal of agent {0} is that agent {1} will reach LM {2}\n"\
                    .format(goals_by_landmark[batch, agent, 1], agent, goals_by_landmark[batch, agent, 0] - num_agents)
            for iteration in range(total_iterations):
                plt.clf()
                fig, ax = plt.subplots()
                plt.axis([0, 16, 0, 16])
                locations_y = locations[batch, iteration, :, 1]
                locations_x = locations[batch, iteration, :, 0]
                utterance_legand = Plot.decoded_utterance(utterance[batch, iteration, :, :], dataset_dictionary)
                for obj in range(len(locations_y)):
                    if obj < num_agents:
                        ax.scatter(locations_x[obj], locations_y[obj], color = colors_plot[obj], marker = marker[obj],
                                        label = utterance_legand[obj])
                    else:
                        ax.scatter(locations_x[obj], locations_y[obj],
                                   color=colors_plot[obj], marker=marker[obj])
                    ax.annotate(text_label[obj], (locations_x[obj]+0.05, locations_y[obj]+0.05))
                    plt.draw()
                # Shrink current axis's height by 10% on the bottom ,  so the legand will not be over the plot
                box = ax.get_position()
                ax.set_position([box.x0, box.y0 + box.height * 0.05, box.width, box.height * 0.75])
                # Put a legend to the right of the current axis
                ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
                          fancybox=True, shadow=True, ncol=1, prop={'size': 8})
                plt.title(title)
                plt.savefig('plots' + os.sep + 'epoch_{0}batchnum_{1}iter_{2}.png'.format(epoch, batch, iteration))
                plt.close()

    @staticmethod
    def extract_data(epoch, dir=None, calculate_utternace=None,calculate_dist=None):
        #extracting the matrices containing the data from the file
        if dir is None:
            dir = os.getcwd() + os.sep
        if calculate_utternace is None and calculate_dist is None:
            return open_dataset(dir + 'locations.h5', epoch), open_dataset(dir + 'colors.h5', epoch), \
               open_dataset(dir +'shape.h5', epoch), open_dataset(dir +'players.h5', epoch), \
               open_dataset(dir +'sentence.h5', epoch), open_dataset(dir +'goals_by_landmark.h5', epoch)
        elif calculate_utternace is not None:
            return open_dataset(os.getcwd() + os.sep +'sentence.h5', epoch)
        elif calculate_dist is not None:
            return open_dataset(os.getcwd() + os.sep + 'dist_from_goal.h5', epoch)

    @staticmethod
    def creating_dot_label(entitle, num_agents):
        text_label = [None] * entitle
        text_label[:num_agents] = [str(num) for num in range(num_agents)]
        for landmark in range(num_agents, entitle):
            text_label[landmark] = 'LM' + " " + str(landmark - num_agents)
        return text_label

    @staticmethod
    def utterance_with_threshold(utterance):
        utterance = [[ABC[i] for i in range(utterance.shape[1]) if utterance[j, i] >= threshold] for j in range(utterance.shape[0])]
        return utterance

    @staticmethod
    def decoded_utterance(utterance, dataset_dictionary):
        #the data is saved only as numpay array
        utterance_encoded = []
        for i in range(len(utterance)):
            utterance_encoded += [' '.join([dataset_dictionary.word_dict.i2w(torch.LongTensor(np.array([word])))[0]
                                            for word in utterance[i]])]
        return utterance_encoded

    @staticmethod
    def create_sucees_reate_plot(sucess_rate_per_epoch, sucess_rate_per_epoch_std, epoch_range):
        epoch_num = [str(x) for x in epoch_range]
        plt.figure(figsize=(20, 3))
        sucess_rate_per_epoch_std = [round(sucess_rate_per_epoch_std[x].item(),2) for x in range(len(sucess_rate_per_epoch_std))]
        plt.bar(epoch_num, sucess_rate_per_epoch, yerr= sucess_rate_per_epoch_std , align='edge')
        plt.tight_layout()
        plt.savefig('sucess_rate.png')





