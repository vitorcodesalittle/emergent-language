import datetime
import os

import numpy as np
import matplotlib.pyplot as plt
from tempfile import TemporaryFile
import progressbar
from time import sleep
import subprocess
from pathlib import Path
import h5py


dict_colors = {'[0.]': 'red', '[1.]': 'green', '[2.]': 'blue'}
dict_shapes = {'[0.]': 'o', '[1.]': 'v'}
proj_dir = str(Path(os.getcwd()))
plots_dir = proj_dir+os.sep+'plots'+os.sep
movies_dir = proj_dir+os.sep+'movies'+os.sep
matrix_dir = proj_dir+os.sep+'matrices'+os.sep
epoch = -1


def save_dataset(file_name, datasetname, dataset, epoch, mode):
    with h5py.File(file_name, mode) as hf:
        if epoch is None:
            epoch = len(list(hf.keys()))
        datasetname = datasetname + str(epoch)
        hf.create_dataset(datasetname, data=dataset)
    return epoch


class Plot:
    def __init__(self, batch_num, total_iteration, num_locations, location_dim, world_dim, num_agents):
        self.batch_num = batch_num
        self.total_iteration = total_iteration + 1
        self.world_dim = world_dim
        self.num_agents = num_agents
        self.location_matrix = np.zeros(shape=(self.batch_num, self.total_iteration , num_locations, location_dim)) # total_iteration + 1 - so it will include the 'start',
        self.color_matrix = np.zeros(shape=(self.batch_num, num_locations, 1))
        self.shape_matrix = np.zeros(shape=(self.batch_num, num_locations, 1))

    def save_h5_file(self, mode):
        self.epoch = save_dataset('.\locations.h5', 'location', self.location_matrix, self.epoch, mode)
        save_dataset('.\clolors.h5', 'colors', self.colors_matrix, self.epoch, mode)
        save_dataset('.\shape.h5', 'shape', self.shape_matrix, self.epoch, mode)
        save_dataset('.\players.h5', 'players', self.num_agents, self.epoch, mode)

    def save_plot_matrix(self, iteration, locations, colors, shapes):
        if iteration == 'start':
            self.location_matrix[:,0,:,:] =  locations
            self.color_matrix[:,:,:] = colors
            self.shape_matrix[:,:,:] = shapes

        elif iteration < self.total_iteration - 2: # i don't need to recreate the color ans shape matrices
            self.location_matrix[:,iteration + 1,:,:] = locations.detach().numpy()
        else:
            self.location_matrix[:,iteration + 1,:,:] = locations.detach().numpy()
            if os.path.isfile('.\locations.h5'):
                # locations, colors, shapes, num_agents = Plot.extract_data_locations()
                self.save_h5_file('a')

            else:
                self.save_h5_file('w')



    @staticmethod
    def create_video(batch):
        for batch in range(batch):
            # create a video from all pictures in movies_dir of the format
            # 'batchnum_batchiter_{int}d.png'
            cmd = 'ffmpeg -f image2 -r 1/2 -i "'+movies_dir+'batchnum_{:02d}iter_%2d.png" -vcodec mpeg4 -y movie{:02d}.mp4'.format(batch, batch)
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

    def save_utterance_matrix(self,utterance, iteration):
        if iteration == 0:
            self.utterance_matrix = np.zeros(shape=(self.batch_num, self.total_iteration, self.num_agents, utterance.shape[2]))
        elif iteration < self.total_iteration - 2:
            self.utterance_matrix[:iteration + 1, :, :] = utterance.detach().numpy()
        else:
            self.utterance_matrix[:iteration + 1, :, :] = utterance.detach().numpy()
            if os.path.isfile('.\sentence.h5'):
                # locations, colors, shapes, num_agents = Plot.extract_data_locations()
                save_dataset('.\sentence.h5', 'sentence', self.utterance_matrix, self.epoch, 'a')

            else:
                save_dataset('.\sentence.h5', 'sentence', self.utterance_matrix, self.epoch, 'w')


    @staticmethod
    def create_plots(epoch, batch_size):

        #extracting the matrices containing the data from the files
        locations, colors, shapes, num_agents, utterance = Plot.extract_data(epoch)
        utterance_legand = np.zeros(shape = (locations.shape[2],utterance.shape[3])) #locations.shape[2] = num of entitels, utterance.shape[3] = vcob size

        #labels for the entitles, will be used in the plot
        text_label = Plot.creating_dot_label(locations.shape[2], num_agents)  #locations.shape[1] = num of entitels
        #opening a status bar
        bar = progressbar.ProgressBar(maxval=batch_size, \
                                      widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        #creating the plots

        total_iterations = locations.shape[1]
        for batch in range(batch_size):
            fig, ax = plt.subplots()
            plt.axis([0, 16, 0, 16])
            marker = np.array([dict_shapes[str(shape)] for shape in np.array(shapes[batch])])
            colors_plot = np.array([dict_colors[str(color)] for color in np.array(colors[batch])])
            for iteration in range(total_iterations):
                plt.clf()
                fig, ax = plt.subplots()

                locations_y = locations[batch, iteration, :, 1]
                locations_x = locations[batch, iteration, :, 0]
                utterance_legand[:num_agents] = utterance[batch, iteration, :, :]
                for obj in range(len(locations_y)):
                    sc = ax.scatter(locations_x[obj], locations_y[obj], color = colors_plot[obj], marker = marker[obj],
                                    label = utterance_legand[obj])
                    ax.annotate(text_label[obj], (locations_x[obj]+0.2, locations_y[obj]+0.2))
                    plt.draw()
                # Shrink current axis's height by 10% on the bottom ,  so the legand will not be over the plot
                box = ax.get_position()
                ax.set_position([box.x0, box.y0 + box.height * 0.25, box.width, box.height * 0.9])
                # Put a legend to the right of the current axis
                ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
                          fancybox=True, shadow=True, ncol=2 ,prop={'size': 5})

                plt.savefig(plots_dir+'epoch_{0}batchnum_{1}iter_{2}.png'.format(epoch, batch, iteration))
            bar.update(batch + 1)
            sleep(0.1)
        bar.finish()

    @staticmethod
    def extract_data (epoch):
        #extracting the matrices containing the data from the file
        with h5py.File('.\sentence.h5', 'r') as hf:
            utterance_file_name = list(hf.keys())[epoch]
            utterance = np.array(hf[utterance_file_name])
        with h5py.File('.\locations.h5', 'r') as hf:
            location_file_name = list(hf.keys())[epoch]
            locations = np.array(hf[location_file_name])
        with h5py.File('.\shape.h5', 'r') as hf:
            shape_file_name = list(hf.keys())[epoch]
            shapes = np.array(hf[shape_file_name])
        with h5py.File('.\colors.h5', 'r') as hf:
            color_file_name = list(hf.keys())[epoch]
            colors = np.array(hf[color_file_name])
        with h5py.File('.\players.h5', 'r') as hf:
            agents_file_name = list(hf.keys())[epoch]
            num_agents = np.array(hf[agents_file_name])
        return locations, colors, shapes, num_agents, utterance

    @staticmethod
    def creating_dot_label (entitle, num_agents):
        text_label = [None] * entitle
        text_label[:num_agents] = [str(num + 1) for num in range(num_agents)]
        for landmark in range(num_agents, entitle):
            text_label[landmark] = 'LM'
        return text_label




