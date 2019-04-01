import os

import numpy as np
import matplotlib.pyplot as plt
from tempfile import TemporaryFile
import progressbar
from time import sleep
import subprocess
from pathlib import Path


dict_colors = {'[0.]': 'red', '[1.]': 'green', '[2.]': 'blue'}
dict_shapes = {'[0.]': 'o', '[1.]': 'v'}
proj_dir = str(Path(os.getcwd()).parent)
plots_dir = proj_dir+os.sep+'plots'+os.sep
movies_dir = proj_dir+os.sep+'movies'+os.sep
matrix_dir = proj_dir+os.sep+'matrix'+os.sep
out_matrix_file = matrix_dir + 'outmatrixfile.npz'
utterance_matrix_file = matrix_dir+'utterancematrixfile.npz'
dict_epoch = {'epoch' : 0}


class Plot:
    def __init__(self, batch_num, total_iteration, num_locations, location_dim, world_dim, num_agents,num_epochs):
        dict_epoch['epoch'] += 1 ####
        self.epoch = dict_epoch['epoch']-1 ####
        self.batch_num = batch_num
        self.num_epochs = num_epochs
        self.total_iteration = total_iteration + 1
        self.world_dim = world_dim
        self.num_agents = num_agents

        self.location_matrix = np.zeros(shape= (self.num_epochs, self.batch_num, self.total_iteration , num_locations, location_dim)) # total_iteration + 1 - so it will include the 'start'
        self.color_matrix = np.zeros(shape = (self.num_epochs, self.batch_num, num_locations, 1))
        self.shape_matrix = np.zeros(shape = (self.num_epochs, self.batch_num, num_locations, 1))

    def save_plot_matrix(self, iteration, locations, colors, shapes):
        if iteration == 'start':
            self.location_matrix[self.epoch,:,0,:,:] =  locations
            self.color_matrix[self.epoch,:,:,:] = colors
            self.shape_matrix[self.epoch,:,:,:] = shapes

        elif iteration < self.total_iteration - 2: # i don't need to recreate the color ans shape matrix
            self.location_matrix[self.epoch,:,iteration + 1,:,:] =  locations.detach().numpy()
        else:
            self.location_matrix[self.epoch,:,iteration + 1,:,:] =  locations.detach().numpy()
            outmatrixfile = TemporaryFile()
            np.savez(out_matrix_file, self.location_matrix, self.color_matrix, self.shape_matrix, np.array([self.num_agents]))

    @staticmethod
    def create_video():
        matrix = np.load(out_matrix_file)
        location_array, color_array, shape_array = matrix.files
        locations = matrix[location_array]
        batch_num = locations.shape[0]
        total_iterations = locations.shape[1]
        for batch in range(batch_num):
            cmd = 'ffmpeg -f image2 -r 1/2 -i "'+movies_dir+'batchnum_{:02d}iter_%2d.png" -vcodec mpeg4 -y movie{:02d}.mp4'.format(batch, batch)
            cmd = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   shell=True,
                                   cwd=os.getcwd())
            out, err = cmd.communicate()
            if cmd.returncode == 0:
                print("Job done.")
            else:
                print("ERROR")
                print(out)

    def save_utterance_matrix(self,utterance, iteration):
        if iteration == 0 :
            self.utterance_matrix = np.zeros(shape = (self.num_epochs ,self.batch_num, self.total_iteration, self.num_agents, utterance.shape[2]))
        elif iteration < self.total_iteration - 2:
            self.utterance_matrix[self.epoch,:,iteration + 1 ,:,:] = utterance.detach().numpy()
        else:
            self.utterance_matrix[self.epoch,:,iteration + 1 ,:,:] = utterance.detach().numpy()
            utterancematrixfile = TemporaryFile()
            np.savez(utterance_matrix_file, self.utterance_matrix)

    @staticmethod
    def create_plots(epoch, batch_size):

        #extracting the matrix containing the data from the file
        locations, colors, shapes, num_agents = Plot.extract_data_locations()
        utterance = Plot.extract_utterance_matrix()
        utterance_legand = np.zeros(shape = (locations.shape[2],utterance.shape[3]))

        #labels for the agents, will be used in the plot
        text_label = Plot.creating_dot_label(locations.shape[2], num_agents)
        #opening a status bar
        bar = progressbar.ProgressBar(maxval=batch_size, \
                                      widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        #creating the plots

        total_iterations = locations.shape[1]
        for batch in range(batch_size):
            fig, ax = plt.subplots()
            plt.axis([0, 16, 0, 16])
            marker = np.array([dict_shapes[str(shape)] for shape in np.array(shapes[epoch, batch])])
            colors_plot = np.array([dict_colors[str(color)] for color in np.array(colors[epoch, batch])])
            for iteration in range(total_iterations):
                plt.clf()
                fig, ax = plt.subplots()

                locations_y = locations[epoch, batch, iteration, :, 1]
                locations_x = locations[epoch, batch, iteration, :, 0]
                utterance_legand[:num_agents] = utterance[epoch, batch, iteration, :, :]
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

                plt.savefig(plots_dir+'batchnum_{0}iter_{1}epoch_{2}.png'.format(batch, iteration,epoch))
            bar.update(batch + 1)
            sleep (0.1)
        bar.finish()

    @staticmethod
    def extract_data_locations ():
        #extracting the matrix containing the data from the file
        matrix = np.load(out_matrix_file)
        location_array, color_array, shape_array, num_agents = matrix.files
        locations = matrix[location_array]
        colors = matrix[color_array]
        shapes = matrix[shape_array]
        num_agents = matrix[num_agents][0]
        return locations, colors, shapes, num_agents

    @staticmethod
    def creating_dot_label (entitle, num_agents):
        text_label = [None] * entitle
        text_label[:num_agents] = [str(num + 1) for num in range(num_agents)]
        for landmark in range(num_agents, entitle):
            text_label[landmark] = 'LM'
        return text_label

    @staticmethod
    def extract_utterance_matrix():
        #extracting the matrix containing the data from the file
        matrix = np.load(utterance_matrix_file)
        utterance_array = matrix.files[0]
        utterance = matrix[utterance_array]

        return utterance
