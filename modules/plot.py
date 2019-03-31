import numpy as np
import matplotlib.pyplot as plt
from tempfile import TemporaryFile
import progressbar
from time import sleep
import  time

dict_colors = {'[0.]': 'red', '[1.]': 'green', '[2.]': 'blue'}
dict_shapes = {'[0.]': 'o', '[1.]': 'v'}

class Plot:
    def __init__(self, batch_num, total_iteration, num_locations, location_dim, world_dim):
        self.location_matrix = np.zeros(shape= (batch_num, total_iteration + 1 , num_locations, location_dim)) # total_iteration + 1 - so it will include the 'start'
        self.color_matrix = np.zeros(shape = (batch_num, num_locations, 1))
        self.shape_matrix = np.zeros(shape = (batch_num, num_locations, 1))
        self.world_dim = world_dim

    def save_plot_matrix(self, iteration, locations, colors, shapes):
        if iteration == 'start':
              self.location_matrix[:,0,:,:] =  locations
              self.color_matrix = colors
              self.shape_matrix = shapes

        else: # i don't need to recreate the color ans shape matrix
            self.location_matrix[:,iteration + 1,:,:] =  locations.detach().numpy()
            outmatrixfile = TemporaryFile()
            np.savez('outmatrixfile.npz', self.location_matrix, self.color_matrix, self.shape_matrix)


    @staticmethod
    def create_plots():
        # plt.ion() #so the plot will be dynamic
        fig, ax = plt.subplots()
        plt.axis([0, 16, 0, 16])
        matrix = np.load(r'C:\Users\user\Desktop\emergent-language\outmatrixfile.npz')
        location_array, color_array, shape_array = matrix.files
        locations = matrix[location_array]
        colors = matrix[color_array]
        shapes = matrix[shape_array]
        batch_num = locations.shape[0]
        total_iterations = locations.shape[1]

        # creating status bar
        bar = progressbar.ProgressBar(maxval=batch_num, \
                                      widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()



        for batch in range(batch_num):
            plt.cla()
            plt.axis([0, 16, 0, 16])

            for iteration in  range(total_iterations):
                marker = np.array([dict_shapes[str(shape)] for shape in np.array(shapes[batch])])
                for value in dict_shapes.values():
                    idx_shape = marker == value
                    locations_x = locations[batch,iteration,:,0][idx_shape]
                    locations_y = locations[batch,iteration,:,1][idx_shape]
                    colors_plot = np.array([dict_colors[str(color)] for color in np.array(colors[batch])])[idx_shape]
                    if iteration == 0:
                        sc = ax.scatter(locations_x, locations_y, color = colors_plot, marker = value)
                        plt.draw()

                else:
                        sc.set_offsets(np.c_[locations_x, locations_y])
                        fig.canvas.draw_idle()

                # plt.show()
                plt.savefig(r'C:\Users\user\Desktop\emergent-language\plots\batchnum_{0}iter_{1}.png'.format(batch, iteration))
            bar.update(batch + 1)
            sleep (0.1)
        bar.finish()















