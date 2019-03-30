import numpy as np
import matplotlib.pyplot as plt

dict_colors = {'[0.]': 'red', '[1.]': 'green', '[2.]': 'blue'}
dict_shapes = {'[0.]': 'o', '[1.]': 'v'}

class Plot:

    @staticmethod
    def save_step_plot(batch_num, iteration, locations, colors, shapes,world_dim):
        for batch in range(batch_num):
            plt.clf()
            marker = np.array([dict_shapes[str(shape)] for shape in np.array(shapes[batch])])
            for value in dict_shapes.values():
                idx_shape = marker == value
                locations_x = locations[batch,:,0].detach().numpy()[idx_shape]
                locations_y = locations[batch,:,1].detach().numpy()[idx_shape]
                colors_plot = np.array([dict_colors[str(color)] for color in np.array(colors[batch])])[idx_shape]
                plt.scatter(locations_x, locations_y, color = colors_plot, marker = value)
                plt.axis([0, world_dim, 0, world_dim])
            # plt.show()
            plt.savefig('plots\\batchnum_{0}iter_{1}.png'.format(batch, iteration))


