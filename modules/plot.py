import numpy as np
import matplotlib.pyplot as plt

dict_colors = {'[0.]': 'red', '[1.]': 'green', '[2.]': 'blue'}
dict_shapes = {'[0.]': 'o', '[1.]': 'v'}

class Plot:

    @staticmethod
    def save_step_plot(batch_mum, iteration, locations, colors, shapes):
        plt.clf()
        marker = [dict_shapes[str(shape)] for shape in np.array(shapes[1])]
        # plt.scatter(np.array(locations[1,:,0]), np.array(locations[1,:,1]) , color = [dict_colors[str(color)] for color in np.array(colors[1])]\
        #          , marker = [dict_shapes[str(shape)] for shape in np.array(shapes[1])]) ##
        # for value in dict_shapes:
        #     idx_shape = marker[marker == value]
        plt.scatter(locations[1,:,0].detach().numpy(), locations[1,:,1].detach().numpy() , color = [dict_colors[str(color)] for color in np.array(colors[1])] \
                    , marker = 'o')
        plt.axis([0, 16, 0, 16])
        plt.show()
        plt.savefig('plots\\batchnum_{0}iter_{1}.png'.format(batch_mum, iteration))
