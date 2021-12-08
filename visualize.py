import math
import cv2
import numpy as np
from collections import defaultdict
from graphviz import Digraph
import torch
from torch.autograd import Variable
from modules.game import GameModule
from configs import get_game_config, get_agent_config



def make_gif(num_agents=2, num_landmarks=3, num_colors=2, num_shapes=2, use_utterances=True, time_horizon=200, model_pt="latest.pt", outpath="viz1.avi"):
    def make_get_coords(topleft, bottomright, height, width):
        minr, minc = topleft
        maxr, maxc = bottomright
        return lambda c, r: (math.floor((c - minc)/(maxc-minc)* width), math.floor((r - minr) / (maxr - minr) * height)) # Todo
    colors = ['red', 'blue', 'green']
    shapes = ['circle', 'square']

    game_config = get_game_config(defaultdict(None, {
        'batch_size': 1,
        'world_dim': 2,
        'max_agents': 10,
        'max_landmarks': 10,
        'min_agents': 2,
        'min_landmarks': 0,
        'num_shapes': num_shapes,
        'num_colors': num_colors,
        'no_utterances': not use_utterances,
        'vocab_size': 20,
        'memory_size': 256,
        'use_cuda': False
    }))

    game = GameModule(game_config, num_agents, num_landmarks)


    def get_shape(index):
        pass

    def get_color(index):
        pass

    def print_in_frame(r, c, shape, color, frame):
        pass

    agent = torch.load(model_pt)
    agent.reset()
    agent.train(False)
    agent.time_horizon = time_horizon
    total_loss, timesteps = agent(game)
    # now we use the timesteps information for building the gif
    
    gwidth = 200
    gheigth = 200
    MEM_UPPER_BOUND = 1e9 # 1 Gbyte
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(outpath, fourcc, 22, (gwidth, gheigth), isColor=1)
    if not out.isOpened():
        raise Exception("can't open video")
    for index, info in enumerate(timesteps):
        locations = info['locations']
        locations = locations.detach().numpy()
        minr = np.min(locations[0,:,0])
        minc = np.min(locations[0,:,1])
        maxr = np.max(locations[0,:,0])
        maxc = np.max(locations[0,:,1])
        get_coords = make_get_coords((minr, minc), (maxr, maxc), gheigth, gwidth)
        frame = np.zeros((gheigth, gwidth, 3), dtype=np.uint8)
        for index, location in enumerate(locations[0]):
            y = location[0]
            x = location[1] # is it tho??
            c, r = get_coords(x, y)
            print((r,c))
            shape = get_shape(index)
            color = get_color(index)
            print_in_frame(r, c, shape, color, frame)
        out.write(frame)
    print(f"Saving video at {outpath}")
    cv2.destroyAllWindows()
    out.release()

make_gif() #debug only
    

def make_dot(var, params=None, filename=None):
    """ Produces Graphviz representation of PyTorch autograd graph
    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function
    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
    """
    if params is not None:
        assert isinstance(params.values()[0], Variable)
        param_map = {id(v): k for k, v in params.items()}

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def size_to_str(size):
        return '('+(', ').join(['%d' % v for v in size])+')'

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                name = param_map[id(u)] if params is not None else ''
                node_name = '%s\n %s' % (name, size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)
    add_nodes(var.grad_fn)
    if filename:
        dot.render(filename, view=True)
    return dot


