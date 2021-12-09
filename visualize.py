import math
import string
import cv2
import numpy as np
from collections import defaultdict
from graphviz import Digraph
import torch
from torch.autograd import Variable
from modules.game import GameModule
from configs import get_game_config, get_agent_config


def make_gif(num_agents=2, num_landmarks=3, num_colors=4, num_shapes=2, use_utterances=True, time_horizon=100, model_pt="latest.pt", outpath="viz1.avi", gwidth=400, gheigth=400, fps=5):
    def make_get_coords(topleft, bottomright, height, width):
        minr, minc = topleft
        maxr, maxc = bottomright
        return lambda c, r: (math.floor((c - minc)/(maxc-minc+1)* width), math.floor((r - minr) / (maxr - minr + 1) * height)) # Todo
    colors = [(66, 135, 245), (214, 19, 201), (135, 168, 230), (242, 144, 7), (242, 144, 7), (255,0,0), (0,255,0), (0,0,255), (255, 255, 0), (0, 255, 255), (255, 0, 255), (133, 133, 133), (12, 133, 256)]
    alphabet = string.ascii_lowercase[:20]
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
        return shapes[int(game.physical[0, index, 1])]

    def get_color(index):
        return colors[int(game.physical[0, index, 0])]
    def print_agent_labels(frame, shapes, colors):
        label = "\n".join(map(str, list(zip(range(num_agents), shapes))))
        frame = putText(frame, label, (50, 75), fontScale=0.5)
        for color in colors:
            pass
    
    shapes = [ get_shape(index) for index in range(0, num_agents + num_landmarks) ]
    colors = [ get_color(index) for index in range(0, num_agents + num_landmarks) ]

    def putText(frame, text, pos, fontScale = 1):
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = pos
        color = (255, 255, 255)
        thickness = 2
        cv2.putText(frame, text, org, font, fontScale, color, thickness, cv2.LINE_AA)

    def print_in_frame(r, c, shape, color, frame):
        size = 10
        if shape == "square":
            frame[max(0, c-size//2):min(gwidth, c+size//2), max(0, r-size//2):min(gheigth, r+size//2)] = color
        elif shape == "circle":
            frame = cv2.circle(frame, (c,r), size//2, color=color, thickness=-1)

    def create_utterances_label(utterances):
        cs = np.argmax(utterances, axis=2).detach().numpy().tolist()
        return ",".join(list(map(lambda c: alphabet[c], cs[0])))

           
    def print_utterances_label(frame, utterances):
        label = create_utterances_label(utterances)
        frame = putText(frame, label, (50, 50))

    agent = torch.load(model_pt)
    agent.reset()
    agent.train(False)
    agent.time_horizon = time_horizon
    total_loss, timesteps = agent(game)
    # now we use the timesteps information for building the gif
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(outpath, fourcc, fps, (gwidth, gheigth), isColor=1)
    if not out.isOpened():
        raise Exception("can't open video")
    locations = np.concatenate(list(map(lambda t: t['locations'].detach().numpy(), timesteps)), axis=0)
    minr = np.min(locations[:,:,0]) - 2
    minc = np.min(locations[:,:,1]) - 2
    maxr = np.max(locations[:,:,0]) + 2
    maxc = np.max(locations[:,:,1]) + 2
    for index, info in enumerate(timesteps):
        locations = info['locations']
        utterances = info['utterances']
        locations = locations.detach().numpy()
        get_coords = make_get_coords((minr, minc), (maxr, maxc), gheigth, gwidth)
        frame = np.zeros((gheigth, gwidth, 3), dtype=np.uint8)
        print_agent_labels(frame, shapes, colors)
        for index, location in enumerate(locations[0]):
            y = location[0]
            x = location[1] # is it tho??
            c, r = get_coords(x, y)
            shape = get_shape(index)
            color = get_color(index)
            print_in_frame(r, c, shape, color, frame)
        print_utterances_label(frame, utterances)
        out.write(frame)
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


