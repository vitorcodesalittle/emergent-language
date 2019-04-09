import argparse
import numpy as np
import torch
from torch.optim import RMSprop
from torch.optim.lr_scheduler import ReduceLROnPlateau
import configs
from modules.agent import AgentModule
from modules.game import GameModule
from collections import defaultdict
from tensorboardX import SummaryWriter  # the tensorboardX is installed in the anaconda console
import time
from pathlib import Path
import os

parser = argparse.ArgumentParser(description="Trains the agents for cooperative communication task")
parser.add_argument('--no-utterances', action='store_true', help='if specified disables the communications channel (default enabled)')
parser.add_argument('--penalize-words', action='store_true', help='if specified penalizes uncommon word usage (default disabled)')
parser.add_argument('--n-epochs', '-e', type=int, help='if specified sets number of training epochs (default 5000)')
parser.add_argument('--learning-rate', type=float, help='if specified sets learning rate (default 1e-3)')
parser.add_argument('--batch-size', type=int, help='if specified sets batch size(default 256)')
parser.add_argument('--n-timesteps', '-t', type=int, help='if specified sets timestep length of each episode (default 32)')
parser.add_argument('--num-shapes', '-s', type=int, help='if specified sets number of colors (default 3)')
parser.add_argument('--num-colors', '-c', type=int, help='if specified sets number of shapes (default 3)')
parser.add_argument('--max-agents', type=int, help='if specified sets maximum number of agents in each episode (default 3)')
parser.add_argument('--min-agents', type=int, help='if specified sets minimum number of agents in each episode (default 1)')
parser.add_argument('--max-landmarks', type=int, help='if specified sets maximum number of landmarks in each episode (default 3)')
parser.add_argument('--min-landmarks', type=int, help='if specified sets minimum number of landmarks in each episode (default 1)')
parser.add_argument('--vocab-size', '-v', type=int, help='if specified sets maximum vocab size in each episode (default 6)')
parser.add_argument('--world-dim', '-w', type=int, help='if specified sets the side length of the square grid where all agents and landmarks spawn(default 16)')
parser.add_argument('--oov-prob', '-o', type=int, help='higher value penalize uncommon words less when penalizing words (default 6)')
parser.add_argument('--load-model-weights', type=str, help='if specified start with saved model weights saved at file given by this argument')
parser.add_argument('--save-model-weights', type=str, help='if specified save the model weights at file given by this argument')
parser.add_argument('--use-cuda', action='store_true', help='if specified enables training on CUDA (default disabled)')


#open a new file for each run
"""
   Create the new folder for the experiment
   :return: Full path of the new folder (str), and only the folder name
   """
def create_new_dir():
    folder_dir = str(Path(os.getcwd())) + os.sep + str(time.strftime("%H%M-%d%m%Y")) + os.sep
    folder_name_suffix = 0
    while True:
        if not os.path.exists(folder_dir):
            os.makedirs(folder_dir + os.sep) #open the folder
            break
        else:
            folder_name_suffix += 1 # add an index to the folder so we will not overwrite the previous folder
            folder_dir = folder_dir + "_" + str(folder_name_suffix)
    plots_dir = folder_dir + 'plots' + os.sep
    movies_dir = folder_dir + 'movies' + os.sep
    tensorboard_dir = folder_dir + 'tensorboard' + os.sep
    os.makedirs(plots_dir) #open the folder
    os.makedirs(movies_dir) #open the folder

    # subprocess.Popen(r'explorer ' + folder_full_path.replace('/', '\\'))
    return folder_dir


def print_losses(epoch, losses, dists, game_config, writer):
    for a in range(game_config.min_agents, game_config.max_agents + 1):
        for l in range(game_config.min_landmarks, game_config.max_landmarks + 1):
            loss = losses[a][l][-1] if len(losses[a][l]) > 0 else 0
            min_loss = min(losses[a][l]) if len(losses[a][l]) > 0 else 0
            dist = dists[a][l][-1] if len(dists[a][l]) > 0 else 0
            min_dist = min(dists[a][l]) if len(dists[a][l]) > 0 else 0
            writer.add_scalar('Loss,' + str(a) + 'agents,' + str(l) + 'landmarks' , loss, epoch)
            writer.add_scalar('dist,' + str(a) + 'agents,' + str(l) + 'landmarks' , dist, epoch)

            print("[epoch %d][%d agents, %d landmarks][%d cases][last loss: %f][min loss: %f][last dist: %f][min dist: %f]" % (epoch, a, l, len(losses[a][l]), loss, min_loss, dist, min_dist))
    print("_________________________")


def main():
    folder_dir = create_new_dir()
    writer = SummaryWriter(folder_dir + 'tensorboard' + os.sep)  #Tensorboard - setting where the temp files will be saved
    args = vars(parser.parse_args())
    agent_config = configs.get_agent_config(args)
    game_config = configs.get_game_config(args)
    training_config = configs.get_training_config(args,folder_dir)
    print("Training with config:")
    print(training_config)
    print(game_config)
    print(agent_config)
    agent = AgentModule(agent_config)
    if training_config.use_cuda:
        agent.cuda()
    optimizer = RMSprop(agent.parameters(), lr=training_config.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True, cooldown=5)
    losses = defaultdict(lambda:defaultdict(list))
    dists = defaultdict(lambda:defaultdict(list))
    for epoch in range(training_config.num_epochs):
        num_agents = np.random.randint(game_config.min_agents, game_config.max_agents+1)
        num_landmarks = np.random.randint(game_config.min_landmarks, game_config.max_landmarks+1)
        agent.reset()
        game = GameModule(game_config, num_agents, num_landmarks, folder_dir)
        if training_config.use_cuda:
            game.cuda()
        optimizer.zero_grad()

        total_loss, _ = agent(game)
        per_agent_loss = total_loss.data[0] / num_agents / game_config.batch_size
        losses[num_agents][num_landmarks].append(per_agent_loss)

        dist = game.get_avg_agent_to_goal_distance()
        avg_dist = dist.data.item() / num_agents / game_config.batch_size
        dists[num_agents][num_landmarks].append(avg_dist)

        print_losses(epoch, losses, dists, game_config, writer)

        total_loss.backward()
        optimizer.step()

        if num_agents == game_config.max_agents and num_landmarks == game_config.max_landmarks:
            scheduler.step(losses[game_config.max_agents][game_config.max_landmarks][-1])

    torch.save(agent.state_dict(), training_config.save_model_file)
    print("Saved agent model weights at %s" % training_config.save_model_file)
    writer.close() # close the tensorboard temp files

    """
    import code
    code.interact(local=locals())
    """


if __name__ == "__main__":
    main()

