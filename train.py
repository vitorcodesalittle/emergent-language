import argparse
import os
from collections import defaultdict

import numpy as np
import torch
from modules import plot
from modules.agent import AgentModule
from modules.game import GameModule
from tensorboardX import SummaryWriter  # the tensorboardX is installed in the anaconda console
from torch.optim import RMSprop
from torch.optim.lr_scheduler import ReduceLROnPlateau

import configs
import create_plots  # for the file dir function

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
parser.add_argument('--min-landmarks', type=int, help='if specified sets minimum number of landmarks in each episode (default save1)')
parser.add_argument('--vocab-size', '-v', type=int, help='if specified sets maximum vocab size in each episode (default 6)')
parser.add_argument('--world-dim', '-w', type=int, help='if specified sets the side length of the square grid where all agents and landmarks spawn(default 16)')
parser.add_argument('--oov-prob', '-o', type=int, help='higher value penalize uncommon words less when penalizing words (default 6)')
parser.add_argument('--load-model-weights', type=str, help='if specified start with saved model weights saved at file given by this argument')
parser.add_argument('--save-model-weights', type=str, help='if specified save the model weights at file given by this argument')
parser.add_argument('--use-cuda', action='store_true', default=False, help='if specified enables training on CUDA (default disabled)')
parser.add_argument('--upload-trained-model', help='if specified the trained model weights will be uploaded and the network will continue the run with then')
parser.add_argument('--dir-upload-model', required=False, type=str, help='Directory to folder containing the trained model')
parser.add_argument('--save-to-a-new-dir', required=False, type=bool, help='define if we want to save the info in a new dir or are we in debag mode and all of the data weill be moved to debag folder')
parser.add_argument('--creating-data-set-mode', required=False, type=bool, help='define if we are in create dataset mode or not')
parser.add_argument('--create-utterance-using-old-code', type=bool, help='use when we want to create dataset, or create the trained model that the dataset code willuse ')
parser.add_argument('--one-sentence-data-set',action='store_true', default=True, help='temp, train the mini FC network on one setuation')
parser.add_argument('--fb-dir', required=False, type=str, help='if specified FB will be fine tuned ussing the reward loss, the fb model weight will be taken from the specifed dir')

def print_losses(epoch, losses, dists, game_config, writer):
    for a in range(game_config.min_agents, game_config.max_agents + 1):
        for l in range(game_config.min_landmarks, game_config.max_landmarks + 1):
            loss = losses[a][l][-1] if len(losses[a][l]) > 0 else 0
            min_loss = min(losses[a][l]) if len(losses[a][l]) > 0 else 0
            dist = dists[a][l][-1] if len(dists[a][l]) > 0 else 0
            min_dist = min(dists[a][l]) if len(dists[a][l]) > 0 else 0
            writer.add_scalar('Loss,' + str(a) + 'agents,' + str(l) + 'landmarks' , loss, epoch) #data for Tensorboard
            writer.add_scalar('dist,' + str(a) + 'agents,' + str(l) + 'landmarks' , dist, epoch) #data for TensorBoard

            print("[epoch %d][%d agents, %d landmarks][%d cases][last loss: %f][min loss: %f][last dist: %f][min dist: %f]" % (epoch, a, l, len(losses[a][l]), loss, min_loss, dist, min_dist))
    print("_________________________")


def main():
    args = vars(parser.parse_args())
    run_config = configs.get_run_config(args)
    agent_config = configs.get_agent_config(args)
    game_config = configs.get_game_config(args)
    training_config = configs.get_training_config(args,run_config.folder_dir)
    utterance_config = configs.get_utterance_config()
    print("Training with config:")
    print(training_config)
    print(game_config)
    print(agent_config)
    print(run_config)
    writer = SummaryWriter(run_config.folder_dir + 'tensorboard' + os.sep)  #Tensorboard - setting where the temp files will be saved
    agent = AgentModule(agent_config, utterance_config, run_config.corpus, run_config.creating_data_set_mode, run_config.create_utterance_using_old_code)
    if run_config.upload_trained_model:
        folder_dir_trained_model = run_config.dir_upload_model
        agent.load_state_dict(torch.load(folder_dir_trained_model))
        agent.eval()
    else:
        pass
    if training_config.use_cuda:
        agent.cuda()
    optimizer = RMSprop(agent.parameters(), lr=training_config.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True, cooldown=5)
    losses = defaultdict(lambda: defaultdict(list))
    dists = defaultdict(lambda: defaultdict(list))
    for epoch in range(training_config.num_epochs):
        num_agents = np.random.randint(game_config.min_agents, game_config.max_agents+1)
        num_landmarks = np.random.randint(game_config.min_landmarks, game_config.max_landmarks+1)
        agent.reset()
        game = GameModule(game_config, num_agents, num_landmarks, run_config.folder_dir)
        if training_config.use_cuda:
            game.cuda()
        optimizer.zero_grad()

        total_loss, _ = agent(game)
        per_agent_loss = total_loss.data[0] / num_agents / game_config.batch_size
        losses[num_agents][num_landmarks].append(per_agent_loss)

        dist, dist_per_agent = game.get_avg_agent_to_goal_distance() #add to tensorboard
        dist_per_agent_file_name = run_config.folder_dir + 'dist_from_goal.h5'
        if os.path.isfile(dist_per_agent_file_name):
            plot.save_dataset(dist_per_agent_file_name, 'dist_from_goal', dist_per_agent.detach().numpy(), 'a')
        else:
            plot.save_dataset(dist_per_agent_file_name, 'dist_from_goal', dist_per_agent.detach().numpy(), 'w')

        avg_dist = dist.data.item() / num_agents / game_config.batch_size
        dists[num_agents][num_landmarks].append(avg_dist)

        print_losses(epoch, losses, dists, game_config, writer)
        torch.autograd.set_detect_anomaly(True)
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

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

