import os
import numpy as np
import torch
from torch.autograd import Variable

import configs
from modules import data, action
from modules.action import ActionModule
from modules.agent import AgentModule
from modules.game import GameModule
from modules.utterance import Utterance
from train import create_new_dir, parser


def main():
    global folder_dir
    folder_dir = create_new_dir()
    args = vars(parser.parse_args())
    agent_config = configs.get_agent_config(args)
    game_config = configs.get_game_config(args)
    training_config = configs.get_training_config(args, folder_dir)
    corpus = data.WordCorpus('data' + os.sep, freq_cutoff=20, verbose=True)
    agent = AgentModule(agent_config, corpus)
    utter = Utterance(agent_config.action_processor, corpus)
    action = ActionModule(agent_config.action_processor, corpus)

    for epoch in range(training_config.num_epochs):
        num_agents = np.random.randint(game_config.min_agents,
                                       game_config.max_agents + 1)
        num_landmarks = np.random.randint(game_config.min_landmarks,
                                          game_config.max_landmarks + 1)
        agent.reset()
        game = GameModule(game_config, num_agents, num_landmarks, folder_dir)
        # goal_predictions = Variable(
        #     agent.Tensor(game.batch_size, game.num_agents, game.num_agents,
        #                  agent.goal_size))
        for agent_num in range(game.num_agents):
            physical_feat = agent.get_physical_feat(game, agent_num)
            # goal = game.observed_goals[:, agent_num]
            mem = Variable(torch.zeros(game.batch_size, game.num_agents,
                                       game_config.memory_size)[:, agent_num])
            # utterance_feat = agent.get_utterance_feat(game, agent_num, goal_predictions)
            utterance_feat = torch.zeros([game.batch_size, 1, 256], dtype=torch.float)
            # movements = Variable(agent.Tensor(game.batch_size,
            #                                   game.num_entities,
            #                                   agent.movement_dim_size).zero_())
            goal = game.observed_goals[:, agent_num]
            processed, mem = action.processed_data(physical_feat, goal, mem,
                                                   utterance_feat)
            loss, utterance = utter(processed)
            print(loss)


if __name__ == "__main__":
    main()
