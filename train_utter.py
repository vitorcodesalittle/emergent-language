import os
import random

import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable

import configs
from modules import data, action
from modules.action import ActionModule
from modules.agent import AgentModule
from modules.game import GameModule
from modules.predefined_utterances_module import PredefinedUtterancesModule
from modules.utterance import Utterance
from train import parser


def main():
    args = vars(parser.parse_args())
    run_default_config = configs.get_run_config(args)
    folder_dir = run_default_config.folder_dir
    agent_config = configs.get_agent_config(args)
    game_config = configs.get_game_config(args)
    training_config = configs.get_training_config(args, folder_dir)
    corpus = data.WordCorpus('data' + os.sep, freq_cutoff=20, verbose=True)
    agent = AgentModule(agent_config, corpus, run_default_config.creating_data_set_mode,
                        run_default_config.create_utterance_using_old_code)
    utter = Utterance(agent_config.action_processor, corpus, run_default_config.create_utterance_using_old_code)
    action = ActionModule(agent_config.action_processor, corpus, run_default_config.create_utterance_using_old_code)
    create_data_set = PredefinedUtterancesModule()
    for epoch in range(training_config.num_epochs):
        num_agents = np.random.randint(game_config.min_agents,
                                       game_config.max_agents + 1)
        num_landmarks = np.random.randint(game_config.min_landmarks,
                                          game_config.max_landmarks + 1)
        agent.reset()
        game = GameModule(game_config, num_agents, num_landmarks, folder_dir)
        df_utterance = [pd.DataFrame(index=range(game.batch_size), columns=agent.df_utterance_col_name
                                          , dtype=np.int64) for i in range(game.num_agents)]
        iter = random.randint(0, game.time_horizon)
        df_utterance = create_data_set.generate_sentences(game, iter, df_utterance, mode="train utter")

        for agent_num in range(game.num_agents):
            physical_feat = agent.get_physical_feat(game, agent_num)
            mem = Variable(torch.zeros(game.batch_size, game.num_agents,game_config.memory_size)[:, agent_num])
            utterance_feat = torch.zeros([game.batch_size, 1, 256], dtype=torch.float)
            goal = game.observed_goals[:, agent_num]
            processed, mem = action.processed_data(physical_feat, goal, mem,
                                                   utterance_feat)
            full_sentence = df_utterance[agent_num]['Full Sentence' + str(iter)]
            loss, utterance = utter(processed, full_sentence)
            print(loss)


if __name__ == "__main__":
    main()
