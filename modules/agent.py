import os

import torch
import torch.nn as nn
from torch.autograd import Variable

from modules.predefined_utterances_module import PredefinedUtterancesModule
from modules.processing import ProcessingModule
from modules.goal_predicting import GoalPredictingProcessingModule
from modules.action import ActionModule
from modules.goal_predicting import GoalPredictingProcessingModule
from modules.processing import ProcessingModule
from modules.word_counting import WordCountingModule
from torch.autograd import Variable
import pandas as pd
import numpy as np

"""
    The AgentModule is the general module that's responsible for the execution of
    the overall policy throughout training. It holds all information pertaining to
    the whole training episode, and at each forward pass runs a given game until
    the end, returning the total cost all agents collected over the entire game
"""
class AgentModule(nn.Module):
    def __init__(self, config):
        super(AgentModule, self).__init__()
        self.init_from_config(config)
        self.total_cost = Variable(self.Tensor(1).zero_())

        self.physical_processor = ProcessingModule(config.physical_processor)
        self.physical_pooling = nn.AdaptiveMaxPool2d((1,config.feat_vec_size))
        self.action_processor = ActionModule(config.action_processor)

        if self.using_utterances:
            self.utterance_processor = GoalPredictingProcessingModule(config.utterance_processor)
            self.utterance_pooling = nn.AdaptiveMaxPool2d((1,config.feat_vec_size))
            if self.penalizing_words:
                self.word_counter = WordCountingModule(config.word_counter)
        if self.pre_defined_utterances:
            # self.utterance_processor = PredefinedUtterancesModule() #TODO can't assign a network here.
            self.create_data_set = PredefinedUtterancesModule()

    def init_from_config(self, config):
        self.training = True
        self.using_utterances = config.use_utterances
        self.pre_defined_utterances = config.pre_defined_utterances
        self.penalizing_words = config.penalize_words
        self.using_cuda = config.use_cuda
        self.time_horizon = config.time_horizon
        self.movement_dim_size = config.movement_dim_size
        self.vocab_size = config.vocab_size
        self.goal_size = config.goal_size
        self.processing_hidden_size = config.physical_processor.hidden_size
        self.Tensor = torch.cuda.FloatTensor if self.using_cuda else torch.FloatTensor

    def reset(self):
        self.total_cost = torch.zeros_like(self.total_cost)
        if self.using_utterances and self.penalizing_words: #TODO: what should we do for pre_defined_utteranc?
            self.word_counter.word_counts = torch.zeros_like(self.word_counter.word_counts)

    def train(self, mode=True):
        super(AgentModule, self).train(mode)
        self.training = mode

    def update_mem(self, game, mem_str, new_mem, agent, other_agent=None):
        # TODO: Look into tensor copying from Variable
        new_big_mem = Variable(self.Tensor(game.memories[mem_str].data))
        if other_agent is not None:
            new_big_mem[:, agent, other_agent] = new_mem
        else:
            new_big_mem[:, agent] = new_mem
        game.memories[mem_str] = new_big_mem

    def process_utterances(self, game, agent, other_agent, utterance_processes, goal_predictions):
        utterance_processed, new_mem, goal_predicted = self.utterance_processor(game.utterances[:,other_agent], game.memories["utterance"][:, agent, other_agent])
        self.update_mem(game, "utterance", new_mem, agent, other_agent)
        utterance_processes[:, other_agent, :] = utterance_processed
        goal_predictions[:, agent, other_agent, :] = goal_predicted

    def process_physical(self, game, agent, other_entity, physical_processes):
        physical_processed, new_mem = self.physical_processor(torch.cat((game.observations[:,agent,other_entity],game.physical[:,other_entity]), 1), game.memories["physical"][:,agent, other_entity])
        self.update_mem(game, "physical", new_mem,agent, other_entity)
        physical_processes[:,other_entity,:] = physical_processed

    def get_physical_feat(self, game, agent):
        physical_processes = Variable(self.Tensor(game.batch_size, game.num_entities, self.processing_hidden_size))
        for entity in range(game.num_entities):
            self.process_physical(game, agent, entity, physical_processes)
        return self.physical_pooling(physical_processes)

    def get_utterance_feat(self, game, agent, goal_predictions):
        if self.using_utterances:
            utterance_processes = Variable(self.Tensor(game.batch_size, game.num_agents, self.processing_hidden_size))
            for other_agent in range(game.num_agents):
                self.process_utterances(game, agent, other_agent, utterance_processes, goal_predictions)
            return self.utterance_pooling(utterance_processes)
        else: #TODO: what should we do for pre_defined_utterances?
            return None

    def get_action(self, game, agent, physical_feat, utterance_feat, movements, utterances):
        movement, utterance, new_mem = self.action_processor(physical_feat, game.observed_goals[:,agent], game.memories["action"][:,agent], self.training, utterance_feat)
        self.update_mem(game, "action", new_mem, agent)
        movements[:,agent,:] = movement
        if self.using_utterances and self.pre_defined_utterances:
            utterances[:,agent,:] = utterance

    def forward(self, game):
        timesteps = []
        col_name = ['agent_color', 'agent_shape', 'lm_color', 'lm_shape', 'sentence']
        self.df_utterance = [pd.DataFrame(index=range(512), columns=col_name, dtype=np.int64) for i in range(game.num_agents)]

        for t in range(self.time_horizon):
            movements = Variable(self.Tensor(game.batch_size, game.num_entities, self.movement_dim_size).zero_())
            utterances = None
            goal_predictions = None
            if self.using_utterances:
                utterances = Variable(self.Tensor(game.batch_size, game.num_agents, self.vocab_size))
                goal_predictions = Variable(self.Tensor(game.batch_size, game.num_agents, game.num_agents, self.goal_size))

            if self.pre_defined_utterances:
                self.df_utterance = self.create_data_set.generate_sentences(game, t, self.df_utterance)

            for agent in range(game.num_agents):
                physical_feat = self.get_physical_feat(game, agent)
                utterance_feat = self.get_utterance_feat(game, agent, goal_predictions)
                self.get_action(game, agent, physical_feat, utterance_feat, movements, utterances)

            cost = game(movements, goal_predictions, utterances, t)
            if self.penalizing_words:
                cost = cost + self.word_counter(utterances)

            self.total_cost = self.total_cost + cost
            if not self.training:
                timesteps.append({
                    'locations': game.locations,
                    'movements': movements,
                    'loss': cost})
                if self.using_utterances and self.pre_defined_utterances:
                    timesteps[-1]['utterances'] = utterances
        if self.pre_defined_utterances:
            input_regex = "agent_color|agent_shape|lm_color|lm_shape"
            dataset_log = pd.DataFrame(index=range(game.batch_size), columns=range(47))
            dataset_log.loc[:, 0] = "<input>"
            dataset_log.loc[:, 1:4] = self.df_utterance[0].filter(regex=input_regex).values
            dataset_log.loc[:, 5:8] = self.df_utterance[1].filter(regex=input_regex).values
            dataset_log.loc[:, 9] = "</input>"
            dataset_log.loc[:, 10] = "<dialog>"
            for j, i in enumerate(range(0,32,2)):
                dataset_log.loc[:, 11 + i] = self.df_utterance[0].filter(regex='Full Sentence{0}'.format(j)).values
                dataset_log.loc[:, 11 + i + 1] = self.df_utterance[1].filter(regex='Full Sentence{0}'.format(j)).values
            dataset_log.loc[:, 43] = "</dialog>"
            dataset_log.loc[:, 44] = "<output>"
            dataset_log.loc[:, 45] = self.df_utterance[0].filter(regex="dist").values
            dataset_log.loc[:, 46] = self.df_utterance[1].filter(regex="dist").values
            dataset_log.loc[:, 47] = "</output>"
            # global folder_dir
            with open("dataset.csv", 'a', newline='') as f:
                dataset_log.to_csv(f, mode='a', header=False, index=False)

        return self.total_cost, timesteps
