import random

import pandas as pd
import torch

vocab = ['go', 'to', 'agent', 'red', 'green', 'blue', 'landmark', 'circle',
         'triangle', 'continue', 'next', 'ahead', 'done', 'good', 'stay',
         'goal']

colors_dict = ['red', 'green', 'blue']
shapes_dict = ['circle', 'triangle']


goto_sentences = [ #16 - 10
    "<agent_color> agent go to <lm_color> landmark",
    "<agent_color> <agent_shape> agent go to <lm_color> <lm_shape> landmark",
    "<agent_shape> agent go to <lm_shape> landmark",
    "<agent_color> agent go to <lm_shape> landmark",
    "<agent_shape> agent go to <lm_color> landmark"]
continue_sentences = [ # 5-3
    "<agent_color> agent continue",
    "<agent_color> <agent_shape> agent continue",
    "<agent_shape> agent continue",
    "you are on the right track"]
stay_sentences = [
    "<agent_color> agent stay",
    "<agent_color> <agent_shape> agent stay",
    "<agent_shape> agent stay"]
done_sentences = [ # 0-3
    "<agent_color> agent is done",
    "<agent_color> <agent_shape> agent is done",
    "<agent_shape> agent is done"
    "<agent_color> good job",
    "<agent_color> <agent_shape> good job",
    "<agent_shape> good job",
    "you go girl"
]

sentence_form = goto_sentences + continue_sentences+stay_sentences+done_sentences


class PredefinedUtterancesModule:

    @staticmethod
    def generate_single_sentence(agent_color, agent_shape, lm_color, lm_shape, dist):
        # TODO: according to distance choose randomly from X_sentences
        sentence = random.choice(sentence_form)
        sentence.replace('<agent_color>', agent_color)
        sentence.replace('<agent_shape>', agent_shape)
        sentence.replace('<lm_color>', lm_color)
        sentence.replace('<lm_shape>', lm_shape)
        return sentence

    @staticmethod
    def generate_sentence(agent_color, agent_shape, lm_color, lm_shape, dist):
        data = {'agent_color': torch.Tensor.numpy(agent_color)[:,0],
                'agent_shape': torch.Tensor.numpy(agent_shape)[:,0],
                'lm_color': torch.Tensor.numpy(lm_color)[:,0],
                'lm_shape': torch.Tensor.numpy(lm_shape)[:,0]
                }
        res_df = pd.DataFrame(index=range(agent_color.shape[0]), data=data)

        # TODO: use applay here
        # function generate_single_sentence on row

    def generate_sentences(self, game):
        # goals = game.goals[:,:,:-1]
        # locations_lm = game.locations[:,game.num_agents:,:]
        locations_agents = game.locations[:, :game.num_agents, :]
        dist_from_goal = game.locations[:, :game.num_agents, :] - game.sorted_goals
        euclidean_distance = torch.sqrt(torch.pow(dist_from_goal, 2))
        colors = game.colors
        shapes = game.shapes
        utter = [PredefinedUtterancesModule.generate_sentence(
            colors[:, i], shapes[:, i], colors[:, game.goal_entities[:,i,:][0][0]], shapes[:, game.goal_entities[:,i,:][0][0]], euclidean_distance[:,i])
            for i in range(game.num_agents)]

