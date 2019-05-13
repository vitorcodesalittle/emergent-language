import random
import numpy as np
import pandas as pd
import torch

vocab = ['go', 'to', 'agent', 'red', 'green', 'blue', 'landmark', 'circle',
         'triangle', 'continue', 'next', 'ahead', 'done', 'good', 'stay',
         'goal']

colors_dict = ['red', 'green', 'blue']
shapes_dict = ['circle', 'triangle']


goto_sentences = [
    "<agent_color> agent go to <lm_color> landmark",
    "<agent_color> <agent_shape> agent go to <lm_color> <lm_shape> landmark",
    "<agent_shape> agent go to <lm_shape> landmark",
    "<agent_color> agent go to <lm_shape> landmark",
    "<agent_shape> agent go to <lm_color> landmark"]
continue_sentences = [
    "<agent_color> agent continue",
    "<agent_color> <agent_shape> agent continue",
    "<agent_shape> agent continue",
    "you are on the right track"]
stay_sentences = [
    "<agent_color> agent stay",
    "<agent_color> <agent_shape> agent stay",
    "<agent_shape> agent stay"]
done_sentences = [
    "<agent_color> agent is done",
    "<agent_color> <agent_shape> agent is done",
    "<agent_shape> agent is done"
    "<agent_color> good job",
    "<agent_color> <agent_shape> good job",
    "<agent_shape> good job",
    "you go girl"
]
sentence_pool = goto_sentences + continue_sentences
sentence_form = goto_sentences + continue_sentences+stay_sentences+done_sentences


class PredefinedUtterancesModule:

    @staticmethod
    def generate_single_sentence(row, sentence_ds):
        sentence = sentence_ds[row['sentence']]
        sentence = sentence.replace('<agent_color>', colors_dict[row['agent_color']])
        sentence = sentence.replace('<agent_shape>', shapes_dict[row['agent_shape']])
        sentence = sentence.replace('<lm_color>', colors_dict[row['lm_color']])
        sentence = sentence.replace('<lm_shape>', shapes_dict[row['lm_shape']])
        return sentence

    @staticmethod
    def generate_sentence(agent_color, agent_shape, lm_color, lm_shape, dist, iter):
        btz = agent_color.shape[0]
        if iter == 0 :
            sentence = [random.randint(0, len(goto_sentences) - 1) for _ in range(btz)]
            sentence_ds = goto_sentences
        elif dist > 3:
            sentence = [random.randint(0, len(sentence_pool) - 1) for _ in range(btz)]
            sentence_ds = sentence_pool
        else:
            sentence = [random.randint(0, len(done_sentences) - 1) for _ in range(btz)]
            sentence_ds = done_sentences
        data = {'agent_color': torch.Tensor.numpy(agent_color)[:,0],
                'agent_shape': torch.Tensor.numpy(agent_shape)[:,0],
                'lm_color': torch.Tensor.numpy(lm_color)[:,0],
                'lm_shape': torch.Tensor.numpy(lm_shape)[:,0],
                'sentence': np.array(sentence),
                }
        res_df = pd.DataFrame(index=range(btz), data=data, columns=data.keys(), dtype=np.int64)
        res_df['Full Sentence'] = res_df.apply(lambda row: PredefinedUtterancesModule.generate_single_sentence(row,sentence_ds), axis = 1)
        return res_df['Full Sentence'].tolist()

    def generate_sentences(self, game, iter):
        # goals = game.goals[:,:,:-1]
        # locations_lm = game.locations[:,game.num_agents:,:]
        locations_agents = game.locations[:, :game.num_agents, :]
        dist_from_goal = game.locations[:, :game.num_agents, :] - game.sorted_goals
        euclidean_distance = torch.sqrt(torch.pow(dist_from_goal, 2))
        colors = game.colors
        shapes = game.shapes
        utter = [PredefinedUtterancesModule.generate_sentence(
            colors[:, i], shapes[:, i], colors[:, game.goal_entities[:,i,:][0][0]],
            shapes[:, game.goal_entities[:,i,:][0][0]], euclidean_distance[:,i], iter)
            for i in range(game.num_agents)]

