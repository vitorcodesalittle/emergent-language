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
    def generate_single_sentence(row, iter, btz):
        row = row
        if iter == 0:
            sentence = random.randint(0, len(goto_sentences) - 1)
            sentence_ds = goto_sentences
        elif row['dist'] > 3:
            sentence = random.randint(0, len(sentence_pool) - 1)
            sentence_ds = sentence_pool
        else:
            sentence = random.randint(0, len(done_sentences) - 1)
            sentence_ds = done_sentences

        sentence = sentence_ds[sentence]
        sentence = sentence.replace('<agent_color>', colors_dict[int(row['agent_color'])])
        sentence = sentence.replace('<agent_shape>', shapes_dict[int(row['agent_shape'])])
        sentence = sentence.replace('<lm_color>', colors_dict[int(row['lm_color'])])
        sentence = sentence.replace('<lm_shape>', shapes_dict[int(row['lm_shape'])])
        return sentence

    @staticmethod
    def generate_sentence(agent_color, agent_shape, lm_color, lm_shape, dist, iter, df_utterance):
        btz = agent_color.shape[0]
        if iter == 0:
            data = {'agent_color': torch.Tensor.numpy(agent_color)[:, 0],
                    'agent_shape': torch.Tensor.numpy(agent_shape)[:, 0],
                    'lm_color': torch.Tensor.numpy(lm_color)[:, 0],
                    'lm_shape': torch.Tensor.numpy(lm_shape)[:, 0],
                    }
            df_utterance = pd.DataFrame(data=data, dtype=np.int64)
        df_utterance['dist'] = dist.detach().numpy()
        df_utterance['Full Sentence' + str(iter)] = df_utterance.apply(
            lambda row: PredefinedUtterancesModule.generate_single_sentence(row, iter, btz), axis=1, reduce=False )
        return df_utterance

    def generate_sentences(self, game, iter, list_df_utterance):
        dist_from_goal = game.locations[:, :game.num_agents, :] - game.sorted_goals
        euclidean_distance = torch.sqrt(torch.sum(torch.pow(dist_from_goal, 2), dim=1))
        colors = game.colors
        shapes = game.shapes
        list_df_utterance = [PredefinedUtterancesModule.generate_sentence(
            colors[:, i], shapes[:, i], colors[:, game.goal_entities[:,i,:][0][0]],
            shapes[:, game.goal_entities[:,i,:][0][0]], euclidean_distance[:,i], iter, list_df_utterance[i])
            for i in range(game.num_agents)]
        return list_df_utterance

