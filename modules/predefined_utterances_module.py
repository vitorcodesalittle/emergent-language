import random
import numpy as np
import pandas as pd
import torch
import re

colors_dict = ['red', 'green', 'blue']
shapes_dict = ['circle', 'triangle']
start_token = 'Hi'
end_token = '<eos>'


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
token_regex = '\w*<(\w*)>\w*'
tokens = set([re.findall(token_regex,sentence)[i]
          for sentence in sentence_form for i in range(len(re.findall(token_regex,sentence)))])

class PredefinedUtterancesModule:

    @staticmethod
    def generate_single_sentence(row, iter):
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

        sentence = start_token + ' ' + sentence_ds[sentence]
        for token in tokens:
            sentence = sentence.replace('<' + token + '>', colors_dict[int(row[token])])
        sentence += ' ' + end_token
        return sentence

    def generate_sentence(self,agent_color, agent_shape, lm_color, lm_shape, dist, iter, df_utterance, mode):
        if iter == 0 or mode is not None:
            data = {'agent_color': torch.Tensor.numpy(agent_color)[:, 0],
                    'agent_shape': torch.Tensor.numpy(agent_shape)[:, 0],
                    'lm_color': torch.Tensor.numpy(lm_color)[:, 0],
                    'lm_shape': torch.Tensor.numpy(lm_shape)[:, 0],
                    }
            df_utterance = pd.DataFrame(data=data, dtype=np.int64)
        df_utterance['dist'] = np.around(dist.detach().numpy(),2)
        df_utterance['Full Sentence' + str(iter)] = df_utterance.apply(
            lambda row: PredefinedUtterancesModule.generate_single_sentence(row, iter), axis=1, reduce=False )
        return df_utterance

    def generate_sentences(self, game, iter, list_df_utterance, mode = None):
        if mode is None:
            dist_from_goal = game.locations[:, :game.num_agents, :] - game.sorted_goals
        else:
            # TODO: use configs and not hard coded dims
            rand_agent_locations = torch.FloatTensor(np.random.uniform(low=0, high=16, size=(32,2,2)))
            dist_from_goal = rand_agent_locations - game.sorted_goals
        euclidean_distance = torch.sqrt(torch.sum(torch.pow(dist_from_goal, 2), dim=1))
        colors = game.colors
        shapes = game.shapes
        list_df_utterance = [self.generate_sentence(
            colors[:, i], shapes[:, i], colors[:, game.goal_entities[:,i,:][0][0]],
            shapes[:, game.goal_entities[:,i,:][0][0]], euclidean_distance[:,i], iter, list_df_utterance[i], mode)
            for i in range(game.num_agents)]
        return list_df_utterance

    def generate_dataset_txt_file(self,btz,df_utterance, df_utterance_col_name):
        input_regex = ""
        for index,col_name in enumerate(df_utterance_col_name):
            input_regex += col_name
            if index < len(df_utterance_col_name)-1:
                input_regex += '|'
            else:
                pass
        len_dataset_log = 2 * len(df_utterance[0].columns) + len(df_utterance_col_name)
        dataset_log = pd.DataFrame(index=range(btz), columns=range(len_dataset_log))
        col_index = 0
        dataset_log.loc[:, col_index] = "<input>"
        col_index += 1
        for agent in range(len(df_utterance)):
            dataset_log.loc[:, col_index:col_index+3] = df_utterance[agent].filter(regex=input_regex).values
            col_index += 4
        col_index += 1
        dataset_log.loc[:, col_index] = "</input>"
        dataset_log.loc[:, col_index+1] = "<dialog>"
        col_index += 1
        len_sentence_iter = 2 * len(df_utterance[0].columns) - 2*len(df_utterance_col_name)
        for j, i in enumerate(range(0, len_sentence_iter, 2)):
            dataset_log.loc[:, col_index + i] = df_utterance[0].filter(regex='Full Sentence{0}'.format(j)).values
            dataset_log.loc[:, col_index + i + 1] = df_utterance[1].filter(regex='Full Sentence{0}'.format(j)).values
        col_index += len_sentence_iter
        dataset_log.loc[:, col_index] = "</dialog>"
        dataset_log.loc[:, col_index+1] = "<output>"
        dataset_log.loc[:, col_index+2] = df_utterance[0].filter(regex="dist").values
        dataset_log.loc[:, col_index+3] = df_utterance[1].filter(regex="dist").values
        dataset_log.loc[:, col_index+4] = "</output>"
        # global folder_dir
        with open("dataset.csv", 'a', newline='') as f:
            dataset_log.to_csv(f, mode='a', header=False, index=False)
