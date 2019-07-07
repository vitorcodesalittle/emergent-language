import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim

from modules.modules_for_lm import Criterion
from modules.processing import ProcessingModule
from modules.utterance import Utterance

"""
    An ActionModule takes in the physical observation feature vector, the
    utterance observation feature vector and the individual goal of an agent
    (alongside the memory for the module), processes the goal to turn it into
    a goal feature vector, and runs the concatenation of all three feature
    vectors through a processing module. The output of the processing module
    is then fed into two independent fully connected networks to output
    utterance and movement actions
"""


class ActionModule(nn.Module):
    def __init__(self, config, utterance_config, dataset_dictionary, use_old_utterance_code):
        super(ActionModule, self).__init__()
        self.using_utterances = config.use_utterances
        self.using_cuda = config.use_cuda
        self.goal_processor = ProcessingModule(config.goal_processor)
        self.processor = ProcessingModule(config.action_processor)
        self.movement_step_size = config.movement_step_size
        self.movement_chooser = nn.Sequential(
                nn.Linear(config.action_processor.hidden_size, config.action_processor.hidden_size),
                nn.ELU(),
                nn.Linear(config.action_processor.hidden_size, config.movement_dim_size),
                nn.Tanh())
        if self.using_utterances:
            self.utter = Utterance(config, utterance_config, dataset_dictionary, use_old_utterance_code)
        #
        if not config.mode == "train_utter":
            folder_dir_fb_model = utterance_config.fb_dir
            with open(folder_dir_fb_model, 'rb') as f:
                self.utter.load_state_dict(torch.load(f))

    def processed_data(self, physical, goal, mem, utterance_feat=None):
        goal_processed, _ = self.goal_processor(goal, mem)
        if self.using_utterances:
            x = torch.cat(
                [physical.squeeze(1), utterance_feat.squeeze(1), goal_processed],
                1).squeeze(1)
        else:
            x = torch.cat([physical.squeeze(0), goal_processed], 1).squeeze(1)
        processed, mem = self.processor(x, mem)
        return processed, mem

    def forward(self, physical, goal, mem, training, use_old_utterance_code, full_sentence, total_loss, utterance_feat=None ):
        processed, mem = self.processed_data(physical, goal, mem, utterance_feat) #what is the goal in this point
        movement = self.movement_chooser(processed)
        if self.using_utterances:
            if use_old_utterance_code:
                utter = self.utter.create_utterance_using_old_code(training, processed)
            else:
                total_loss, utter, utter_super = self.utter(processed, full_sentence, total_loss)

        else:
            utter = None
        final_movement = (movement * 2 * self.movement_step_size) - self.movement_step_size
        return final_movement, utter, mem, total_loss, utter_super

