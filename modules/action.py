import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim

from modules.dialog_model import DialogModel
from modules.gumbel_softmax import GumbelSoftmax
from modules.modules_for_lm import Criterion
from modules.processing import ProcessingModule


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
    def __init__(self, config, dataset_dictionary):
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
            self.utterance_chooser = nn.Sequential(
                    nn.Linear(config.action_processor.hidden_size, config.hidden_size),
                    nn.ELU(),
                    nn.Linear(config.hidden_size, config.vocab_size))
            self.gumbel_softmax = GumbelSoftmax(config.use_cuda)

        self.args = {'init_range': 0.1, 'nhid_lang': 128, 'nembed_word': 256,
                     'nhid_ctx': 256, 'dropout': 0.5, 'momentum':0.1,
                     'lr':1, 'nesterov':True, 'clip':0.5}
        self.dataset_dictionary = dataset_dictionary
        self.lm_model = DialogModel(dataset_dictionary.word_dict, None,
                                    None, 4,
                                    self.args,
                                    None)
        self.crit = Criterion(dataset_dictionary.word_dict, device_id=None)
        self.opt = optim.SGD(self.lm_model.parameters(), lr=self.args['lr'],
                             momentum=self.args['momentum'],
                             nesterov=(self.args['nesterov'] and self.args['momentum'] > 0))

    def forward(self, physical, goal, mem, training, utterance=None):
        goal_processed, _ = self.goal_processor(goal, mem)
        if self.using_utterances:
            x = torch.cat([physical.squeeze(1), utterance.squeeze(1), goal_processed], 1).squeeze(1)
        else:
            x = torch.cat([physical.squeeze(0), goal_processed], 1).squeeze(1)
        processed, mem = self.processor(x, mem)
        movement = self.movement_chooser(processed)
        if self.using_utterances:
            # utter = self.utterance_chooser(processed) ##OLD CODE
            # perform forward for the language model
            utter = ["Red","agent","go","to","green","landmark","<eos>",""] #TODO: do not hard code sentance, rather generate it using the logic we wrote and pad
            encoded_utter = self.dataset_dictionary.word_dict.w2i(utter)
            inpt = [encoded_utter] * 512
            inpt = Variable(torch.LongTensor(inpt))
            inpt = inpt.transpose(0,1)
            # create initial hidden state for the language rnn
            lang_h = self.lm_model.zero_hid(processed.size(0), self.lm_model.args['nhid_lang'])
            out, lang_h = self.lm_model.forward_lm(inpt, lang_h, processed.unsqueeze(0))

            tgt = Variable(torch.LongTensor(encoded_utter * 512))
            loss = self.crit(out.view(-1, len(self.dataset_dictionary.word_dict)), tgt)
            self.opt.zero_grad()
            # backward step with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.lm_model.parameters(),
                                           self.args['clip'])
            self.opt.step()

            #TODO: we stopped here
            if training:
                utterance = self.gumbel_softmax(utter)
            else:
                utterance = torch.zeros(utter.size())
                if self.using_cuda:
                    utterance = utterance.cuda()
                max_utter = utter.max(1)[1]
                max_utter = max_utter.data[0]
                utterance[0, max_utter] = 1
        else:
            utterance = None
        final_movement = (movement * 2 * self.movement_step_size) - self.movement_step_size
        return final_movement, utterance, mem
