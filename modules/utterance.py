import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import numpy as np
from modules.gumbel_softmax import GumbelSoftmax
from configs import DEFAULT_VOCAB_SIZE
import torch.nn.functional as F
from modules.dialog_model import DialogModel
from modules.modules_for_lm import Criterion


class Utterance(nn.Module):
    def __init__(self, action_processor_config, utterance_config, dataset_dictionary, use_utterance_old_code):
        super(Utterance, self).__init__()
        self.use_utterance_old_code = use_utterance_old_code

        self.utterance_chooser = nn.Sequential(
                    nn.Linear(action_processor_config.action_processor.hidden_size, action_processor_config.hidden_size),
                    nn.ELU(),
                    nn.Linear(action_processor_config.hidden_size, action_processor_config.vocab_size))
        self.gumbel_softmax = GumbelSoftmax(action_processor_config.use_cuda)

        self.dataset_dictionary = dataset_dictionary
        self.lm_model = DialogModel(dataset_dictionary.word_dict, None,
                                None, 4,
                                utterance_config,
                                None)
        self.crit = Criterion(dataset_dictionary.word_dict, device_id=None)
        # self.opt = optim.SGD(self.lm_model.parameters(), lr=utterance_config.lr,
        #                  momentum=utterance_config.momentum,
        #                  nesterov=(utterance_config.nesterov and utterance_config.momentum > 0))
        self.opt = optim.Adam(self.lm_model.parameters(),
                              lr=utterance_config.lr, betas=(0.9, 0.999),
                              eps=1e-08, weight_decay=0, amsgrad=False)
        self.total_loss = 0
        # embedding for words
        self.word_encoder = nn.Embedding(len(dataset_dictionary.word_dict), utterance_config.nembed_word)
        # a writer, a RNNCell that will be used to generate utterances
        self.writer = nn.GRUCell(
            input_size=utterance_config.nhid_ctx + utterance_config.nembed_word,
            hidden_size=utterance_config.nhid_lang,
            bias=True)
        self.decoder = nn.Linear(utterance_config.nhid_lang, utterance_config.nembed_word)
        self.config = utterance_config

    def forward(self, processed, full_sentence, mode=None):
        # perform forward for the language model
        utter = full_sentence.tolist()
        # utter = ['Hi red agent continue <eos>']*32 #TMP just for testing
        encoded_utter = np.array([self.dataset_dictionary.word_dict.w2i(utter[i].split(" "))
                                  for i in range(len(full_sentence))])
        encoded_pad = self.dataset_dictionary.word_dict.w2i(['<pad>'])
        # longest_sentence = len(max(encoded_utter, key=len))
        longest_sentence = DEFAULT_VOCAB_SIZE
        encoded_utter = [encoded_utter[i] + encoded_pad * (longest_sentence - len(encoded_utter[i]))
                         if len(encoded_utter[i]) < longest_sentence else encoded_utter[i]
                         for i in range(len(full_sentence))]
        encoded_utter = Variable(torch.LongTensor(encoded_utter))
        encoded_utter = encoded_utter.transpose(0, 1)

        # create initial hidden state for the language rnn
        lang_h = self.lm_model.zero_hid(processed.size(0), self.lm_model.config.nhid_lang)
        out, lang_h = self.lm_model.forward_lm(encoded_utter, lang_h, processed.unsqueeze(0))

        # remove batch dimension from the language and context hidden states
        lang_h = lang_h.squeeze(1)

        # inpt2 = Variable(torch.LongTensor(1, self.config.batch_size))
        # inpt2.data.fill_(self.dataset_dictionary.word_dict.get_idx('Hi'))

        # decode words using the inverse of the word embedding matrix
        decoded_lang_h = self.decoder(lang_h)
        scores = F.linear(decoded_lang_h, self.word_encoder.weight).div(self.config.temperature)
        # subtract constant to avoid overflows in exponentiation
        scores = scores.add(-scores.max().item()).squeeze(0)

        prob = F.softmax(scores, dim=2)
        word = torch.transpose(prob, 0, 1).multinomial(num_samples=DEFAULT_VOCAB_SIZE).detach()
        tgt = encoded_utter.reshape(encoded_utter.shape[0]*encoded_utter.shape[1])
        # in FB code the inpt and tgt is one dimension less than the original data
        loss = self.crit(out.view(-1, len(self.dataset_dictionary.word_dict)), tgt)
        if mode is None:

            # backward step with gradient clipping, use retain_graph=True
            loss.backward(retain_graph=True)
            self.opt.zero_grad()
            torch.nn.utils.clip_grad_norm_(self.lm_model.parameters(),
                                           self.config.clip)
            self.opt.step()
            print(loss)
            print(self.dataset_dictionary.word_dict.i2w(word[1, :]))
            return loss, word
        else:
            self.total_loss += loss
            self.opt.zero_grad()
            print(self.total_loss)
            print(self.dataset_dictionary.word_dict.i2w(word[1, :]))
            return self.total_loss, word

    def create_utterance_using_old_code(self, training, processed):
        utter = self.utterance_chooser(processed)
        if training:
            utterance = self.gumbel_softmax(utter)
        else:
            utterance = torch.zeros(utter.size())
            if self.using_cuda:
                utterance = utterance.cuda()
            max_utter = utter.max(1)[1]
            max_utter = max_utter.data[0]
            utterance[0, max_utter] = 1
        return utterance
