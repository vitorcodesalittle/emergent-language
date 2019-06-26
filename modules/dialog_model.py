# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
An RNN based dialogueue model. Performce both language and choice generation.
"""

import re
import torch
import torch.nn as nn
import torch.nn.init
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from modules import modules_for_lm


class DialogModel(modules_for_lm.CudaModule):
    def __init__(self, word_dict, item_dict, context_dict, output_length, config, device_id):
        super(DialogModel, self).__init__(device_id)

        self.word_dict = word_dict
        self.config = config

        # embedding for words
        self.word_encoder = nn.Embedding(len(self.word_dict), config.nembed_word)

        # a reader RNN, to encode words
        self.reader = nn.GRU(
            input_size=config.nhid_ctx + config.nembed_word,
            hidden_size=config.nhid_lang,
            bias=True)
        self.decoder = nn.Linear(config.nhid_lang, config.nembed_word)
        # a writer, a RNNCell that will be used to generate utterances
        self.writer = nn.GRUCell(
            input_size=config.nhid_ctx + config.nembed_word,
            hidden_size=config.nhid_lang,
            bias=True)

        # tie the weights of reader and writer
        self.writer.weight_ih = self.reader.weight_ih_l0
        self.writer.weight_hh = self.reader.weight_hh_l0
        self.writer.bias_ih = self.reader.bias_ih_l0
        self.writer.bias_hh = self.reader.bias_hh_l0

        self.dropout = nn.Dropout(config.dropout)

        # mask for disabling special tokens when generating sentences
        self.special_token_mask = torch.FloatTensor(len(self.word_dict))

        self.init_weights()

        item_pattern = re.compile('^item([0-9])=([0-9\-])+$')
        # fill in the mask
        for i in range(len(self.word_dict)):
            w = self.word_dict.get_word(i)
            special = item_pattern.match(w) or w in ('<unk>', 'Hi', '<pad>')
            self.special_token_mask[i] = -999 if special else 0.0

        self.special_token_mask = self.to_device(self.special_token_mask)

    def set_device_id(self, device_id):
        self.device_id = device_id
        self.special_token_mask = self.to_device(self.special_token_mask)

    def zero_hid(self, bsz, nhid=None, copies=None):
        """A helper function to create an zero hidden state."""
        nhid = self.config.nhid_lang if nhid is None else nhid
        copies = 1 if copies is None else copies
        hid = torch.zeros(copies, bsz, nhid)
        hid = self.to_device(hid)
        return Variable(hid)

    def init_weights(self):
        """Initializes params uniformly."""
        self.decoder.weight.data.uniform_(-self.config.init_range, self.config.init_range)
        self.decoder.bias.data.fill_(0)

        modules_for_lm.init_rnn(self.reader, self.config.init_range)

        self.word_encoder.weight.data.uniform_(-self.config.init_range, self.config.init_range)

    def read(self, inpt, lang_h, ctx_h, prefix_token="THEM:"):
        """Reads a given utterance."""
        # inpt contains the pronounced utterance
        # add a "THEM:" token to the start of the message
        prefix = Variable(torch.LongTensor(1).unsqueeze(1))
        prefix.data.fill_(self.word_dict.get_idx(prefix_token))
        inpt = torch.cat([self.to_device(prefix), inpt])

        # embed words
        inpt_emb = self.word_encoder(inpt)

        # append the context embedding to every input word embedding
        ctx_h_rep = ctx_h.expand(inpt_emb.size(0), ctx_h.size(1), ctx_h.size(2))
        inpt_emb = torch.cat([inpt_emb, ctx_h_rep], 2)

        # finally read in the words
        self.reader.flatten_parameters()
        out, lang_h = self.reader(inpt_emb, lang_h)

        return out, lang_h

    def word2var(self, word):
        """Creates a variable from a given word."""
        result = Variable(torch.LongTensor(1))
        result.data.fill_(self.word_dict.get_idx(word))
        result = self.to_device(result)
        return result

    STOP_TOKENS = [
        '<eos>'
    ]

    # this is the LSTM network, will be also used in the selfplay mode or in our case in the "fine tune"
    def write(self, lang_h, processed, max_words, temperature,
            stop_tokens=STOP_TOKENS, resume=False):
        """Generate a sentence word by word and feed the output of the
        previous timestep as input to the next.
        """
        lang_h = lang_h[:,1,:]
        processed = processed[:,1,:]
        encoded_pad = self.word_dict.w2i(['<pad>'])
        # btz_total = lang_h.size()[1]
        btz_total = 1
        # lang_h = lang_h.squeeze(0)
        # processed = processed.squeeze(0)
        outs_btz = torch.LongTensor(size=[max_words,btz_total])
        # lang_hs_btz = []
        scores_loss = Variable(torch.FloatTensor(size=[max_words, btz_total, len(self.word_dict.idx2word)]))
        for btz in range(btz_total):
            outs, logprobs, lang_hs = [], [], []
            # remove batch dimension from the language and context hidden states
            # lang_h_btz = lang_h[btz].unsqueeze(0)
            # processed_btz = processed[btz].unsqueeze(0)
            if resume:
                inpt = None
            else:
                # if we start a new sentence, prepend it with 'Hi'
                inpt = Variable(torch.LongTensor(1))
                inpt.data.fill_(self.word_dict.get_idx('Hi'))
                inpt = self.to_device(inpt)
            # generate words until max_words have been generated or <eos>
            for word_idx in range(max_words):
                if inpt is not None:
                    # add the context to the word embedding
                    inpt_emb = torch.cat([self.word_encoder(inpt), processed], 1)
                    # update RNN state with last word
                    lang_h = self.writer(inpt_emb, lang_h)
                    lang_hs.append(lang_h)
                # decode words using the inverse of the word embedding matrix

                out = self.decoder(lang_h)
                scores = F.linear(out, self.word_encoder.weight).div(temperature)
                # subtract constant to avoid overflows in exponentiation
                scores = scores.add(-scores.max().item()).squeeze(0)
                # disable special tokens from being generated in a normal turns
                if not resume:
                    mask = Variable(self.special_token_mask)
                    scores = scores.add(mask)
                prob = F.softmax(scores,dim=0)
                logprob = F.log_softmax(scores,dim=0)

                # explicitly defining num_samples for pytorch 0.4.1

                word = prob.multinomial(num_samples=1).detach()
                # logprob = logprob.gather(0, word)

                # logprobs.append(logprob)

                outs.append(word.view(word.size()[0], 1))
                inpt = word
                scores_loss[word_idx, btz] = Variable(scores)

                # check if we generated an <eos> token
                if self.word_dict.get_word(word.data[0]) in stop_tokens:
                    break
            # update the hidden state with the <eos> token
            inpt_emb = torch.cat([self.word_encoder(inpt), processed], 1)
            lang_h = self.writer(inpt_emb, lang_h)
            lang_hs.append(lang_h)

            # add batch dimension back
            lang_h = lang_h.unsqueeze(1)
            if len(outs)<max_words:
                outs = [outs[i].item() for i in range(len(outs))]
                outs = outs + encoded_pad * (max_words - len(outs))
                outs = [torch.LongTensor([[i]]) for i in outs]
            outs_btz[:,btz] = torch.cat(outs).squeeze(1)
            lang_h[btz]= lang_h
            lang_hs += [torch.cat(lang_hs)]
        return outs_btz, lang_h, lang_hs, scores_loss

    def forward_lm(self, inpt, lang_h, ctx_h):
        """Run forward pass for language modeling."""
        # embed words
        inpt_emb = self.word_encoder(inpt)

        # append the context embedding to every input word embedding
        ctx_h_rep = ctx_h.narrow(0, ctx_h.size(0) - 1, 1).expand(
            inpt.size(0), ctx_h.size(1), ctx_h.size(2))
        inpt_emb = torch.cat([inpt_emb, ctx_h_rep], 2)

        inpt_emb = self.dropout(inpt_emb)

        # compact weights to reduce memory footprint
        self.reader.flatten_parameters()
        out, _ = self.reader(inpt_emb, lang_h)
        decoded = self.decoder(out.view(-1, out.size(2)))

        # tie weights between word embedding/decoding
        decoded = F.linear(decoded, self.word_encoder.weight)

        return decoded.view(out.size(0), out.size(1), decoded.size(1)) , out
