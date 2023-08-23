#!/usr/bin/python
#encoding=utf-8

import math

LOG_ZERO = -99999999.0
LOG_ONE = 0.0

class BeamEntry:
    "information about one single beam at specific time-step"
    def __init__(self):
        self.prTotal=LOG_ZERO      # blank and non-blank
        self.prNonBlank=LOG_ZERO   # non-blank
        self.prBlank=LOG_ZERO      # blank
        self.y=()                  # labelling at current time-step


class BeamState:
    "information about beams at specific time-step"
    def __init__(self):
        self.entries={}

    def norm(self):
        "length-normalise probabilities to avoid penalising long labellings"
        for (k,v) in self.entries.items():
            labellingLen=len(self.entries[k].y)
            self.entries[k].prTotal=self.entries[k].prTotal*(1.0/(labellingLen if labellingLen else 1))

    def sort(self):
        "return beams sorted by probability"
        u=[v for (k,v) in self.entries.items()]
        s=sorted(u, reverse=True, key=lambda x:x.prTotal)
        return [x.y for x in s]

class ctcBeamSearch(object):
    def __init__(self, classes, beam_width, lm, lm_alpha=0.01, blank_index=0):
        self.classes = classes
        self.beamWidth = beam_width
        self.lm_alpha = lm_alpha
        self.lm = lm
        self.blank_index = blank_index

    def log_add_prob(self, log_x, log_y):
        if log_x <= LOG_ZERO:
            return log_y
        if log_y <= LOG_ZERO:
            return log_x
        if (log_y - log_x) > 0.0:
            log_y, log_x = log_x, log_y
        return log_x + math.log(1 + math.exp(log_y - log_x))

    def calcExtPr(self, k, y, t, mat, beamState):
        "probability for extending labelling y to y+k"

        # language model (char bigrams)
        bigramProb=LOG_ONE
        if self.lm:
            c1=self.classes[y[-1]] if len(y) else ''
            c2=self.classes[k]
            bigramProb = self.lm.get_bi_prob(c1,c2) * self.lm_alpha

        # optical model (RNN)
        if len(y) and y[-1]==k and mat[t-1, self.blank_index] < 0.9:
            return math.log(mat[t, k]) + bigramProb + beamState.entries[y].prBlank
        else:
            return math.log(mat[t, k]) + bigramProb + beamState.entries[y].prTotal

    def addLabelling(self, beamState, y):
        "adds labelling if it does not exist yet"
        if y not in beamState.entries:
            beamState.entries[y]=BeamEntry()

    def decode(self, inputs, inputs_list):
        '''
        mat : FloatTesnor batch * timesteps * class
        '''
        batches, maxT, maxC = inputs.size()
        res = []

        for batch in range(batches):
            mat = inputs[batch].numpy()
            # Initialise beam state
            last=BeamState()
            y=()
            last.entries[y]=BeamEntry()
            last.entries[y].prBlank=LOG_ONE
            last.entries[y].prTotal=LOG_ONE

            # go over all time-steps
            for t in range(inputs_list[batch]):
                curr=BeamState()
                #跳过概率很接近1的blank帧，增加解码速度
                if (1 - mat[t, self.blank_index]) < 0.1:
                    continue
                #取前beam个最好的结果
                BHat=last.sort()[0:self.beamWidth]
                # go over best labellings
                for y in BHat:
                    prNonBlank=LOG_ZERO
                    # if nonempty labelling
                    if len(y)>0:
                        #相同的y两种可能，加入重复或者加入空白,如果之前没有字符，在NonBlank概率为0
                        prNonBlank=last.entries[y].prNonBlank + math.log(mat[t, y[-1]])

                    # calc probabilities
                    prBlank = (last.entries[y].prTotal) + math.log(mat[t, self.blank_index])
                    # save result
                    self.addLabelling(curr, y)
                    curr.entries[y].y=y
                    curr.entries[y].prNonBlank = self.log_add_prob(curr.entries[y].prNonBlank, prNonBlank)
                    curr.entries[y].prBlank = self.log_add_prob(curr.entries[y].prBlank, prBlank)
                    prTotal = self.log_add_prob(prBlank, prNonBlank)
                    curr.entries[y].prTotal = self.log_add_prob(curr.entries[y].prTotal, prTotal)

                    #t时刻加入其它的label,此时Blank的概率为0，如果加入的label与最后一个相同，因为不能重复，所以上一个字符一定是blank
                    for k in range(maxC):
                        if k != self.blank_index:
                            newY=y+(k,)
                            prNonBlank=self.calcExtPr(k, y, t, mat, last)

                            # save result
                            self.addLabelling(curr, newY)
                            curr.entries[newY].y=newY
                            curr.entries[newY].prNonBlank = self.log_add_prob(curr.entries[newY].prNonBlank, prNonBlank)
                            curr.entries[newY].prTotal = self.log_add_prob(curr.entries[newY].prTotal, prNonBlank)

                # set new beam state
                last=curr

            BHat=last.sort()[0:self.beamWidth]
            # go over best labellings
            curr = BeamState()
            for y in BHat:
                newY = y
                c1 = self.classes[y[-1]]
                c2 = ""
                prNonBlank = last.entries[newY].prTotal + self.lm.get_bi_prob(c1, c2) * self.lm_alpha
                self.addLabelling(curr, newY)
                curr.entries[newY].y=newY
                curr.entries[newY].prNonBlank = self.log_add_prob(curr.entries[newY].prNonBlank, prNonBlank)
                curr.entries[newY].prTotal = self.log_add_prob(curr.entries[newY].prTotal, prNonBlank)

            last = curr
            # normalise probabilities according to labelling length
            last.norm()

            # sort by probability
            bestLabelling=last.sort()[0] # get most probable labelling

            # map labels to chars
            res_b =' '.join([self.classes[l] for l in bestLabelling])
            res.append(res_b)
        return res








###################################################


import re
import math

n_grams = ["unigram", 'bigram', 'trigram', '4gram', '5gram']

class LanguageModel:
    """
    New version of LanguageModel which can read the text arpa file ,which
    is generate from kennlm
    """
    def __init__(self, arpa_file=None, n_gram=2, start='<s>', end='</s>', unk='<unk>'):
        "Load arpa file to get words and prob"
        self.n_gram = n_gram
        self.start = start
        self.end = end
        self.unk = unk
        self.scale = math.log(10)    #arpa格式是以10为底的对数概率，转化为以e为底
        self.initngrams(arpa_file)

    def initngrams(self, fn):
        "internal init of word bigrams"
        self.unigram = {}
        self.bigram = {}
        if self.n_gram == 3:
            self.trigrame = {}

	    # go through text and create each bigrams
        f = open(fn, 'r')
        recording = 0
        for lines in f.readlines():
            line = lines.strip('\n')
            #a = re.match('gram', line)
            if line == "\\1-grams:":
                recording = 1
                continue
            if line == "\\2-grams:":
                recording = 2
                continue
            if recording == 1:
                line = line.split('\t')
                if len(line) == 3:
                    self.unigram[line[1]] = [self.scale * float(line[0]), self.scale * float(line[2])]   #save the prob and backoff prob
                elif len(line) == 2:
                    self.unigram[line[1]] = [self.scale * float(line[0]), 0.0]
            if recording == 2:
                line = line.split('\t')
                if len(line) == 3:
                    #print(line[1])
                    self.bigram[line[1]] = [self.scale * float(line[0]), self.scale * float(line[2])]
                elif len(line) == 2:
                    self.bigram[line[1]] = [self.scale * float(line[0]), 0.0]
        f.close()
        self.unigram['UNK'] = self.unigram[self.unk]


    def get_uni_prob(self, wid):
        "Returns unigram probabiliy of word"
        return self.unigram[wid][0]

    def get_bi_prob(self, w1, w2):
        '''
        Return bigrams probability p(w2 | w1)
        if bigrame does not exist, use backoff prob
        '''
        if w1 == '':
            w1 = self.start
        if w2 == '':
            w2 = self.end
        key = w1 + ' ' + w2
        if key not in self.bigram:
            return self.unigram[w1][1] + self.unigram[w2][0]
        else:
            return self.bigram[key][0]

    def score_bg(self, sentence):
        '''
        Score a sentence using bigram, return P(sentence)
        '''
        val = 0.0
        words = sentence.strip().split()
        val += self.get_bi_prob(self.start, words[0])
        for i in range(len(words)-1):
            val += self.get_bi_prob(words[i], words[i+1])
        val += self.get_bi_prob(words[-1], self.end)
        return val



##############################################################

import numpy as np

class Decoder(object):
    "解码器基类定义，作用是将模型的输出转化为文本使其能够与标签计算正确率"
    def __init__(self, blank_index = 0, remove_rep=True, silent = None):
        self.blank_index = blank_index
        self.silent = silent
        self.remove_rep = remove_rep
    def process_lists(self,seqs):
            processed_lists = []
            for seq in seqs:
                list = self.process_list(seq)
                processed_lists.append(list)
            return processed_lists

    def process_list(self,seq):
        list = []
        for i, char in enumerate(seq):
            if char != self.blank_index and char not in self.silent:
                if self.remove_rep and i != 0 and char == seq[i - 1]: #remove dumplicates
                    pass
                else:
                    list = list + [char]
        return list

    def edit_distance(self, src_seq, tgt_seq):
        L1, L2 = len(src_seq), len(tgt_seq)
        if L1 == 0: return L2
        if L2 == 0: return L1
        # construct matrix of size (L1 + 1, L2 + 1)
        dist = [[0] * (L2 + 1) for i in range(L1 + 1)]
        for i in range(1, L2 + 1):
            dist[0][i] = dist[0][i-1] + 1
        for i in range(1, L1 + 1):
            dist[i][0] = dist[i-1][0] + 1
        for i in range(1, L1 + 1):
            for j in range(1, L2 + 1):
                if src_seq[i - 1] == tgt_seq[j - 1]:
                    cost = 0
                else:
                    cost = 1
                dist[i][j] = min(dist[i][j-1] + 1, dist[i-1][j] + 1, dist[i-1][j-1] + cost)
        return dist[L1][L2]


    def phone_word_error(self, preds, targets):
        '''计算词错率和字符错误率
        Args:
            prob_tensor     :   模型的输出
            frame_seq_len   :   每个样本的帧长
            targets         :   样本标签
            target_sizes    :   每个样本标签的长度
        Returns:
            wer             :   词错率，以space为间隔分开作为词
            cer             :   字符错误率
        '''
        preds = preds.detach().cpu().numpy().tolist()
        targets = targets.detach().cpu().numpy().tolist()
        preds = self.process_lists(preds)
        cer = 0
        num_char=0
        for x in range(len(targets)):
            cer += self.edit_distance(preds[x], targets[x])
            num_char += len(targets[x])
        return cer/num_char



import torch
class GreedyDecoder(Decoder):
    "直接解码，把每一帧的输出概率最大的值作为输出值，而不是整个序列概率最大的值"
    def decode(self, prob_tensor, frame_seq_len):
        '''解码函数
        Args:
            prob_tensor   :   网络模型输出
            frame_seq_len :   每一样本的帧数
        Returns:
            解码得到的string，即识别结果
        '''
        prob_tensor = prob_tensor.transpose(0,1)
        _, decoded = torch.max(prob_tensor, 2)
        decoded = decoded.view(decoded.size(0), decoded.size(1))
        decoded = self._convert_to_strings(decoded.numpy(), frame_seq_len)
        return self._process_strings(decoded, remove_rep=True)

class BeamDecoder(Decoder):
    "Beam search 解码。解码结果为整个序列概率的最大值"
    def __init__(self, int2char, beam_width = 200, blank_index = 0, lm_path=None, lm_alpha=0.01):
        self.beam_width = beam_width
        super(BeamDecoder, self).__init__(int2char, blank_index=blank_index)

        import sys
        sys.path.append('../')
        # import utils.BeamSearch as uBeam
        # import utils.NgramLM as uNgram
        lm = LanguageModel(arpa_file=lm_path)
        self._decoder = ctcBeamSearch(int2char, beam_width, lm, lm_alpha=lm_alpha, blank_index = blank_index)

    def decode(self, prob_tensor, frame_seq_len=None):
        '''解码函数
        Args:
            prob_tensor   :   网络模型输出
            frame_seq_len :   每一样本的帧数
        Returns:
            res           :   解码得到的string，即识别结果
        '''
        probs = prob_tensor.transpose(0, 1)
        probs = torch.exp(probs)
        res = self._decoder.decode(probs, frame_seq_len)
        return res
