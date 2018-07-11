import numpy as np
import logging
import copy
from IPython.core.debugger import set_trace

# Alignment algorithm code was adapted from the web: https://github.com/alevchuk/pairwise-alignment-in-python

class Alignment:
    def __init__(self,w2v,match_award=1,mismatch_penalty=0,gap1_penalty=-100000,gap2_penalty=0):
        self.match_award = match_award
        self.mismatch_penalty = mismatch_penalty
        self.gap1_penalty = gap1_penalty
        self.gap2_penalty = gap2_penalty
        self.w2v = w2v
        
    def apply_mask(self,annotation_words,annotation_mask):
        inxs = np.where(np.array(annotation_mask) == True)
        annotation_words = np.array(annotation_words)[inxs]
        return annotation_words

    # return the score and mask
    def score(self,words,annotation_words,computed_mask=None):
        #import pdb
        #pdb.set_trace()
        words = copy.deepcopy(words)
        nwords = len(words)
        identity, score, align1, align2 = self.needle(words,annotation_words)
        score = score*1./len(annotation_words)
        score = round(score*100)/100.

        if computed_mask != None:
            matched_inxs = np.where(np.array(align2) != '-')
            matched_words = np.array(align1)[matched_inxs]
            # Need to convert the inxs to the original space because we could have added a gap
            matched_inxs = []
            for mword in matched_words:
                if mword != '-':
                    inx = words.index(mword)
                    words[inx] = 'this_word_is_removed'
                    matched_inxs.append(inx)

            mask = np.full(nwords, fill_value=False)
            mask[matched_inxs] = True
            computed_mask.append(mask.tolist())
        return score

    def train_v1(self,positive_words,positive_masks,negative_words,negative_masks):
        self.positive_words = positive_words
        self.positive_masks = positive_masks
        
    def train_v2(self,positive_words,positive_masks,negative_words,negative_masks):
        self.positive_words = positive_words
        self.positive_masks = positive_masks
        self.negative_words = negative_words
        self.negative_masks = negative_masks
    
    # performs the prediction using only the positive words and uses the max
    def predict_v1(self,words):
        positive_scores = []
        masks = []
        for i in range(len(self.positive_words)):
            sc, mask = self.score_v1(words,self.positive_words[i],self.positive_masks[i])
            positive_scores.append(sc)
            masks.append(mask)
        #if len(positive_scores) > 5:
        #    set_trace()
        inx = np.argmax(np.array(positive_scores))
        score = positive_scores[inx]
        mask = masks[inx]
        return score,mask

    # performs the prediction using only the positive words and uses the max
    def predict_v2(self,words):
        positive_scores = []
        positive_masks = []
        negative_scores = []
        negative_masks = []
        for i in range(len(self.positive_words)):
            sc, mask = self.score_v1(words,self.positive_words[i],self.positive_masks[i])
            positive_scores.append(sc)
            positive_masks.append(mask)
        for i in range(len(self.positive_words)):
            sc, mask = self.score_v1(words,self.negative_words[i],self.negative_masks[i])
            negative_scores.append(sc)
            negative_masks.append(mask)
        inx = np.argmax(np.array(positive_scores))
        pos_score = positive_scores[inx]
        pos_mask = positive_masks[inx]
        inx = np.argmax(np.array(negative_scores))
        neg_score = negative_scores[inx]
        neg_mask = negative_masks[inx]
        return pos_score,pos_mask,neg_score,neg_mask
    
    def zeros(self,shape):
        retval = []
        for x in range(shape[0]):
            retval.append([])
            for y in range(shape[1]):
                retval[-1].append(0)
        return retval

    def match_score(self,alpha, beta):
        if alpha == beta:
            return self.match_award
        elif alpha == '-':
            return self.gap1_penalty
        elif beta == '-':
            return self.gap2_penalty
        elif self.w2v != None:
            try:
                #import pdb
                #pdb.set_trace()
                return self.w2v.similarity(alpha,beta)
            except KeyError as ex:
                #if alpha not in self.w2v and beta in self.w2v:
                #    print('"%s" not in vocab. "%s" in vocab.: %s.'%(alpha,beta,ex))
                #elif alpha in self.w2v and beta not in self.w2v:
                #    print('"%s" in vocab. "%s" not in vocab.: %s.'%(alpha,beta,ex))
                #else:
                #    print('Both "%s" and "%s" not in vocab: %s.'%(alpha,beta,ex))
                return 0.
        else:
            return self.mismatch_penalty # here is where w2vec should go I think

    def finalize(self,align1, align2):
        align1 = align1[::-1]    #reverse sequence 1
        align2 = align2[::-1]    #reverse sequence 2

        i,j = 0,0

        #calcuate identity, score and aligned sequeces
        symbol = []
        found = 0
        score = 0
        identity = 0
        for i in range(0,len(align1)):
            # if two AAs are the same, then output the letter
            if align1[i] == align2[i]:
                symbol = symbol + [align1[i]]
                identity = identity + 1
                score += self.match_score(align1[i], align2[i])

            # if they are not identical and none of them is gap
            elif align1[i] != align2[i] and align1[i] != '-' and align2[i] != '-':
                score += self.match_score(align1[i], align2[i])
                symbol += [' ']
                found = 0

            #if one of them is a gap, output a space
            elif align1[i] == '-':
                symbol += [' ']
                score += self.gap1_penalty
            elif align2[i] == '-':
                symbol += [' ']
                score += self.gap2_penalty

        identity = float(identity) / len(align1) * 100

        return identity, score, align1, align2


    def needle(self,seq1, seq2):
        m, n = len(seq1), len(seq2)  # length of two sequences

        # Generate DP table and traceback path pointer matrix
        score = self.zeros((m+1, n+1))      # the DP table

        # Calculate DP table
        for i in range(0, m + 1):
            score[i][0] = self.gap2_penalty * i
        for j in range(0, n + 1):
            score[0][j] = self.gap1_penalty * j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                match = score[i - 1][j - 1] + self.match_score(seq1[i-1], seq2[j-1])
                delete = score[i - 1][j] + self.gap2_penalty
                insert = score[i][j - 1] + self.gap1_penalty
                score[i][j] = max(match, delete, insert)

        # Traceback and compute the alignment
        align1, align2 = [], []
        i,j = m,n # start from the bottom right cell
        while i > 0 and j > 0: # end toching the top or the left edge
            score_current = score[i][j]
            score_diagonal = score[i-1][j-1]
            score_up = score[i][j-1]
            score_left = score[i-1][j]

            if score_current == score_diagonal + self.match_score(seq1[i-1], seq2[j-1]):
                align1 += [seq1[i-1]]
                align2 += [seq2[j-1]]
                i -= 1
                j -= 1
            elif score_current == score_left + self.gap2_penalty:
                align1 += [seq1[i-1]]
                align2 += ['-']
                i -= 1
            elif score_current == score_up + self.gap1_penalty:
                align1 += ['-']
                align2 += [seq2[j-1]]
                j -= 1

        # Finish tracing up to the top left cell
        while i > 0:
            align1 += [seq1[i-1]]
            align2 += ['-']
            i -= 1
        while j > 0:
            align1 += ['-']
            align2 += [seq2[j-1]]
            j -= 1

        #print align1
        #print align2
        return self.finalize(align1, align2)

    def water(self,seq1, seq2):
        m, n = len(seq1), len(seq2)  # length of two sequences

        # Generate DP table and traceback path pointer matrix
        score = self.zeros((m+1, n+1))      # the DP table
        pointer = self.zeros((m+1, n+1))    # to store the traceback path

        max_score = 0        # initial maximum score in DP table
        # Calculate DP table and mark pointers
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                score_diagonal = score[i-1][j-1] + self.match_score(seq1[i-1], seq2[j-1])
                score_up = score[i][j-1] + self.gap_penalty
                score_left = score[i-1][j] + self.gap_penalty
                score[i][j] = max(0,score_left, score_up, score_diagonal)
                if score[i][j] == 0:
                    pointer[i][j] = 0 # 0 means end of the path
                if score[i][j] == score_left:
                    pointer[i][j] = 1 # 1 means trace up
                if score[i][j] == score_up:
                    pointer[i][j] = 2 # 2 means trace left
                if score[i][j] == score_diagonal:
                    pointer[i][j] = 3 # 3 means trace diagonal
                if score[i][j] >= max_score:
                    max_i = i
                    max_j = j
                    max_score = score[i][j];

        align1, align2 = [], []    # initial sequences

        i,j = max_i,max_j    # indices of path starting point

        #traceback, follow pointers
        while pointer[i][j] != 0:
            if pointer[i][j] == 3:
                align1 += [seq1[i-1]]
                align2 += [seq2[j-1]]
                i -= 1
                j -= 1
            elif pointer[i][j] == 2:
                align1 += ['-']
                align2 += [seq2[j-1]]
                j -= 1
            elif pointer[i][j] == 1:
                align1 += [seq1[i-1]]
                align2 += ['-']
                i -= 1

        return self.finalize(align1, align2)
