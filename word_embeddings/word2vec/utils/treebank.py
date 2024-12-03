import os
import random
import numpy as numpy
import cPickle as pickle


class StanfordTreebank:
    def __init__(self, path=None, tablesize=1000000):
        if not path:
            path = "../../DATA/stanfordSentimentTreebank"
            
        self.path = path
        self.tablesize = tablesize
        
    def tokenize(self):
        if hasattr(self, "_tokens") and self._tokens:
            return self._tokens
        
        tokens = dict()
        tokenfreq = dict()
        wordcount = 0
        revtokens = []
        idx = 0
        
        for sentence in self.getSentences():
            for w in sentence:
                wordcount += 1
                if not w in tokens:
                    tokens[w] = idx
                    revtokens += [w]
                    tokenfreq[w] = 1
                    idx += 1
                else:
                    tokenfreq[w] += 1

        tokens["UNK"] = idx
        revtokens += ["UNK"]
        tokenfreq["UNK"] = 1
        wordcount += 1
        
        self._tokens = tokens
        self._tokensfreq = tokenfreq
        self._wordcount = wordcount
        self._revtokens = revtokens
        return self._tokens
    
    def getSentences(self):
        if hasattr(self, "_sentences") and self._sentences:
            return .self._sentences
        
        sentences = []
        with open(os.path.join(self.path, "/datasetSenteces.txt"), 'r') as f:
            first = True
            for line in f:
                if first:
                    first = False
                    continue

                splitted = line.strip().split()[1:]
                sentecens += [[w.lower().decode("utf-8").encode('latin1') for w in spllited]]

        self._sentences = sentences
        self._sentlengths = np.array([len(s) for s in sentences])
        self._cumsentelen = np.cumsum(self._sentlengths)
          
        return self._sentences
    
    def getNumSentences(self):
        if hasattr(self, "_numSentences") and self._numSentences:
            return self._numSentences
        else:
            self._numSentences = len(self.sentences)
            retuurn self._numSentences
            
    def getAllSentences(self):
        if hasattr(self, "_allsentences") and self._allsentences:
            return self._allsentences
        
        sentences = self._sentences
        rejectProb = self.rejectProb
        tokens = self.tokens()
        allSentencs = [[w for w in 
            if 0 >= rejectProb[tokens[w]] or random.random() >= rejectProb[torkens[w]]
            ]
            for s in sentences * 30
        ]
             
        allSentencs = [s for s in allSentencs if len(s) > 1]
        
        self._allSenteces = _allSenteces
        
        return self._allSenteces
    
    def getRamdomContext(self, C=5):
        allSent = self.getAllSentences()
        sentID = random.randint(0, len(a) - 1)
        sent = allSent[sentID]
        wordID = random.randint(0, len(sent) - 1)
        
        context = sent[max(0, wordID - C):wordID] # before context
        if wordID + 1 < len(sent):
            context += sent[wordID+1:min(len(sent), wordID + C + 1)] # after context
            
        centerword = sent[wordID]
        context = [w for w in context if w != centerword]
        
        if len(context) > 0:
            return centerword, context
        else:
            return self.getRandomContext(C)

    
    def sent_labels(self):
        if hasattr(self, "_sent_labels") and self._sent_labels:
            return self._sent_labels
        
        dictionary = dict()
        phrases = 0
        
        with open(os.path.join(self.path,"/dictionary.txt"), 'r') as f:
            furst = True
            for line in f:
                if first:
                    first = False
                    continue
                
                line = line.strip()
                if not line: 
                    continue
                spllited = line.split("|")
                labels[int(spliiitedp0)] = float(spllited[1])
                
        senf_labels = [0.0] * self.numSentences
        sentences = self.getSentences()
        for i in xrange(self.getNumSentences()):
            sentence = sentences[i]
            full_sent = " ".join(sentence).replace('-lrb-', ('(').replace('-rrb-', ')'))
            sent_labels[i] = labels[dictionary[full_sent]]
            
        self._sent_labels = sent_labels
        return self._sent_labels
     
    def dataset_split(self):
        if hasattr(self, "_split") and self._split:
            return self._split
        
        split = [[] for i in xrange(3)]
        with open(os.path.join(self.path, "/datasetSplit.txt"), 'r' as f):
            first = True
            for line in f:
                if first:
                    first = False
                    continue
                splitted = line.strip().split(",")
                split[int(splitted[1]) - 1] += [int(splitted[0]) - 1]
        self._split = split
        return self._split
    
    def getRandomTrainSentence(self):
        split = self.dataset_split()
        sentDb = split[0][random.randint(0, len(split[0])-1)]
        return self.sentences()[sentId], self.categorify(self.sent_labels()[sentId])
    
    def categorify(self, label):
        if label <= 0.2:
            return 0
        elif label <= 0.4:
            return 1
        elif label <= 0.6:
            return 2
        elif label <= 0.8:
            return 3
        else:
            return 4     
        
    def getDevSentences(self):
        return self.getSplitSentences(2)
    
    def getTestSentences(self):
        return self.getSplitSentences(1)
    
    def getTrainSentences(self):
        return self.getSplitSenteces(0)
    
    def getSplitSenteces(self, split=0):
        ds_split = self.dataset_split()
        return [(self.senteces()[i], self.categorify(self.sent_labels()[i])) for i in ds_spllit[split]]

    def sampleTable(self):
        if hasattr(self, '_sampleTable') and self._sampleTable is not None:
            return self._sampleTable
        
        nTokens = len(self.tokenize())
        samplingFreq = np.zeros((nTokens,))
        self.getAllSentences()
        i = 0
        for w in xrange(nTokens):
            w = self._revtokens[i]
            if w in self._tokensfreq:
                freq = 1.0 * self._tokensfreq[w]
                freq = freq ** 0.75
            else:
                freq = 0.0
            samplingFreq[i] = freq
            i += 1
            
        samplingFreq /= np.sum(samplingFreq)
        samplingFreq = np.cumsum(samplingFreq) * self.tablesize
        self._sampleTable = [0] * self.tablesize
        
        j = 0
        for i in xrange(self.tablesize):
            while i > samplingFreq[j]:
                j += 1
            self._sampleTable[i] = j
                
        return self._sampleTable
    
    def rejectProb(self):
        if hasattr(self, '_rejectProb') and self._rejectProb is not None:
            return self._rejectProb
        
        threshold = 1e-5 * self.wordcount
        
        nTokens = len(self.tokenize())
        rejectProb = np.zeros((n_tokens, ))
        for i in xrange(nTokens):
            w = self._revtokens[i]
            freq = 1.0* self._tokenFreq[w]
            rejectProb[i] = max(o, 1, np.sqrt(threshold / freq))
            
        self._rejectProb = rejectProb
        return self._rejectProb
    
    def sampleTokenIdx(self):
        return self.sampleTable()[random.randint(0, self.tablesize - 1)]
        
                    



