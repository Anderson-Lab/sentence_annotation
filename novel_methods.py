#from . import word2vec
import numpy as np
import alignment

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
from sklearn.pipeline import Pipeline
from .common import *
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import HashingVectorizer

from . import alignment
from . import common

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.base import BaseEstimator, ClassifierMixin

class AlignmentThresholdClassifier(BaseEstimator, ClassifierMixin):  
    """An example of classifier"""

    def __init__(self, alignment, storage_func, mask_func, pos_label="Test", threshold=0.8):
        """
        Called when initializing the classifier
        """
        self.threshold = threshold
        self.alignment = alignment
        self.storage_func = storage_func
        self.mask_func = mask_func

    def fit(self, X, y=None):
        """
        """
        pos_inxs = np.where(y == 1)[0]
        self.X = list(np.array(X)[pos_inxs])
        self.y = list(np.array(y)[pos_inxs])
        return self

    def predict(self, X, y=None):
        try:
            getattr(self, "X")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")
            
        scores = self.score(X, y = y)
        
        return([2*int(score >= self.threshold)-1 for score in scores])

    def score(self, X, y=None):
        # counts number of values bigger than mean
        scores = []
        max_computed_mask = None
        max_labeled_sample_words = None
        for sample in X:
            max_score = None
            for labeled_sample in self.X:
                sample_words = self.storage_func()[int(sample)]
                labeled_sample_words = self.mask_func()[int(labeled_sample)]
                computed_mask = []
                score = self.alignment.score(sample_words,labeled_sample_words,computed_mask=computed_mask)
                if max_score == None or score > max_score:
                    max_score = score
                    max_computed_mask = computed_mask
                    max_labeled_sample_words = labeled_sample_words
            if max_score < 1 and max_score >= 0.85:
                print('Score: ',max_score)
                j = 0
                sample_line_fields = []
                labeled_line_fields = []
                for i,word in enumerate(sample_words):
                    sample_line_fields.append(word)
                    if max_computed_mask[0][i]:
                        labeled_line_fields.append(max_labeled_sample_words[j])
                        j += 1
                    else:
                        labeled_line_fields.append("".join(["-" for c in word]))
                top_line = " & ".join(sample_line_fields)
                bottom_line = " & ".join(labeled_line_fields)
                print("".join(["c" for c in top_line]))
                print(top_line)
                print(bottom_line)
            scores.append(max_score)
        return scores

def create_pipeline_alignment_threshold(alignment, threshold=0.8):
    storage = []
    storage_func = lambda: storage
    mask_storage = []
    mask_func = lambda: mask_storage
    clf = AlignmentThresholdClassifier(alignment,storage_func,mask_func,threshold=threshold)
    pipeline = Pipeline([
        ('selector', AlignmentSelector(key='sentence',mask_key='mask',storage_func=storage_func,mask_func=mask_func)),

        # Use classifier
        ('clf', clf)
    ])
    return pipeline
    
def create_pipeline(first_arg,get_distance_metric_func):
    storage = []
    storage_func = lambda: storage
    mask_storage = []
    mask_func = lambda: mask_storage
    max_metric = get_distance_metric_func(first_arg,storage_func,mask_func)
    clf = KNeighborsClassifier(n_neighbors=3, metric=max_metric)
    pipeline = Pipeline([
        ('selector', AlignmentSelector(key='sentence',mask_key='mask',storage_func=storage_func,mask_func=mask_func)),
        
        #('vectorizer', vectorizer),
        
        #('dbg',Debug()),

        # Use classifier
        ('clf', clf)
    ])
    return pipeline

def return_max_similarity(words,annotation_words,w2v):
    max_sim = -np.inf
    for word in words:
        for annotation_word in annotation_words:
            #import pdb
            #pdb.set_trace()
            try:
                sim = w2v.similarity(word,annotation_word)
            except KeyError as ex:
                sim = 0
            if sim > max_sim:
                max_sim = sim
    return max_sim

def return_mean_similarity(words,annotation_words,w2v):
    sim_values = []
    for word in words:
        for annotation_word in annotation_words:
            try:
                sim = w2v.similarity(word,annotation_word)
            except KeyError as ex:
                sim = 0
            sim_values.append(sim)
    return np.mean(sim_values)

def get_max_distance_metric(w2v,storage_func,mask_func):
    mydist = lambda x,y: 1-return_max_similarity(mask_func()[int(x)],storage_func()[int(y)],w2v)
    return mydist

def get_mean_distance_metric(w2v,storage_func,mask_func):
    mydist = lambda x,y: 1-return_mean_similarity(mask_func()[int(x)],storage_func()[int(y)],w2v)
    return mydist

def get_alignment_distance_metric(alignment,storage_func,mask_func):
    mydist = lambda x,y: 1 - alignment.score(storage_func()[int(y)],mask_func()[int(x)])
    return mydist

def alignment_v1(w2v,threshold,training_positive_annots,training_negative_annots,test_positive_annots,test_negative_annots):
    train_positive_words = [entry["words"] for entry in training_positive_annots]
    train_negative_words = [entry["words"] for entry in training_negative_annots]
    test_positive_words = [entry["words"] for entry in test_positive_annots]
    test_negative_words = [entry["words"] for entry in test_negative_annots]

    train_positive_masks = [entry["mask"] for entry in training_positive_annots]
    train_negative_masks = [entry["mask"] for entry in training_negative_annots]
    test_positive_masks = [entry["mask"] for entry in test_positive_annots]
    test_negative_masks = [entry["mask"] for entry in test_negative_annots]

    alignment_method = alignment.Alignment(w2v)

    alignment_method.train_v1(train_positive_words,train_positive_masks,train_negative_words,train_negative_masks)

    train_positive_labels = np.ones(len(train_positive_words),).tolist()
    train_negative_labels = np.zeros(len(train_negative_words),).tolist()
    train_labels = train_positive_labels + train_negative_labels
    test_positive_labels = np.ones(len(test_positive_words),).tolist()
    test_negative_labels = np.zeros(len(test_negative_words),).tolist()
    test_labels = test_positive_labels + test_negative_labels

    train_data = train_positive_words + train_negative_words
    training_pred_scores = []
    training_pred_masks = []
    for words in train_data:
        score,mask = alignment_method.predict_v1(words)
        training_pred_scores.append(score)
        training_pred_masks.append(mask)
    training_pred_labels = [float(score >= threshold) for score in training_pred_scores]

    test_data = test_positive_words + test_negative_words
    testing_pred_scores = []
    testing_pred_masks = []
    for words in test_data:
        score,mask = alignment_method.predict_v1(words)
        testing_pred_scores.append(score)
        testing_pred_masks.append(mask)
    testing_pred_labels = [float(score >= threshold) for score in testing_pred_scores]

    return np.array(training_pred_labels), np.array(testing_pred_labels), np.array(train_labels), np.array(test_labels), np.array(training_pred_scores), np.array(training_pred_masks), np.array(testing_pred_scores), np.array(testing_pred_masks) 

def alignment_v2(w2v,training_positive_annots,training_negative_annots,test_positive_annots,test_negative_annots):
    train_positive_words = [entry["words"] for entry in training_positive_annots]
    train_negative_words = [entry["words"] for entry in training_negative_annots]
    test_positive_words = [entry["words"] for entry in test_positive_annots]
    test_negative_words = [entry["words"] for entry in test_negative_annots]

    train_positive_masks = [entry["mask"] for entry in training_positive_annots]
    train_negative_masks = [entry["mask"] for entry in training_negative_annots]
    test_positive_masks = [entry["mask"] for entry in test_positive_annots]
    test_negative_masks = [entry["mask"] for entry in test_negative_annots]

    alignment_method = alignment.Alignment(w2v)

    alignment_method.train_v2(train_positive_words,train_positive_masks,train_negative_words,train_negative_masks)

    train_positive_labels = np.ones(len(train_positive_words),).tolist()
    train_negative_labels = np.zeros(len(train_negative_words),).tolist()
    train_labels = train_positive_labels + train_negative_labels
    test_positive_labels = np.ones(len(test_positive_words),).tolist()
    test_negative_labels = np.zeros(len(test_negative_words),).tolist()
    test_labels = test_positive_labels + test_negative_labels

    train_data = train_positive_words + train_negative_words
    training_pred_scores = []
    training_pred_masks = []
    training_pred_labels = []
    for words in train_data:
        pos_score,pos_mask,neg_score,neg_mask = alignment_method.predict_v2(words)
        if pos_score > neg_score:
            score = pos_score
            mask = pos_mask
            label = 1.0
        else:
            score = 1 - neg_score
            mask = neg_mask
            label = 0.0
        training_pred_scores.append(score)
        training_pred_masks.append(mask)
        training_pred_labels.append(label)

    test_data = test_positive_words + test_negative_words
    testing_pred_scores = []
    testing_pred_masks = []
    testing_pred_labels = []
    for words in test_data:
        pos_score,pos_mask,neg_score,neg_mask = alignment_method.predict_v2(words)
        if pos_score > neg_score:
            score = pos_score
            mask = pos_mask
            label = 1.0
        else:
            score = 1 - neg_score
            mask = neg_mask
            label = 0.0
        testing_pred_scores.append(score)
        testing_pred_masks.append(mask)
        testing_pred_labels.append(label)

    return np.array(training_pred_labels), np.array(testing_pred_labels), np.array(train_labels), np.array(test_labels), np.array(training_pred_scores), np.array(training_pred_masks), np.array(testing_pred_scores), np.array(testing_pred_masks) 