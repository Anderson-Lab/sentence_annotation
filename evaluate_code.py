import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse import hstack
from sklearn.naive_bayes import MultinomialNB
import copy
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import make_scorer

def convert_to_df(annots):
    sentences = []
    masks = []
    for annotation in annots:
        sentence = " ".join(annotation['words'])
        mask = " ".join([str(int(v)) for v in annotation['mask']])
        masks.append(mask)
        sentences.append(sentence)
    return pd.DataFrame({"sentence":sentences,"mask":masks})

class ItemSelector(BaseEstimator, TransformerMixin):
    """
    Parameters
    ----------
    key : hashable, required
        The key corresponding to the desired value in a mappable.
    """
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, df):
        return list(df[self.key])
    
class PredictorPipelineSelector(BaseEstimator, TransformerMixin):
    """
    Parameters
    ----------
    key : hashable, required
        The key corresponding to the desired value in a mappable.
    """
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def fit(self, x, y=None):
        return self

    def transform(self, df):
        return pd.DataFrame(self.pipeline.predict_proba(df)).values


class NotItemSelector(BaseEstimator, TransformerMixin):
    """
    Parameters
    ----------
    key : hashable, required
        The key corresponding to the desired value in a mappable.
    """
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, df):
        #import pdb
        #pdb.set_trace()
        return df.drop(self.key, axis=1).values
    
class Debug(BaseEstimator, TransformerMixin):

    def transform(self, X):
        print(X.shape)
        print(type(X))
        return X

    def fit(self, X, y=None, **fit_params):
        return self

def create_pipeline_bow_rf():
    clf = RandomForestClassifier(random_state=0)
    pipeline = Pipeline([
        # Use FeatureUnion to combine the features from subject and body
        ('union', FeatureUnion(
            transformer_list=[
                # Pipeline for standard bag-of-words model for body
                ('sale_description_bow', Pipeline([
                    ('selector', ItemSelector(key='sentence')),
                    ('tfidf', TfidfVectorizer())
                    #,
                    #('best', TruncatedSVD(n_components=50)),
                ]))
                #,

                # Pipeline for pulling ad hoc features from post's body
                #('other_features', Pipeline([
                #    ('selector', NotItemSelector(key='sentence'))
                #]))

            ],
        )),
        
        #('dbg',Debug()),

        # Use a SVC classifier on the combined features
        ('clf', clf)
    ])
    return pipeline

def evaluate_method(pipeline,training_data_dir="training_data",scoring='f1_macro'):
    np.random.seed(0) # Set the random seed
    labels = open(training_data_dir+"/labels","r").read().split("\n")
    iterations = 20
    eval_results = {}
    for label in labels:
        print("Evaluating",label)
        positive_annots = json.loads(open(training_data_dir+"/"+label+"_positive.txt").read())
        negative_annots = json.loads(open(training_data_dir+"/"+label+"_negative.txt").read())
        
        # balance the dataset
        if len(positive_annots) != len(negative_annots):
            n = min([len(positive_annots),len(negative_annots)])
            pos_inxs = np.arange(len(positive_annots))
            np.random.shuffle(pos_inxs)
            positive_annots = np.array(positive_annots)[pos_inxs[0:n]].tolist()
            neg_inxs = np.arange(len(negative_annots))
            np.random.shuffle(neg_inxs)
            negative_annots = np.array(negative_annots)[neg_inxs[0:n]].tolist()
        
        positive_df = convert_to_df(positive_annots)
        negative_df = convert_to_df(negative_annots)
        if positive_df.shape[0] < 10 or negative_df.shape[0] < 10:
            continue
        
        df_for_ml = positive_df.append(negative_df)
        
        targets = -1*np.ones((df_for_ml.shape[0],))
        targets[0:positive_df.shape[0]] = 1
                    
        if len(positive_annots) < 10:
            continue
        training_sizes = np.array(list(range(len(positive_annots)-1)),dtype=int)+1
        eval_results[label] = {}
        for training_size in training_sizes:
            print(df_for_ml.shape[0],2*training_size)
            cv = StratifiedShuffleSplit(n_splits=iterations, train_size=2*training_size, test_size=None)
            eval_results[label][training_size] = cross_val_score(pipeline, df_for_ml, targets, cv=cv, scoring=scoring)

    return eval_results

def old_evaluate_method(method_func,alignment_flag=False):
    np.random.seed(0) # Set the random seed
    labels = open("training_data/labels","r").read().split("\n")
    iterations = 20
    prediction_record = {}
    global w2v, w2v_tweets
    for label in labels:
        print("Evaluating",label)
        positive_annots = json.loads(open("training_data/"+label+"_positive.txt").read())
        negative_annots = json.loads(open("training_data/"+label+"_negative.txt").read())
        if len(positive_annots) != len(negative_annots):
            n = min([len(positive_annots),len(negative_annots)])
            pos_inxs = np.arange(len(positive_annots))
            np.random.shuffle(pos_inxs)
            positive_annots = np.array(positive_annots)[pos_inxs[0:n]].tolist()
            neg_inxs = np.arange(len(negative_annots))
            np.random.shuffle(neg_inxs)
            negative_annots = np.array(negative_annots)[neg_inxs[0:n]].tolist()
            
        if len(positive_annots) < 10:
            continue
        training_sizes = np.array(list(range(len(positive_annots)-1)))+1
        prediction_record[label] = {}
        for training_size in training_sizes:
            prediction_record[label][training_size] = {}
            for it in range(iterations):
                prediction_record[label][training_size][it] = {}
                all_inxs = np.arange(len(positive_annots))
                np.random.shuffle(all_inxs)

                # Select which positive annotations are training and testing
                training_inxs = all_inxs[0:training_size]
                testing_inxs = all_inxs[training_size:len(positive_annots)]

                positive_annots_array = np.array(positive_annots)
                training_positive_annots = positive_annots_array[training_inxs].tolist()
                testing_positive_annots = positive_annots_array[testing_inxs].tolist()

                # Select which negative annotations are training and testing
                neg_all_inxs = np.arange(len(negative_annots))
                np.random.shuffle(neg_all_inxs)

                neg_training_inxs = neg_all_inxs[0:training_size]
                neg_testing_inxs = all_inxs[training_size:len(positive_annots)] # keep the classes balanced by going with the positive annotions length
                negative_annots_array = np.array(negative_annots)
                training_negative_annots = negative_annots_array[neg_training_inxs].tolist()
                testing_negative_annots = negative_annots_array[neg_testing_inxs].tolist()            

                if not alignment_flag:
                    training_pred_labels,testing_pred_labels,training_labels,testing_labels = method_func(training_positive_annots,training_negative_annots,testing_positive_annots,testing_negative_annots)
                else:
                    training_pred_labels,testing_pred_labels,training_labels,testing_labels,training_pred_scores,training_pred_masks, testing_pred_scores, testing_pred_masks = method_func(training_positive_annots,training_negative_annots,testing_positive_annots,testing_negative_annots)
                    
                prediction_record[label][training_size][it]["training_pred_labels"] = training_pred_labels
                prediction_record[label][training_size][it]["testing_pred_labels"] = testing_pred_labels
                prediction_record[label][training_size][it]["training_labels"] = training_labels
                prediction_record[label][training_size][it]["testing_labels"] = testing_labels
                if alignment_flag:
                    prediction_record[label][training_size][it]["training_pred_scores"] = training_pred_scores
                    prediction_record[label][training_size][it]["training_pred_masks"] = training_pred_masks
                    prediction_record[label][training_size][it]["testing_pred_scores"] = testing_pred_scores
                    prediction_record[label][training_size][it]["testing_pred_masks"] = testing_pred_masks
    return prediction_record