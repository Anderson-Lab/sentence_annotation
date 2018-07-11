from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

def str_to_ord(string,n=3):
    new_str_arr = []
    for ch in string:
        addition = str(ord(ch))
        if len(addition) < n:
            for i in range(n-len(addition)):
                addition = "0" + addition
        elif len(addition) > n:
            raise Exception("n is not large enough")
        new_str_arr.append(addition)
    return int("".join(new_str_arr))

def ord_to_str(ordinal,n=3):
    line = str(ordinal)
    try:
        characters = [chr(int(line[i:i+n])) for i in range(0, len(line), n)]
    except:
        import pdb
        pdb.set_trace()
    return "".join(characters)

class AlignmentSelector(BaseEstimator, TransformerMixin):
    """
    Parameters
    ----------
    key : hashable, required
        The key corresponding to the desired value in a mappable.
    mask_key: hashable, required
        The key corresponding to the mask.
    """
    def __init__(self, key, mask_key, storage_func, mask_func):
        self.key = key
        self.mask_key = mask_key
        self.storage_func = storage_func
        self.mask_func = mask_func
        
    def apply_mask(self,sentence,mask):
        mask = np.array(mask.split(" ")) == '1'
        masked_words = np.array(sentence.split(" "))[mask]
        return masked_words
        
    def fit(self, x, y=None):
        return self

    def transform(self, df):
        res = list(df[self.key])
        masks = list(df[self.mask_key])
        for i in range(len(res)):
            orig = res[i]
            res[i] = len(self.storage_func())
            self.storage_func().append(orig.split(" "))
            self.mask_func().append(self.apply_mask(orig,masks[i]))

        return np.reshape(res,(-1, 1))

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