from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np

def bag_of_words_nb(training_positive_annots,training_negative_annots,test_positive_annots,test_negative_annots):
    train_positive_data = [" ".join(entry["words"]) for entry in training_positive_annots]
    train_negative_data = [" ".join(entry["words"]) for entry in training_negative_annots]
    test_positive_data = [" ".join(entry["words"]) for entry in test_positive_annots]
    test_negative_data = [" ".join(entry["words"]) for entry in test_negative_annots]
    
    train_data = train_positive_data + train_negative_data
    
    train_positive_labels = np.ones(len(train_positive_data),).tolist()
    train_negative_labels = np.zeros(len(train_negative_data),).tolist()
    train_labels = train_positive_labels + train_negative_labels
    test_positive_labels = np.ones(len(test_positive_data),).tolist()
    test_negative_labels = np.zeros(len(test_negative_data),).tolist()
    test_labels = test_positive_labels + test_negative_labels
    
    #set_trace() # debugging starts here
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(train_data)
    tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
    X_train_tfidf = tf_transformer.transform(X_train_counts)
    clf = MultinomialNB().fit(X_train_tfidf, train_labels)
    
    X_test_counts = count_vect.transform(test_positive_data+test_negative_data)
    X_test_tfidf = tf_transformer.transform(X_test_counts)
    
    testing_pred_labels = clf.predict(X_test_tfidf)
    training_pred_labels = clf.predict(X_train_tfidf)
    
    return training_pred_labels, testing_pred_labels, train_labels, test_labels

# positive class = 1
# negative class = 0
def guess_positive(training_positive_annots,training_negative_annots,testing_positive_annots,testing_negative_annots):
    training_pred_labels = np.ones((len(training_positive_annots)+len(training_negative_annots)),)
    testing_pred_labels = np.ones((len(testing_positive_annots)+len(testing_negative_annots)),)
    return training_pred_labels, testing_pred_labels

from sklearn.linear_model import SGDClassifier

def svm_v1(training_positive_annots,training_negative_annots,test_positive_annots,test_negative_annots):
    train_positive_data = [" ".join(entry["words"]) for entry in training_positive_annots]
    train_negative_data = [" ".join(entry["words"]) for entry in training_negative_annots]
    test_positive_data = [" ".join(entry["words"]) for entry in test_positive_annots]
    test_negative_data = [" ".join(entry["words"]) for entry in test_negative_annots]
    
    train_data = train_positive_data + train_negative_data
    
    train_positive_labels = np.ones(len(train_positive_data),).tolist()
    train_negative_labels = np.zeros(len(train_negative_data),).tolist()
    train_labels = train_positive_labels + train_negative_labels
    test_positive_labels = np.ones(len(test_positive_data),).tolist()
    test_negative_labels = np.zeros(len(test_negative_data),).tolist()
    test_labels = test_positive_labels + test_negative_labels
    
    clf = SGDClassifier(loss='hinge', penalty='l2',
                        alpha=1e-3, random_state=42,
                        max_iter=5, tol=None)
    
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(train_data)
    tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
    X_train_tfidf = tf_transformer.transform(X_train_counts)
    clf.fit(X_train_tfidf, train_labels)
    
    X_test_counts = count_vect.transform(test_positive_data+test_negative_data)
    X_test_tfidf = tf_transformer.transform(X_test_counts)
    
    testing_pred_labels = clf.predict(X_test_tfidf)
    training_pred_labels = clf.predict(X_train_tfidf)
    
    return training_pred_labels, testing_pred_labels, train_labels, test_labels