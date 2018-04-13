import word2vec
import numpy as np
import alignment

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