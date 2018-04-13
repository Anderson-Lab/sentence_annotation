import json
import numpy as np

def evaluate_method(method_func,alignment_flag=False):
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