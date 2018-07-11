import numpy as np
import os
import pickle
import warnings

import sys
sys.path.append("..")

import sentence_annotation.evaluate as evaluate
import sentence_annotation.novel_methods as novel_methods
import sentence_annotation.standard_methods as standard_methods
import sentence_annotation.alignment as alignment
import sentence_annotation.word2vec as word2vec
import sentence_annotation.word2vecReader as word2vecReader

methods = []
for i in range(1,len(sys.argv)):
    method = sys.argv[i]
    methods.append(method)

if len(methods) == 0:
    print("Usage: python3.6 eval_methods.py <list of methods>")
    print("Methods:","max","mean","alignment_v1","alignment_threshold","standard")
    print("\nIn order to run the methods that depend on word2vec, you must download the pre-trained word2vec model, which is available for download at: http://yuca.test.iminds.be:8900/fgodin/downloads/word2vec_twitter_model.tar.gz")
else:
    # Only load w2v model when we need it
    if "max" in methods or "mean" in methods or "alignment_v1" in methods or "alignment_threshold" in methods:
        w2v = word2vecReader.Word2Vec.load_word2vec_format("/dev/shm/word2vec_twitter_model.bin",binary=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for method in methods:
            print(method)
            if method == "max":
                eval_results_max = evaluate.evaluate_method(novel_methods.create_pipeline(w2v,novel_methods.get_max_distance_metric),training_data_dir="training_data")
                open("results/tweets/eval_results_max.pkl","wb").write(pickle.dumps(eval_results_max))
            elif method == "mean":
                eval_results_mean = evaluate.evaluate_method(novel_methods.create_pipeline(w2v,novel_methods.get_mean_distance_metric),training_data_dir="training_data")
                open("results/tweets/eval_results_mean.pkl","wb").write(pickle.dumps(eval_results_mean))
            elif method == "alignment_v1":
                alignment_method = alignment.Alignment(w2v)
                eval_results_alignment_v1 = evaluate.evaluate_method(
                    novel_methods.create_pipeline(alignment_method,novel_methods.get_alignment_distance_metric),
                    training_data_dir="training_data")
                open("results/tweets/eval_results_alignment_v1.pkl","wb").write(pickle.dumps(eval_results_alignment_v1))
            elif method == "alignment_threshold":
                alignment_method = alignment.Alignment(w2v)
                eval_results_alignment_threshold = evaluate.evaluate_method(
                    novel_methods.create_pipeline_alignment_threshold(alignment_method),training_data_dir="training_data")
                open("results/tweets/eval_results_alignment_threshold.pkl","wb").write(pickle.dumps(eval_results_alignment_threshold))
            elif method == "standard":
                eval_results_nb = evaluate.evaluate_method(standard_methods.create_pipeline_bow_nb(),training_data_dir="training_data")
                open("results/tweets/eval_results_nb.pkl","wb").write(pickle.dumps(eval_results_nb))
                eval_results_svm = evaluate.evaluate_method(standard_methods.create_pipeline_bow_svm(),training_data_dir="training_data")
                open("results/tweets/eval_results_svm.pkl","wb").write(pickle.dumps(eval_results_svm))
                eval_results_rf = evaluate.evaluate_method(standard_methods.create_pipeline_bow_rf(),training_data_dir="training_data")
                open("results/tweets/eval_results_rf.pkl","wb").write(pickle.dumps(eval_results_rf))
