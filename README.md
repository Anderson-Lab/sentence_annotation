# Alignment-Based Topic Extraction Using Word Embedding

This work introduces a new method for predicting topics within textual data that minimizes the dependence upon a large, properly-labelled training set. By using word vectors trained using unlabeled data, our method allows us to reduce the amount of human labor that must be done in the text annotation process.

## Getting Started

If you would like to explore or test our algorithm, this repository contains all the necessary code to do so.

### Prerequisites

* [*Python 3*](https://www.python.org) - In order to run our code, you will need to have Python 3 installed on your machine.
* [*Jupyter Notebooks*](https://jupyter.org) - We use Python Jupyter Notebooks for most of our testing and also for our annotator widget. In order to use these you will need to also have Jupyter installed.
* [*Pandas*](https://pandas.pydata.org) - Pandas is a library that we use to structure the data for annotation, so you must have it installed for the annotation interface to work.

### Installing

Once you have cloned this repository onto your machine, you can use Jupyter to explore the notebooks we have already created or you can create your own.

With Jupyter installed, you can run this simple command in your terminal/command prompt:
```
jupyter notebook
```
This will start up the Jupyter server/interface. From there, you can navigate to the folder that contains this project.

If you open up the *Tweet.ipynb* file, you can see how the annotation interface for our Twitter data is run, which can be seen with the following code:
```
ORIGINAL_FILE_NAME = 'training_data/sources/tweets/tweets-3461.txt'
MASK_FILE_NAME = 'tweets/mask_df.pickle'
run_annotator(ORIGINAL_FILE_NAME, MASK_FILE_NAME)
```
where `ORIGINAL_FILE_NAME` is the name of the file that contains the original tweet data and `MASK_FILE_NAME` is the name of the file that contains the serialized annotation data.

## Training and Testing

We have created a wrapper Python script called *eval_methods.py* that provides a standardized method for training and evaluating the algorithms.

## Visualizing Results

We have created a Jupyter notebook to show you can demonstrate the results called *Analyze_Methods.ipynb*.

## Built With

* [Python](https://www.python.org) - Programming Language
* [Jupyter](https://jupyter.org) - Development and Testing
* [Pandas](https://pandas.pydata.org) - Data Structuring

## Authors

* **Tyler Newman**
* **Paul Anderson**

See also the list of [contributors](https://github.com/Anderson-Lab/sentence-annotation/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE) file for details
