# Collaborative filtering based on sequences
This python library includes multiple collaboraborative filtering algorithm that make use of the sequence of actions of the user: they not only use the fact that a user rated a certain item, but also that he rated before this other item or after that other one.
Some standard algorithms that do not use sequence information are also present for easier comparison.

All those algorithms aims to solve the "item recommendation" or "top-N recommendation" problem, which mean that they are not designed to predict ratings values, but only to predict which items are of interest for a given user.

## Installation
The library has many dependencies: numpy/scipy, theano and lasagne for the neural networks, Gensim for word2vec and pandas for the data manipulation.

Numpy, scipy and Theano can be sometimes difficult to install, and we recommend looking at Theano's installation tutorial: http://deeplearning.net/software/theano/install.html
Gensim and pandas are easily installed with pip. Lasagne is also installed with pip but you have to specify the version >=0.2.dev1.

On Ubuntu, the following commands should install everything that you need:
````
sudo apt-get install python-numpy python-scipy python-dev python-pip python-nose g++ libopenblas-dev git
sudo pip install Theano pandas gensim https://github.com/Lasagne/Lasagne/archive/master.zip
````

## Usage
The library is designed to be used in command line through three scripts:
* preprocess.py for the preparation of the dataset
* train.py for training models
* test.py for testing models

calling these scripts with the `--help` option will display the available options (e.g. `python preprocess.py --help`). 

### preprocess.py

This script takes a file containing a dataset of user/item interactions and split it into training/validation/test sets and save them in the format used by train.py and test.py.
The original dataset must be in a format where each line correspond to a single user/item interaction.

The only required argument is `-f path/to/dataset`, which is used to specify the original dataset. The script will create subfolders named "data", "models" and "results" in the folder containing the original dataset. "data" is used by preprocess.py to store all the files it produces, "models" is used by train.py to store the trained models and "results" is used by test.py to store the results of the tests.

The optional arguments are the following:

Option | Desciption
------ | ----------
`--columns` | Order of the columns in the file (eg: "uirt"), u for user, i for item, t for timestamp, r for rating. If r is not present a default rating of 1 is given to all interaction. If t is not present interactions are assumed to be in chronological order. Extra columns are ignored. Default: uit
`--sep` | Separator between the column. If unspecified pandas will try to guess the separator
`--min_user_activity` | Users with less interactions than this will be removed from the dataset. Default: 2
`--min_item_pop` | Items with less interactions than this will be removed from the dataset. Default: 5
`--val_size` | Number of users to put in the validation set. If in (0,1) it will be interpreted as the fraction of total number of users. Default: 0.1
`--test_size` | Number of users to put in the test set. If in (0,1) it will be interpreted as the fraction of total number of users. Default: 0.1
`--seed` | Seed for the random train/val/test split

#### Example 1
In the movielens 1M dataset each line has the following format:
````
UserID::MovieID::Rating::Timestamp
````
To process it you have to specify the order of the columns, in this case uirt (for user, item, rating, timestamp), and the separator ("::"). If you want to use a hundred users for the validation set and a hundred others for the test set, you'll have to use the following command:
````
python preprocess.py -f path/to/ratings.dat --columns uirt --sep :: --val_size 100 --test_size 100
````
#### Example 2
Consider a dataset where each line has the following format:
````
timestamp, user_id, some_useless_data, item_id, more_useless_data
````
You can specify the order of columns with "tuxiy" where x and y are placeholder names for the columns that will be discarted by the script. Using "tuxi" will also work, as all the columns not mentioned are discarded. As no rating column is present, each interaction will recieve the rating "1". If you also want for example to remove users with less than 10 interactions, use the following command:
````
python preprocess.py -f path/to/file --columns tuxi --min_user_activity 10
````

### train.py

This script is used to train models and offers many options regarding when to save new models and when to stop training.
The basic usage is the following:
````
python train.py -d path/to/dataset/ -m Method_name
````

The argument `-d` is used to specify the path to the folder that contains the "data", "models" and "results" subfolders created by preprocess.py. 
If you have multiple datasets with a partly common path (e.g. path/to/dataset1/, path/to/dataset2/, etc.) you can specify this common path in the variable DEFAULT_DIR of helpers/data_handling.py. For example, setting DEFAULT_DIR = "path/to/" and using the argument `-d dataset1` will look for the dataset in "path/to/dataset1/".

The optional arguments are the following:
Option | Desciption
------ | ----------
`--dir dirname/` | Name of the subfolder of "path/to/dataset/models/" in which to save the model. By default it will be saved directly in the models/ folder, but using subfolders can be useful when many models are tested.
`--progress {int or float}` | Number of iterations (or seconds) between two evaluations of the model on the validation set. When the model is evaluated, progress is shown on the command line, and the model might be saved (depending on the `--save` option). An float value means that the evaluations happen at geometric intervals (rather than linear). Default: 2.0
`--save [All, Best, None]` | Policy for saving models. If "All", the current model is saved each time the model is evaluated on the validation set, and no model is destroyed. If "Best", the current model is only saved if it improves over the previous best results on the validation set, and the previous best model is deleted. If "None", no model is saved.
`--time_based_progress` | Base the interval between two evaluations on the number of elapsed seconds rather than on the number of iterations.
`--mpi value` | Max number of iterations (or seconds) between two evaluations (useful when using geometric intervals). Default: inf.
`--max_iter value` | Max number of iterations (default: inf).
`--max_time value` | Max training time in seconds (default: inf).
`--min_iter value` | Min number of iterations before making the first evaluation (default: 0).
`--extended_set` | Use extended training set (contains first half of validation and test set). This is necessary for factorization based methods such as BPRMF and FPMC because they need to build a model for every user.
`--tshuffle` | Shuffle the order of sequences between epochs.
`--load_last_model` | Load Last model before starting training (it will search for a model build with all the same options and take the one with the largest number of epochs).
`--es_m [WorstTimesX, StopAfterN, None]` | Early stopping method (by default none is used, and training continues until max_iter or max_time is reached). WorstTimesX will stop training if the number of iterations since the last best score on the validation set is longer than X times the longest time between two consecutive best scores. StopAfterN will stop the training if the model has not improved for the N last evaluations on the validation set.
`--es_n N` | N parameter for StopAfterN (default: 5).
`--es_x X` | X parameter for WorstTimesX (default: 2).
`--es_min_wait num_epochs` | Mininum number of epochs before stopping (for WorstTimesX). Default: 1.
`--es_LiB` | Lower is better for validation score. By default a higher validation score is considered better, but if it is not the case you can use this option.

The options specific to each method are explained in the Methods section.

### test.py

This script test the models built with train.py on the test set.
The basic usage is:
````
python test.py -d path/to/dataset/ -m Method_name
````
The argument `-d` works in the same way as with train.py, and the precise model to test is specified by the `--dir` option and the methods-specific options.
If multiple models fit the options (They are in the same subfolder and were trained with the same method and same options), they are all evaluated one after the other, except if the argument `-i epoch_number` is also specified, which will then select the model based on the number of epochs.

When the `--save` option is used, the results are saved in a file in "path/to/dataset/results/".
the results of each model form a line of the file, and the following metrics are saved (in this order):
1. Number of epochs
2. precision
3. sps
4. user coverage
7. Number of distinct items correctly recommended
10. recall
11. NDCG
12. Percentage of correct recommendations among the 1% most popular items

All the metrics are computed "@k", with k=10 by default. k can be changed using the `-k` option.

## Methods

### Neural Networks
#### Recurrent neural networks

Use it with `-m RNN`.
The RNN have many options allowing to change the type/size/number of layers, the training procedure and the objective function, and some options are specific to a particular objective function.

##### Layers

Option | Desciption
------ | ----------
`--r_t [LSTM, GRU]` | Type of recurrent layer (default is GRU)
`--r_l size_of_layer1-size_of_layer2-etc.` | Size and number of layers. for example, `--r_l 100-50-50` creates a layer with 50 hidden neurons on top of another layer with 50 hidden neurons on top of a layer with 100 hidden neurons. Default: 32.
`--r_bi` | Use bidirectional layers.

##### Update mechanism

Option | Desciption
------ | ----------
`--u_m [adagrad, adadelta, rmsprop, nesterov, adam]` | Update mechanism (see [Lasagne doc](http://lasagne.readthedocs.io/en/latest/modules/updates.html)). Default is adagrad
`--u_l float` | Learning rate (default: 0.1).
`--u_rho float` | rho parameter for Adadelta and RMSProp, or momentum for Nesterov momentum (default: 0.9).
`--u_b1 float` | Beta 1 parameter for Adam (default: 0.9).
`--u_b2 float` | Beta 2 parameter for Adam (default: 0.999).

##### Noise

Option | Desciption
------ | ----------
`--n_dropout P` | Dropout probability (default: 0.)
`--n_shuf P` | Probability that an item is swapped with another one (default: 0.).
`--n_shuf_std STD` | If an item is swapped, the position of the other item is drawn from a normal distribution whose std is defined by this parameter (default: 5.).

##### Other options

Option | Desciption
------ | ----------
`-b int` | Size of the mini-batchs (default: 16)
`--max_length int` | Maximum length of sequences (default: 200)
`-g val` | Gradient clipping (default: 100)
`--repeated_interactions` | Use when a user can interact multiple times with the same item. If not set, the items that the user already saw are never recommended.

##### Objective functions

Option | Desciption
------ | ----------
`--loss [CCE, Blackout, TOP1, BPR, hinge, logit, logsig]` | Objective function. CCE is the categorical cross-entropy, BPR, TOP1 and Blackout are based on sampling, and hinge, logit and logsig allow to have multiple targets. Default is CCE.
`-r float` | *Only for CCE*. Add a regularization term. A positive value will use L2 regularization and a negative value will use L1. Default: 0.
`--db float` | *Only for CCE, Blackout, BPR and TOP1*. Increase the diversity bias to put more pressure on learning correct recomendations for unfrequent items (default: 0.).
`--sampling float or int` | *Only for Blackout, BPR and TOP1*. Number of items to sample in the error computation. Use a float in [0,1] to express it as a fraction of the number of items in the catalog, or an int > 0 to specify the number of samples directly. Default: 32.
`--n_targets N` | *Only for hinge, logit and logsig*. Number of items in the sequence that are used as targets. Default: 1.


#### Stacked denoising auto-encoders

Use it with `-m SDAE`.
SDAE the RNN options described in "[Update mechanism](#update-mechanism)" and "[Other options](#other-options)".

Option | Desciption
------ | ----------
`--L size_of_layer1-size_of_layer2-etc.` | Size and number of layers. for example, `--r_l 50-32-50` creates a layer with 50 hidden neurons on top of another layer with 32 hidden neurons on top of a layer with 50 hidden neurons. Default: 20.
`--in_do float` | Dropout rate applied to the input layer of the SDAE (default: 0.2).
`--do float` | Dropout rate applied to the hidden layers of the SDAE (default: 0.5).

### Factorization-based
#### FPMC
#### BPR-MF
### Lazy
#### POP
#### Markov Chain
#### User KNN