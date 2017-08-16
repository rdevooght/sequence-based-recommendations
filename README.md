# Collaborative filtering based on sequences
This python library includes multiple collaboraborative filtering algorithm that make use of the sequence of actions of the user: they not only use the fact that a user rated a certain item, but also that he rated before this other item or after that other one.
Some standard algorithms that do not use sequence information are also present for easier comparison.

All those algorithms aims to solve the "item recommendation" or "top-N recommendation" problem, which mean that they are not designed to predict ratings values, but only to predict which items are of interest for a given user.

Our code was used to produce the experiments in "[Collaborative Filtering with Recurrent Neural Networks](https://arxiv.org/abs/1608.07400)" and "[Long and Short-Term Recommendations with Recurrent
Neural Networks](http://iridia.ulb.ac.be/~rdevooght/papers/UMAP__Long_and_short_term_with_RNN.pdf)". 
If you use this code in your research, please cite us:
````
@inproceedings{Rec_with_RNN,
 author = {Devooght, Robin and Bersini, Hugues},
 title = {Long and Short-Term Recommendations with Recurrent Neural Networks},
 booktitle = {Proceedings of the 25th Conference on User Modeling, Adaptation and Personalization},
 series = {UMAP '17},
 year = {2017},
 isbn = {978-1-4503-4635-1},
 location = {Bratislava, Slovakia},
 pages = {13--21},
 numpages = {9},
 url = {http://doi.acm.org/10.1145/3079628.3079670},
 doi = {10.1145/3079628.3079670},
 acmid = {3079670},
 publisher = {ACM},
} 
````

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
`--metrics value` | Metrics computed on the validation set, separated by commas. Available metrics are recall, sps, ndcg, item\_coverage, user\_coverage and blockbuster\_share. Default: sps.
`--save [All, Best, None]` | Policy for saving models. If "None", no model is saved. If "All", the current model is saved each time the model is evaluated on the validation set, and no model is destroyed. If "Best", the current model is only saved if it improves over the previous best results on the validation set, and the previous best model is deleted. If "Best" and multiple metrics are used, all the pareto-optimal models are saved. 
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

`--metrics` allows to specify the list of metrics to compute, separated by commas. By default the metrics are: sps, recall, item\_coverage, user\_coverage, blockbuster_share.
The "blockbuster share" is the percentage of correct recommendations among the 1% most popular items.
The other available metrics are the sps, the ndcg and the assr (when clustering is used).

All the metrics are computed "@k", with k=10 by default. k can be changed using the `-k` option.

When the `--save` option is used, the results are saved in a file in "path/to/dataset/results/".
the results of each model form a line of the file, and each line contains the number of epochs followed by the metrics specified by `--metrics`.

When testing a method based on clustering, the option `--ignore_clusters` can be used to test how the method performs without clusters.

## Methods

The available methods are:
* [Recurrent Neural Networks](#recurrent-neural-networks)
* [Stacked Denoising Autoencoder](#stacked-denoising-autoencoders)
* [Latent Tarjectory Modeling/word2vec](#latent-trajectory-modeling)
* [BPR-MF](#bpr-mf)
* [FPMC](#fpmc)
* [FISM](#fism)
* [Fossil](#fossil)
* [Markov Chains](#markov-chain)
* [User KNN](#user-knn)
* [Popularity baseline](#pop)

### Neural Networks
#### Recurrent Neural Networks

Use it with `-m RNN`.
The RNN have many options allowing to change the type/size/number of layers, the training procedure and the objective function, and some options are specific to a particular objective function.

##### Layers

Option | Desciption
------ | ----------
`--r_t [LSTM, GRU, Vanilla]` | Type of recurrent layer (default is GRU)
`--r_l size_of_layer1-size_of_layer2-etc.` | Size and number of layers. for example, `--r_l 100-50-50` creates a layer with 50 hidden neurons on top of another layer with 50 hidden neurons on top of a layer with 100 hidden neurons. Default: 32.
`--r_bi` | Use bidirectional layers.
`--r_emb size` | Adds an embedding layer before the recurrent layer. By default no embedding layer is used, but it is adviced to use one (e.g. `--r_emb 100`).

##### Update mechanism

Option | Desciption
------ | ----------
`--u_m [adagrad, adadelta, rmsprop, nesterov, adam]` | Update mechanism (see [Lasagne doc](http://lasagne.readthedocs.io/en/latest/modules/updates.html)). Default is adam
`--u_l float` | Learning rate (default: 0.001). The default learning rate works well with adam. For adagrad `--u_l 0.1` is adviced.
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

##### Clustering

It is possible to combine RNNs with an item-clustering method. This leads to faster prediction on large dataset and creates meaningful item clusters.
In order to use it, use the option `--clusters nb_of_clusters`.  
For example, `python train.py -d path/to/dataset/ -m RNN --loss BPR --clusters 10` will train an RNN with the BPR loss and 10 clusters of items.
Note that the clustering is only compatible with sampling-based loss (BPR, Blackout and TOP1). 
It also works with `--loss CCE`, but a sampling version of CCE is then used instead of the normal categorical cross-entropy.
	

#### Stacked Denoising Autoencoders

Use it with `-m SDAE`.
SDAE the RNN options described in "[Update mechanism](#update-mechanism)" and "[Other options](#other-options)".

Option | Desciption
------ | ----------
`--L size_of_layer1-size_of_layer2-etc.` | Size and number of layers. for example, `--r_l 50-32-50` creates a layer with 50 hidden neurons on top of another layer with 32 hidden neurons on top of a layer with 50 hidden neurons. Default: 20.
`--in_do float` | Dropout rate applied to the input layer of the SDAE (default: 0.2).
`--do float` | Dropout rate applied to the hidden layers of the SDAE (default: 0.5).

#### Latent Trajectory Modeling

Use it with `-m  LTM`.
LTM is a method based on word2vec, described in "[Latent Trajectory Modeling: A Light and Efficient Way to Introduce Time in Recommender Systems](http://dl.acm.org/citation.cfm?id=2799676)".
LTM works in two steps: it first produces an embedding of the items with the word2vec algorithm using the sequence of items in the training set, then it estimates for each user a translation vector that would best explain the trajectory of that user in the embedded space.
Predictions are made by finding the closest items to the last user item translated by the user's translation vector.
Our implementation is mainly a wrapper around [Gensim's word2vec implementation](https://radimrehurek.com/gensim/models/word2vec.html).

Option | Desciption
------ | ----------
`-H int` | Number of neurons (default: 20).
`--ltm_window int` | Size of word2vec's window (default: 5).
`--ltm_damping float` | Temporal damping (default: 0.8).
`--ltm_no_trajectory` | Use this option to make predictions directly with word2vec, without the trajectory estimation proposed in the LTM paper.

### Factorization-based
#### FPMC

FPMC is a method combining factorized markov chains with the factorization of the user-item matrix (see "Factorizing personalized Markov chains for next-basket recommendation" by Rendle et al. in *Proceedings of WWW'10*).
Use it with `-m FPMC`

Option | Desciption
------ | ----------
`--k_cf int` | Rank of the user-item matrix factorization (default: 32).
`--k_mc int` | Rank of the factorized Markov chain (default: 32).
`-l val` | Learning rate (default: 0.01).
`--cooling val` | Multiplicative factor applied to the learning rate after each epoch (default: 1)
`--init_sigma val` | Standard deviation of the gaussian initialization (default: 1).
`--fpmc_bias val` | Sampling bias (default: 100). By default the SGD process uses adaptive sampling to speed up learning. This parameter is used to control how much the sampling is biased towards high error items.
`--no_adaptive_sampling` | No adaptive sampling
`-r float` | Add a regularization term. A positive value will use L2 regularization and a negative value will use L1. Default: 0.

#### BPR-MF

BPR-MF is a matrix factorization method based on the BPR loss (see "BPR: Bayesian personalized ranking from implicit feedback" by Rendle et al. in *Proceedings of the twenty-fifth conference on uncertainty in artificial intelligence*)
Use it with `-m BPRMF`

Option | Desciption
------ | ----------
`-H int` | Rank of the user-item matrix factorization (default: 20).
`-l val` | Learning rate (default: 0.01).
`--cooling val` | Multiplicative factor applied to the learning rate after each epoch (default: 1)
`--init_sigma val` | Standard deviation of the gaussian initialization (default: 1).
`--fpmc_bias val` | Sampling bias (default: 100). By default the SGD process uses adaptive sampling to speed up learning. This parameter is used to control how much the sampling is biased towards high error items.
`--no_adaptive_sampling` | No adaptive sampling
`-r float` | Add a regularization term. A positive value will use L2 regularization and a negative value will use L1. Default: 0.

#### FISM

FISM is a method based of item-item factorization (see "Fism: factored item similarity models for top-n recommender systems" by Kabbur et al. in *Proceedings of SIGKDD'13*).
It has the advantage over BPR-MF that it does not build a representation for each user. This leads to smaller models, and the ability to make recommendation to new users.
Use it with `-m FISM --loss [BPR, RMSE]`

Option | Desciption
------ | ----------
`--loss [BPR, RMSE]` | Loss function. "BPR" is the same loss as for BPR-MF, "RMSE" optimizes the square error. This cannot be left to default because the default loss is CCE, which is not compatible with FISM.
`-H int` | Rank of the matrix factorization (default: 20).
`--fism_alpha float` | Alpha parameter in FISM. (default: 0.2).
`-l val` | Learning rate (default: 0.01).
`--cooling val` | Multiplicative factor applied to the learning rate after each epoch (default: 1)
`--init_sigma val` | Standard deviation of the gaussian initialization (default: 1).
`-r float` | Add a regularization term. A positive value will use L2 regularization and a negative value will use L1. Default: 0.

FISM can be combined with item-clustering the same way that RNN can.
To do so, add the option `--clusters nb_of_clusters`.
When using clustering, a completely different implementation is used, which is based on Theano instead of Numpy.
This has some implications on the available options:
* The loss must be choosen among CCE, BPR, Blackout and TOP1 instead of BPR and RMSE.
* The number of samples for each training step can be specified using `--sampling nb_of_samples`.
* The update mechanism is controled by the options defined in [Update mechanism](#update-mechanism) instead of `-l` and `--cooling`.

#### Fossil

Fossil combines FISM with factorized markov chains (see "Fusing Similarity Models with Markov Chains for Sparse Sequential Recommendation" by He and McAuley in *Proceedings of ICDM'16*).
Unlike FPMC, Fossil can use higher-order markov chains.
Use it with `-m Fossil`

Option | Desciption
------ | ----------
`-H int` | Rank of the matrix factorization (default: 20).
`--fism_alpha float` | Alpha parameter in FISM. (default: 0.2).
`--fossil_order int` | Order of the markov chains in Fossil. (default: 1).
`-l val` | Learning rate (default: 0.01).
`--cooling val` | Multiplicative factor applied to the learning rate after each epoch (default: 1)
`--init_sigma val` | Standard deviation of the gaussian initialization (default: 1).
`-r float` | Add a regularization term. A positive value will use L2 regularization and a negative value will use L1. Default: 0.

### Lazy

Lazy methods do not build models, they make recommendation directly based on the dataset.
They should therefor not be used with `train.py`, but only with `test.py`.

#### POP

Use it with `-m POP`.
Always predict the most popular items.

#### Markov Chain

Use it with `-m MM`.
Recommends the items that follow most often the last item the user's sequence.

#### User KNN

Use it with `-m UKNN`.
User-based nearest neighbors approach. 
The similarity measure between users is the cosine similarity: #number-of-common-items / sqrt(#number-of-items-of-user-a * #number-of-items-of-user-b).

Option | Desciption
------ | ----------
`--ns int` | Neighborhood size (default: 80).
	
