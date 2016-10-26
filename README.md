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
sudo pip install Theano pandas gensim Lasagne>=0.2.dev1
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

### test.py

## Methods
