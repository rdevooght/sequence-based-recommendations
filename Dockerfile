from 	ubuntu:16.04

run	    apt-get -yqq update
run	    apt-get install -yqq python-dev python-pip python-nose g++ libopenblas-dev python-numpy python-scipy

add	    . /root/sequence-based-recommendations
workdir /root/sequence-based-recommendations

run     pip install --upgrade pip
run     pip install -r requirements.txt


run     pip install -r https://raw.githubusercontent.com/Lasagne/Lasagne/master/requirements.txt && \
        pip install https://github.com/Lasagne/Lasagne/archive/master.zip

run     apt-get install -y unzip curl
run     curl -o ml-1m.zip http://files.grouplens.org/datasets/movielens/ml-1m.zip
run     unzip ml-1m.zip
run     rm ml-1m.zip

run     yes | python preprocess.py -f ml-1m/ratings.dat --columns uirt --sep :: --val_size 100 --test_size 100     
