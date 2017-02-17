from 	ubuntu:16.04

run	    apt-get -yqq update
run	    apt-get install -yqq python-dev python-pip python-nose g++ libopenblas-dev python-numpy python-scipy

add	    . /root/sequence-based-recommendations
workdir /root/sequence-based-recommendations

run     pip install --upgrade pip
run     pip install -r requirements.txt


run     pip install -r https://raw.githubusercontent.com/Lasagne/Lasagne/master/requirements.txt && \
        pip install https://github.com/Lasagne/Lasagne/archive/master.zip