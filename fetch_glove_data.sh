#!/bin/bash

url=http://nlp.stanford.edu/data/glove.6B.zip
fname=`basename $url`

curl -SLO $url
mkdir -p data
unzip $fname -d data/glove/
