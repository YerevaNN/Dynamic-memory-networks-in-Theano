#!/bin/bash

url=http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz
fname=`basename $url`

curl -SLO $url
tar zxvf $fname 
mkdir -p data
mv tasks_1-20_v1-2/* data/
rm -r tasks_1-20_v1-2

cat data/shuffled/*_train.txt > data/en/all_shuffled_train.txt
cat data/shuffled/*_test.txt > data/en/all_shuffled_test.txt