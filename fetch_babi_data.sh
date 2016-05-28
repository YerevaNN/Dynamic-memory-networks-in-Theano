#!/bin/bash

url=http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz
fname=`basename $url`

curl -SLO $url
tar zxvf $fname 
mkdir -p data
mv tasks_1-20_v1-2/* data/
rm -r tasks_1-20_v1-2

tar zxvf babi_all_shuffles.tar.gz
