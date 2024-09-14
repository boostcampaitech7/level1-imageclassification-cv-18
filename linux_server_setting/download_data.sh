#!/bin/zsh

wget -P /data/ephemeral/home https://aistages-api-public-prod.s3.amazonaws.com/app/Competitions/000307/data/data.tar.gz

tar -xvzf /data/ephemeral/home/data.tar.gz

find /data/ephemeral/home/data/ -name "._*" -type f -delete 