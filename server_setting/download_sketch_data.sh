#!/bin/zsh

# download data (path: /data/ephemeral/home/)
wget -P /data/ephemeral/home https://aistages-api-public-prod.s3.amazonaws.com/app/Competitions/000307/data/data.tar.gz

# unzip data.tar.gz
tar -xvzf /data/ephemeral/home/data.tar.gz

# remove "._ ~ "  data (이상한 이름의 데이터 제거)
find /data/ephemeral/home/data/ -name "._*" -type f -delete 