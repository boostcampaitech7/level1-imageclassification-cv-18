#!/bin/zsh

cd ~

apt-get update

apt-get install curl

apt install git
git --version 

sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
