#!/bin/zsh

cd ~

# install basic function
apt-get update -y
apt-get install curl -y
apt-get install wget -y
apt-get install nano -y
apt-get install libglib2.0-0 -y

# install git
apt install git -y
git --version 

# install oh-my-zsh
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

