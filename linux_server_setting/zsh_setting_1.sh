#!/bin/zsh

cd ~

apt-get update -y
apt-get install curl -y
apt-get install wget -y

apt install git -y
git --version 

sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
