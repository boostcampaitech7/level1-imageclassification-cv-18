#!/bin/zsh

cd ~

apt-get update -y
apt-get install curl -y
apt-get install wget -y
apt-get install libglib2.0-0

apt install git -y
git --version 

sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
