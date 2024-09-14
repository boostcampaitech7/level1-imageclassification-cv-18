#!/bin/bash

cd ~

apt-get update -y
apt-get install zsh -y
zsh --version
which zsh

chsh -s $(which zsh)