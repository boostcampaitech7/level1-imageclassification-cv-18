#!/bin/bash

cd ~

# install zsh shell
apt-get update -y
apt-get install zsh -y
zsh --version
which zsh

# set default zsh shell
chsh -s $(which zsh)