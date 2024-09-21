import configparser
import argparse

config = configparser.ConfigParser()
config.read('./config.ini')
defaults = config['default']

parser = argparse.ArgumentParser()
a = parser.parse_args()
a = argparse.Namespace(**dict(defaults))
# print(type(a))
# print(dict(defaults))
print(a)