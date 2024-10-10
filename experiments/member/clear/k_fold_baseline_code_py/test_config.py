import argparse
import configparser

config = configparser.ConfigParser()
config.read('./config.ini')
defaults = config['default']

parser = argparse.ArgumentParser()
a = parser.parse_args()
a = argparse.Namespace(**dict(defaults))