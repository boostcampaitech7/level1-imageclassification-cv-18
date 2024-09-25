import argparse
import configparser

config = configparser.ConfigParser()
config.read('D:\\clear\\level1-imageclassification-cv-18\\clear\\k_fold_baseline_code_py\\config.ini')
defaults = config['default']

parser = argparse.ArgumentParser()
a = parser.parse_args()
a = argparse.Namespace(**dict(defaults))