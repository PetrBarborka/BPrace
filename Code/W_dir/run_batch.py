

import argparse
import os, sys
import re
import json

import subprocess

"""Crawl specified config folder for
	pics jsons and outconf jsons in format
	"pics_name.json" and "outconf_name.json"
	as create_configs.py generates them.
	run them all with specified method config
	(argument 2) """

def addFile(file_dict, f, split, subkey):
	key = f.split(split)[1].split(".")[0]
	if key:
		if key in file_dict.keys():
			file_dict[key][subkey] = f
		else:
			file_dict[key] = {subkey: f}

def runConf(file_dict, config, key):
	bashCommand = r'./BP --cjsn ' + config + ' --pjsn ' + file_dict[key]["pics"] + ' --out ' + file_dict[key]["out"]
	print "running ", bashCommand
	subprocess.Popen(bashCommand.split())

descstr = 'Crawl specified config folder for \
	pics jsons and outconf jsons in format \
	"pics_name.json" and "outconf_name.json" \
 	as create_configs.py generates them.  \
	run them all with specified method config \
	(argument 2) '

parser = argparse.ArgumentParser(description='Short: Run BP app given a folder with configs\n\nLong: ' + descstr)
parser.add_argument('cfolder', type=str,
					help='folder containing pics and outconfs')
parser.add_argument('config', type=str,
					help='global config file containing method selection etc. One for all in folder')

args = parser.parse_args()

confs = {}
for root, dirs, files in os.walk(args.cfolder):

	re_pics = re.compile(ur"pics_\w+\.json")
	re_outconf = re.compile(ur"outconf_\w+\.json")

	pcs = []
	ocnf = []
	for f in files:
		key = ""
		if re_pics.match(f):
			# pcs.append(f)
			# key = f.split("pics_")[1]
			# print "key: ", key
			f = root + f
			addFile(confs, f, "pics_", "pics")

		elif re_outconf.match(f):
			ocnf.append(f)
			f = root + f
			addFile(confs, f, "outconf_", "out")

# print confs

keys = confs.keys()

for k in keys:
	runConf(confs, args.config, k)

sys.exit()
