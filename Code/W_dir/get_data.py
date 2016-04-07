

"""Crawl given root for data.csv files from BP app.
	Create a complete dataset and subsets,save as .csv."""

import argparse
import os, sys
import re
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

descstr = "Crawl given root for data.csv files from BP app. \
			Create a complete dataset and subsets,save as .csv."

parser = argparse.ArgumentParser(description='Short: Crawl given folder for given name of data csv file\n\nLong: ' + descstr)
parser.add_argument('folder', type=str,
					help='root folder of a tree to scan for dataname')
parser.add_argument('dataname', type=str,
					help='name of dataset files (.csv) i.e. data.csv')

args = parser.parse_args()

dflist = []

cols = ["folder", "pic1", "pic2", "det", "desc", "matches", "inliers", "score", "det time",
		"desc time", "hmg time", "saved as", "blank"]


for root, dirs, files in os.walk(args.folder):
	if args.dataname in files:
		print root + "/" + args.dataname
		t_df = pd.read_csv(root + "/" + args.dataname)
		t_df = pd.DataFrame(t_df.values, columns=cols)
		dflist.append(t_df)
		print "number of t_df rows: ", len(t_df)
		print "t_df: \n", t_df.head(3)

df = pd.concat(dflist, axis=0)

# score percent conversion
df["score"] = 100*(1 - np.arctan(np.array(df["score"])*0.0001)/(np.pi/2.))

# kill the blank column
df = pd.DataFrame(df.loc[:, ["folder", "pic1", "pic2", "det", "desc", "matches", "inliers", "score", "det time",
		"desc time", "hmg time", "saved as"]])

df.to_csv("data_all.csv")

def createSubset(folders):
	inlst = []
	for f in folders:
		inlst.append(df[df["folder"] == f])
	return pd.concat(inlst)

# subset ZOOM: ------------------------------------------------
folders = ["out/ASTERIX/", "out/BELLEDONNE/", "out/BIP/", "out/CROLLE/"]

df_zoom = createSubset(folders)
df_zoom.to_csv("data_zoom.csv")
# -------------------------------------------------------------

# subset BLUR: ------------------------------------------------
folders = ["out/bikes/"]

df_blur = createSubset(folders)
df_blur.to_csv("data_blur.csv")
# -------------------------------------------------------------

# subset ROT: ------------------------------------------------
folders = ["out/boat/", "out/EAST_PARK/", "out/MARS/", "out/MONET/", "out/NewYork/"]
df_rot = createSubset(folders)

df_rot.to_csv("data_rot.csv")
# -------------------------------------------------------------

# subset ANGLE: ------------------------------------------------
folders = ["out/graff/", "out/Graffiti6/"]
df_angle = createSubset(folders)

df_angle.to_csv("data_angle.csv")
# -------------------------------------------------------------

# subset LIGHT: ------------------------------------------------
folders = ["out/light/"]
df_light = createSubset(folders)

df_light.to_csv("data_light.csv")
# -------------------------------------------------------------

# subset RES: ------------------------------------------------
folders = ["out/ubc/"]
df_res = createSubset(folders)

df_res.to_csv("data_res.csv")
# -------------------------------------------------------------


sys.exit(0)
