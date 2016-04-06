

"""Crawl given root for data.csv files from BP app.
	Make graphs, tables etc."""

import argparse
import os, sys
import re
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

descstr = "Crawl given root for data.csv files from BP app. \
			Make graphs, tables etc."

parser = argparse.ArgumentParser(description='Short: Crawl given folder for given name of data csv file\n\nLong: ' + descstr)
parser.add_argument('folder', type=str,
					help='root folder of a tree to scan for dataname')
parser.add_argument('dataname', type=str,
					help='name of dataset files (.csv) i.e. data.csv')

args = parser.parse_args()

# df = pd.DataFrame()
# dflist = []

cols = ["folder", "pic1", "pic2", "det", "desc", "matches", "inliers", "score", "det time",
		"desc time", "hmg time", "saved as", "blank"]

df = pd.read_csv("data_all.csv")
df_zoom = pd.read_csv("data_zoom.csv")
df_blur = pd.read_csv("data_blur.csv")
df_rot = pd.read_csv("data_rot.csv")
df_angle = pd.read_csv("data_angle.csv")
df_light = pd.read_csv("data_light.csv")
df_res = pd.read_csv("data_res.csv")

print "df_mean: ", df[df["det"] == " Harris"].mean()["score"]

# TAB: descriptor performance ============================================================
datasets = [df, df_zoom, df_blur, df_rot, df_angle, df_light, df_res]
det_methods = [" Harris", " GFTT", " SIFT", " SURF", " FAST", " MSER", " ORB"]

tmplst = []
cat = "det"
for dm in det_methods:
	tmplst.append([dm])
	for ds in datasets:
		tmplst[-1].append(ds[ds[cat] == dm].mean()["score"])

det_header = ["det method", "total score", "zoom score", "blur score", "rot score", "angle score", "light score", "res score"]
df_performance_desc = pd.DataFrame(tmplst, columns=det_header)
# df_performance_desc = pd.DataFrame(df_performance_desc, index="total score")

to_plot = df_performance_desc.values

print to_plot
tab = plt.table(cellText=to_plot,
				# colWidths=[0.08] * len(df.columns),
				loc="center",
				cellLoc='center',
				colLabels=det_header)
tab.set_fontsize(50)
plt.show()

# for root, dirs, files in os.walk(args.folder):
# 	if args.dataname in files:
# 		print root + "/" + args.dataname
# 		t_df = pd.read_csv(root + "/" + args.dataname)
# 		t_df = pd.DataFrame(t_df.values, columns=cols)
# 		dflist.append(t_df)
# 		print "number of t_df rows: ", len(t_df)
# 		print "t_df: \n", t_df.head(3)

# df = pd.concat(dflist, axis=0)

# # df.to_csv("data_all.csv")
# df = pd.read_csv("data_all.csv")

# # df["score"] = 100.*((np.pi/2.)-np.arctan(df["score"]*0.0001))

# # score percent conversion
# df["score"] = 100*(1 - np.arctan(np.array(df["score"])*0.0001)/(np.pi/2.))

# # kill the blank column
# df = pd.DataFrame(df.loc[:, ["folder", "pic1", "pic2", "det", "desc", "matches", "inliers", "score", "det time",
# 		"desc time", "hmg time", "saved as"]])

# df.to_csv("data_all.csv")

# print "df: \n", df.head(3)
# print "number of df columns: ", len(df.columns)
# print "number of df rows: ", len(df)


# # subset Harris: ----------------------------------------------
# Harris = df[df["det"] == " Harris"]

# print "Harris: \n", Harris.head(10)
# print "number of df columns: ", len(Harris.columns)
# print "number of df rows: ", len(Harris)
# # -------------------------------------------------------------

# subset ZOOM: ------------------------------------------------
# folders = ["out/ASTERIX/", "out/BELLEDONNE/", "out/BIP/", "out/CROLLE/"]
# inlst = []
# for f in folders:
# 	inlst.append(df[df["folder"] == f])
# df_zoom = pd.concat(inlst)

# df_zoom.to_csv("data_zoom.csv")
# df_zoom = pd.read_csv("data_zoom.csv")

# print "df_zoom: \n", df_zoom.head(10)
# print "number of df_zoom columns: ", len(df_zoom.columns)
# print "number of df_zoom rows: ", len(df_zoom)
# -------------------------------------------------------------

# subset BLUR: ------------------------------------------------
# folders = ["out/bikes/"]
# inlst = []
# for f in folders:
# 	inlst.append(df[df["folder"] == f])
# df_blur = pd.concat(inlst)

# df_blur.to_csv("data_blur.csv")
# df_blur = pd.read_csv("data_blur.csv")

# print "df_blur: \n", df_blur.head(10)
# print "number of df_blur columns: ", len(df_blur.columns)
# print "number of df_blur rows: ", len(df_blur)
# -------------------------------------------------------------

# subset ROT: ------------------------------------------------
# folders = ["out/boat/", "out/EAST_PARK/", "out/MARS/", "out/MONET/", "out/NewYork/"]
# inlst = []
# for f in folders:
# 	inlst.append(df[df["folder"] == f])
# df_rot = pd.concat(inlst)

# df_rot.to_csv("data_rot.csv")
# df_rot = pd.rot_csv("data_rot.csv")

# print "df_rot: \n", df_rot.head(10)
# print "number of df_rot columns: ", len(df_rot.columns)
# print "number of df_rot rows: ", len(df_rot)
# -------------------------------------------------------------

# subset ANGLE: ------------------------------------------------
# folders = ["out/graff/", "out/Graffiti6/"]
# inlst = []
# for f in folders:
# 	inlst.append(df[df["folder"] == f])
# df_angle = pd.concat(inlst)

# df_angle.to_csv("data_angle.csv")
# df_angle = pd.read_csv("data_angle.csv")

# print "df_angle: \n", df_angle.head(10)
# print "number of df_angle columns: ", len(df_angle.columns)
# print "number of df_angle rows: ", len(df_angle)
# -------------------------------------------------------------

# subset LIGHT: ------------------------------------------------
# folders = ["out/light/"]
# inlst = []
# for f in folders:
# 	inlst.append(df[df["folder"] == f])
# df_light = pd.concat(inlst)

# df_light.to_csv("data_light.csv")
# df_light = pd.read_csv("data_light.csv")

# print "df_light: \n", df_light.head(10)
# print "number of df_light columns: ", len(df_light.columns)
# print "number of df_light rows: ", len(df_light)
# -------------------------------------------------------------

# subset RES: ------------------------------------------------
# folders = ["out/ubc/"]
# inlst = []
# for f in folders:
# 	inlst.append(df[df["folder"] == f])
# df_res = pd.concat(inlst)

# df_res.to_csv("data_res.csv")
# df_res = pd.read_csv("data_res.csv")

# print "df_res: \n", df_res.head(10)
# print "number of df_res columns: ", len(df_res.columns)
# print "number of df_res rows: ", len(df_res)
# -------------------------------------------------------------

# TAB: descriptor performance ============================================================
# df_performance_desc = pd.DataFrame(["Harris", df
# 	])

# print df_zoom['score'].describe()

# # df = pd.DataFrame(df, columns=cols)
# df = pd.DataFrame(df.values, columns=cols)

# ttp = pd.DataFrame(df.loc[:10, ['pic1', 'pic2', "det", "desc", "matches",
# 								"inliers", "score", "det time",
# 								"desc time", "hmg time"]])

# print "table to plot: \n", ttp

# tab = plt.table(cellText=ttp.values,
# 				# colWidths=[0.08] * len(df.columns),
# 				loc="center",
# 				cellLoc='center')
# tab.set_fontsize(50)
# tab.scale(2, 2)

# plt.axis("off")
# plt.show()

# sys.exit(0)
