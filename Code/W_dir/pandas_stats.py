

"""Create tables & graphs from datasets generated with
	get_data.py"""

import argparse
import os, sys
import re
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------------------------------------------
# args parsing:

params = {	'legend.fontsize': 6,
			'legend.handlelength': 2}
plt.rcParams.update(params)

descstr = "Create tables & graphs from datasets generated with \
			get_data.py"

parser = argparse.ArgumentParser(description=descstr)
parser.add_argument('--detperf', help='show detector performance table', action="store_true")
parser.add_argument('--descperf', help='show descriptor performance table', action="store_true")
parser.add_argument('--comboperf', help='show descriptor/descriptor average time table', action="store_true")
parser.add_argument('--dettimes', help='show descriptor average time table', action="store_true")
parser.add_argument('--desctimes', help='show descriptor average time table', action="store_true")
parser.add_argument('--matchcount', help='avg count of matches/inliers/score', action="store_true")
parser.add_argument('--graphzoom', help='graph performance on zoom dataset', action="store_true")
parser.add_argument('--graphrot', help='graph performance on rot dataset', action="store_true")

parser.add_argument('--save', help='save all drawn plots', action="store_true")

# --detperf --descperf --comboperf --dettimes --desctimes --matchcount --graphzoom --graphrot

fignames = []
figs = []
args = parser.parse_args()

# -------------------------------------------------------------------------------------------------------------
# functions:

def plotTable(df, title, header=None):

	fig = plt.figure()
	fig.suptitle(title, fontsize="x-large")

	figs.append(fig)
	fignames.append(title)

	if not header:
		tab = plt.table(cellText=df.values,
					# colWidths=[0.08] * len(df.columns),
					loc="center",
					cellLoc='center')
	else:
		tab = plt.table(cellText=df.values,
					# colWidths=[0.08] * len(df.columns),
					loc="center",
					cellLoc='center',
					colLabels=header)

	plt.axis("off")
	# tab.set_fontsize(50)
	# plt.show()

def createMeanTable(datasets, cat, cat_vals, columns, title, header=None):
	# datasets = [df, df_zoom, df_blur, df_rot, df_angle, df_light, df_res]
	# det_methods = [" Harris", " GFTT", " SIFT", " SURF", " FAST", " MSER", " ORB"]

	tmplst = []
	# cat = "det"
	for dm in cat_vals:
		tmplst.append([dm])
		for ds in datasets:
			for c in columns:
				tmplst[-1].append(ds[ds[cat] == dm].mean()[c])

	# det_header = ["det method", "total score", "zoom score", "blur score", "rot score", "angle score", "light score", "res score"]
	df_performance = pd.DataFrame(tmplst)
	plotTable(df_performance, title, header)

def createComboTable(datasets, cat1, cat2, cat1_vals, cat2_vals, columns, title, header=None):
	# datasets = [df, df_zoom, df_blur, df_rot, df_angle, df_light, df_res]
	# det_methods = [" Harris", " GFTT", " SIFT", " SURF", " FAST", " MSER", " ORB"]

	tmplst = []
	# cat = "det"
	for dm1 in cat1_vals:
		for dm2 in cat2_vals:
			tmplst.append([dm1 + " ->" + dm2])
			for ds in datasets:
				for c in columns:
					# tmplst[-1].append(ds[ds[cat1] == dm1 and ds[cat2] == dm2].mean()[c])
					ds_cat1 = ds[ds[cat1] == dm1]
					ds_cat2 = ds_cat1[ds_cat1[cat2] == dm2]
					tmplst[-1].append(ds_cat2.mean()[c])

	# det_header = ["det method", "total score", "zoom score", "blur score", "rot score", "angle score", "light score", "res score"]
	df_performance = pd.DataFrame(tmplst)
	plotTable(df_performance, title, header)

def graphDataset(df_tst, title):

	df_tst["methods"] = df_tst["det"] + " ->" + df_tst["desc"]
	df_tst["pics"] = df_tst["pic1"] + " ->" + df_tst["pic2"]
	df_tst = df_tst.loc[:,["pics", "methods", "score"]]

	pics = df_tst["pics"].values
	# print pics
	out = []
	num = re.compile(ur"\d+")
	for p in pics:
		out.append( int(re.findall(num, p)[1]) )
	df_tst["pics"] = out

	# print pics
	# df_tst["pics"] = df_tst["pics"][-4:]
	df_tst = df_tst.pivot(index="pics", columns="methods", values="score")

	# df_tst["pics"] = df_tst["pics"].split()

	# print df_tst.head(100)

	# plotTable(df_tst)
	# fig = plt.figure()
	dfplt = df_tst.plot()
	dfplt.set_title(title, fontsize="x-large")

	fig = dfplt.figure

	figs.append(fig)
	fignames.append(title)

	# plt.plot(df_tst.values)

	# plt.show()


# header
cols = ["folder", "pic1", "pic2", "det", "desc", "matches", "inliers", "score", "det time",
		"desc time", "hmg time", "saved as", "blank"]

# -------------------------------------------------------------------------------------------------------------
# dataset loading - see get_data.py:
df = pd.read_csv("data_all.csv")
df_zoom = pd.read_csv("data_zoom.csv")
df_blur = pd.read_csv("data_blur.csv")
df_rot = pd.read_csv("data_rot.csv")
df_angle = pd.read_csv("data_angle.csv")
df_light = pd.read_csv("data_light.csv")
df_res = pd.read_csv("data_res.csv")

df_1zoom = df_zoom[df_zoom["folder"] == "out/ASTERIX/"]
df_1rot = df_rot[df_rot["folder"] == "out/MONET/"]

datasets = [df, df_zoom, df_blur, df_rot, df_angle, df_light, df_res]
# TABS: ============================================================
if args.detperf:
	det_methods = [" Harris", " GFTT", " SIFT", " SURF", " FAST", " MSER", " ORB"]
	det_header = [	"det method", "total score [%]", "zoom score [%]", "blur score [%]", "rot score [%]",
				  	"angle score [%]", "light score [%]", "res score [%]"]
	cat = "det"

	createMeanTable(datasets, cat, det_methods, ["score"], "Detector performance across datasets", det_header)

if args.descperf:
	det_methods = [" BRIEF", " SIFT", " SURF", " ORB"]
	desc_header = ["desc method", "total score [%]", "zoom score [%]", "blur score [%]", "rot score [%]",
				  	"angle score [%]", "light score [%]", "res score [%]"]
	cat = "desc"

	createMeanTable(datasets, cat, det_methods, ["score"], "Descriptor performance across datasets", desc_header)

if args.comboperf:
	det_methods = [" Harris", " GFTT", " SIFT", " SURF", " FAST", " MSER", " ORB"]
	desc_methods = [" BRIEF", " SIFT", " SURF", " ORB"]
	desc_header = [	"methods", "total score [%]", "zoom score [%]", "blur score [%]", "rot score [%]",
				  	"angle score [%]", "light score [%]", "res score [%]"]

	createComboTable(datasets, "det", "desc", det_methods, desc_methods, ["score"],
					 "Detector+descriptor performance across datasets", desc_header)

if args.dettimes:
	det_methods = [" Harris", " GFTT", " SIFT", " SURF", " FAST", " MSER", " ORB"]
	det_header = ["Detection method", "avg det time [s]"]

	cat = "det"

	columns = ["det time"]

	createMeanTable([df], cat, det_methods, columns, "Detection times across datasets", det_header)

if args.desctimes:
	det_methods = [" BRIEF", " SIFT", " SURF", " ORB"]
	desc_header = ["Description method", "avg desc time [s]"]
	# tmplst = []
	cat = "desc"

	columns = ["desc time"]

	createMeanTable([df], cat, det_methods, columns, "Description times across datasets", desc_header)

if args.matchcount:
	det_methods = [" Harris", " GFTT", " SIFT", " SURF", " FAST", " MSER", " ORB"]
	desc_methods = [" BRIEF", " SIFT", " SURF", " ORB"]
	header = ["methods", "avg matches", "avg inliers", "avg score [%]"]

	createComboTable([df], "det", "desc", det_methods, desc_methods, ["matches", "inliers", "score"],
						"Match count across datasets", header)

# GRAPHS ======================================================================

if args.graphzoom:

	graphDataset(df_1zoom, "Method combination performance across Asterix dataset (zoom)")

if args.graphrot:

	graphDataset(df_1rot, "Method combination performance across Monet dataset (rotation)")

if args.save:
	for i in range(len(figs)):
		finame = fignames[i] + ".pdf"
		with open(finame, "w") as fileToWrite:
			figs[i].savefig(fileToWrite, format='pdf')
		# f.savefig(f.title(), format='pdf')
else:
	plt.show()

