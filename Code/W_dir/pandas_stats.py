

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
parser.add_argument('--tex', help='save all drawn plots as tex', action="store_true")

# --detperf --descperf --comboperf --dettimes --desctimes --matchcount --graphzoom --graphrot

fignames = []
figs = []
texs = []
filenames = []
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

def createMeanTable(datasets, cat, cat_vals, columns):
	tmplst = []
	for dm in cat_vals:
		tmplst.append([dm])
		for ds in datasets:
			for c in columns:
				tmplst[-1].append(ds[ds[cat] == dm].mean()[c])
	df_performance = pd.DataFrame(tmplst)
	return df_performance

def createComboTable(datasets, cat1, cat2, cat1_vals, cat2_vals, columns):
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
	return df_performance

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

def createTexTable(values, form=None, header=None, r_header=None, label=None):

	width = len(values[0])

	if r_header is not None:
		assert len(r_header) == len(values), "row header has to be same length as number of rows in values: len(values) == " \
											+ str(len(values)) + " len(r_header) == " + str(len(r_header))
		width += 1

	if form is None:
		outstr = "\\begin{tabular}{ " + "l " * width + "}\n"
	else:
		assert len(form) == len(values[0]), "form has to be same length as number of columns in values + header if present: \
											len(values[0]) == " \
											+ width + " len(form) == " + str(len(form))
		fstr = ""
		for f in form: fstr += f + " "
		outstr = "\\begin{tabular}{ " + fstr + "}\n"

	if label is not None:
		outstr += "\\label{" + label + "}\n"

	if header is not None:
		assert len(header) == len(values[0]), "header has to be same length as number of columns in values: len(values[0]) == " \
											+ str(len(values[0])) + " len(header) == " + str(len(header))
		outstr += "\t"

		for hr in range(len(header)):
			if type(header[hr]) == str:
				for o in range(len(header[hr])):
					if header[hr][o] == "%":
						header[hr] = header[hr][:o] + "\\%" + header[hr][o+1:]
						break

		if r_header is not None:
			outstr += " & "
		for h in header:
			outstr += str(h) + " & "
		outstr = outstr[:-2]
		outstr += "\\\\\n\t\hline\n"

	r_h_idx = 0

	for r in values:
		outstr += "\t"
		if r_header is not None:
			outstr += str(r_header[r_h_idx]) + " & "
			r_h_idx += 1
		for c in r:
			val = ""
			if type(c) == float:
				val = "{:1.2f}".format(c)
			elif type(c) == str:
				val = c
				for o in range(len(c)):
					if c[o] == "%":
						print "values: putting \\"
						val = c[:o] + "\\" + c[o:]
			else:
				val = str(c)
			outstr += val + " & "
		outstr = outstr[:-2]
		outstr += "\\\\\n"
	outstr = outstr[:-4]
	outstr += "\n\end{tabular}"

	return outstr


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
	title = "Detector performance across datasets"
	cat = "det"

	tab = createMeanTable(datasets, cat, det_methods, ["score"])

	filename = "tab_detperf"
	filenames.append(filename)

	if args.tex:
		texs.append(createTexTable(tab.values, header=det_header, label=filename, form=["l|", "r", "r", "r", "r", "r", "r", "r"]))
	else:
		plotTable(tab, title, det_header)

if args.descperf:
	det_methods = [" BRIEF", " SIFT", " SURF", " ORB"]
	desr_header = ["desc method", "total score [%]", "zoom score [%]", "blur score [%]", "rot score [%]",
				  	"angle score [%]", "light score [%]", "res score [%]"]
	cat = "desc"

	title = "Descriptor performance across datasets"

	tab = createMeanTable(datasets, cat, det_methods, ["score"])

	filename = "tab_descperf"
	filenames.append(filename)

	if args.tex:
		texs.append(createTexTable(tab.values, header=desr_header, label=filename, form=["l|", "r", "r", "r", "r", "r", "r", "r"]))
	else:
		plotTable(tab, title, desr_header)

if args.comboperf:
	det_methods = [" Harris", " GFTT", " SIFT", " SURF", " FAST", " MSER", " ORB"]
	desc_methods = [" BRIEF", " SIFT", " SURF", " ORB"]
	# desr_header = [	"methods", "total score [%]", "zoom score [%]", "blur score [%]", "rot score [%]",
				  	# "angle score [%]", "light score [%]", "res score [%]"]
	desr_header = [	"methods", "total[%]", "zoom[%]", "blur[%]", "rot[%]",
				  	"angle[%]", "light[%]", "res[%]"]
	title = "Detector+descriptor performance across datasets"

	tab = createComboTable(datasets, "det", "desc", det_methods, desc_methods, ["score"])

	filename = "tab_comboperf"
	filenames.append(filename)

	if args.tex:
		texs.append(createTexTable(tab.values, header=desr_header, label=filename, form=["l|", "r", "r", "r", "r", "r", "r", "r"]))
	else:
		plotTable(tab, title, desr_header)

if args.dettimes:
	det_methods = [" Harris", " GFTT", " SIFT", " SURF", " FAST", " MSER", " ORB"]
	det_header = ["Detection method", "avg det time [s]"]

	cat = "det"
	title = "Detection times across datasets"

	columns = ["det time"]

	tab = createMeanTable([df], cat, det_methods, columns)

	filename = "tab_dettimes"
	filenames.append(filename)

	if args.tex:
		texs.append(createTexTable(tab.values, header=det_header, label=filename, form=["l|", "r"]))
	else:
		plotTable(tab, title, det_header)

if args.desctimes:
	det_methods = [" BRIEF", " SIFT", " SURF", " ORB"]
	desr_header = ["Description method", "avg desc time [s]"]
	# tmplst = []
	cat = "desc"
	title = "Description times across datasets"

	columns = ["desc time"]

	tab = createMeanTable([df], cat, det_methods, columns)

	filename = "tab_desctimes"
	filenames.append(filename)

	if args.tex:
		texs.append(createTexTable(tab.values, header=desr_header, label=filename, form=["l|", "r"]))
	else:
		plotTable(tab, title, desr_header)

if args.matchcount:
	det_methods = [" Harris", " GFTT", " SIFT", " SURF", " FAST", " MSER", " ORB"]
	desc_methods = [" BRIEF", " SIFT", " SURF", " ORB"]
	header = ["methods", "avg matches", "avg inliers", "avg score [%]"]
	title = "Match count across datasets"

	tab = createComboTable([df], "det", "desc", det_methods, desc_methods, ["matches", "inliers", "score"])

	filename = "tab_matchcount"
	filenames.append(filename)

	if args.tex:
		texs.append(createTexTable(tab.values, header=header, label=filename, form=["l|", "r", "r", "r"]))
	else:
		plotTable(tab, title, header)

# GRAPHS ======================================================================

if args.graphzoom:

	graphDataset(df_1zoom, "Method combination performance across Asterix dataset (zoom)")

if args.graphrot:

	graphDataset(df_1rot, "Method combination performance across Monet dataset (rotation)")

if args.save:
	for i in range(len(filenames)):
		finame = filenames[i]
		if args.tex:
			finame += ".tex"
			with open(finame, "w") as fileToWrite:
				fileToWrite.write(texs[i])
		else:
			finame += ".pdf"
			with open(finame, "w") as fileToWrite:
				figs[i].savefig(fileToWrite, format='pdf')
			# f.savefig(f.title(), format='pdf')
else:
	plt.show()
	for tx in texs:
		print tx

# tstdat = [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]
# header = ["tstval1", "tstval2", 'val3', 'val4']
# form = ["l", "c", "c", "r"]
# r_header = ["row1", "row2", "row1000"]

# print createTexTable(tstdat, form=form, header=header, r_header=r_header)
