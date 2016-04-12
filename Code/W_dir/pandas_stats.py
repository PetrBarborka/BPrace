# -*- coding: utf-8 -*-

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

parser.add_argument('--all_tabs', help='generate all tables (see -h)', action="store_true")

parser.add_argument('--graphzoom', help='graph performance on zoom dataset', action="store_true")
parser.add_argument('--graphrot', help='graph performance on rot dataset', action="store_true")

# parser.add_argument('--save', help='save all drawn plots', action="store_true")
parser.add_argument('--tex', help='save all drawn plots as tex', action="store_true")

parser.add_argument('--out', help='specify folder to which to put files', type=str)

# --detperf --descperf --comboperf --dettimes --desctimes --matchcount --graphzoom --graphrot

fignames = []
figs = []
texs = []
filenames = []
args = parser.parse_args()

if args.all_tabs:
	args.detperf = args.descperf = args.comboperf = args.dettimes = args.desctimes = args.matchcount = True

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

def plotGraph(df, title):

	dfplt = df.plot()
	dfplt.set_title(title, fontsize="x-large")

	fig = dfplt.figure

	figs.append(fig)
	fignames.append(title)


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
			# tmplst.append([dm1 + " ->" + dm2])
			tmplst.append([dm1, dm2])
			for ds in datasets:
				for c in columns:
					# tmplst[-1].append(ds[ds[cat1] == dm1 and ds[cat2] == dm2].mean()[c])
					ds_cat1 = ds[ds[cat1] == dm1]
					ds_cat2 = ds_cat1[ds_cat1[cat2] == dm2]
					tmplst[-1].append(ds_cat2.mean()[c])

	# det_header = ["det method", "total score", "zoom score", "blur score", "rot score", "angle score", "light score", "res score"]
	df_performance = pd.DataFrame(tmplst)
	return df_performance

def graphDataset(df_tst):

	df_tst["methods"] = df_tst["det"] + " ->" + df_tst["desc"]
	df_tst["pics"] = df_tst["pic1"] + " ->" + df_tst["pic2"]
	df_tst = df_tst.loc[:,["pics", "methods", "score"]]

	pics = df_tst["pics"].values
	# print pics
	out = []
	num = re.compile(ur"\d+")
	for p in pics:
		out.append( int(re.findall(num, p)[1]))
	df_tst["pics"] = out

	# print pics
	# df_tst["pics"] = df_tst["pics"][-4:]
	df_tst = df_tst.pivot(index="pics", columns="methods", values="score")

	return df_tst


def createTexTable(values, form=None, header=None, r_header=None, label=None, caption=None):

	width = len(values[0])

	outstr = "\\begin{table}[htbp]\\centering\n"

	if r_header is not None:
		assert len(r_header) == len(values), "row header has to be same length as number of rows in values: len(values) == " \
											+ str(len(values)) + " len(r_header) == " + str(len(r_header))
		width += 1

	if form is None:
		outstr += "\\begin{tabular}{ " + "l " * width + "}\n"
	else:
		assert len(form) == len(values[0]), "form has to be same length as number of columns in values + header if present: \
											len(values[0]) == " \
											+ str(width) + " len(form) == " + str(len(form))
		fstr = ""
		for f in form: fstr += f + " "
		outstr += "\\begin{tabular}{ " + fstr + "}\n"

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
		outstr += "\\\\\n\t\\hline\n"

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
	outstr += "\n\\end{tabular}\n"
	if caption is not None:
		outstr += "	\\caption[Short Heading]{\\protect " + caption + "}"
		if label is not None:
			outstr += "\\label{" + label + "}"
		outstr += "\n"
	outstr += "\\end{table}"
	return outstr

def createTexBoxplot(values, span, header=None, label=None, caption=None):
	tv = np.zeros((len(values[0]), len(values)))
	# mtrx transposition
	for i in range(len(values)):
		for j in range(len(values[0])):
			tv[j][i] = values[i][j]
	values = np.array(tv, dtype=float)
	tv = []
	for i in range(len(values)):
		# [mean - std, mean, mean std]
		m = np.nanmean(values[i])
		v = np.nanstd(values[i])
		tv.append([np.clip(m-v, 0, 100), np.clip(m, 0, 100), np.clip(m+v, 0, 100)])
	values = np.array(tv)
	outstr = "% Preamble: \pgfplotsset{width=7cm,compat=1.13}\usepgfplotslibrary{statistics}\n" + \
			"\\begin{figure} \n " + \
			"\\begin{tikzpicture} \n " + \
			"\\footnotesize\n " + \
			"\t\\begin{axis}[ \n" + \
			"\t\tboxplot/draw direction=y, \n" + \
			"\t\txticklabel style = {xshift=-0.3cm, rotate=75},\n" + \
			"\t\tylabel = výkonnost \\%,\n" + \
			"\t\txtick={"
	for i in range(len(values)-1):
		outstr += str(i) + ", "
	outstr += str(len(values)-1)
	outstr += "},\n"
	if header is not None:
		outstr += "\t\txticklabels={"
		for h in header:
			outstr += h + ", "
		outstr = outstr[:-2] + "}\n"
	outstr += "\t]\n"
	c = 0
	for v in values:
		outstr += "\t\\addplot+[\n\t\tboxplot prepared={\n\t\tdraw position=" + str(c) + ",\n"
		outstr += "\t\tlower whisker=" + str(span[0]) + ",\n\t\tlower quartile=" + str(v[0]) + ",\n"
		outstr += "\t\tmedian=" + str(v[1]) + ",\n\t\tupper quartile=" + str(v[2]) + ",\n"
		outstr += "\t\tupper whisker=" + str(span[1]) + "}]\n"
		outstr += "\t\tcoordinates {};\n"
		# outstr += "val " + str(c) + ":\n" + str(v) + "\n"
		# outstr += "type of v[0]: " + str(type(v[0])) + "\n"
		c += 1

	outstr += "\t\\end{axis} \n  \\end{tikzpicture}\n"
	if caption is not None:
		outstr += "	\\caption[Short Heading]{\\protect " + caption + "}"
		if label is not None:
			outstr += "\\label{" + label + "}"
		outstr += "\n"
	outstr += "\\end{figure}"
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
	det_header = ["Detektor", "celkově[%]", "zoom[%]", "blur[%]", "rot[%]",
				   "angle[%]", "light[%]", "res[%]"]
	title = "Celková výkonnost detektorů na datasetech"
	cat = "det"

	tab = createMeanTable(datasets, cat, det_methods, ["score"])

	filename = "tab_detperf"
	filenames.append(filename)

	if args.tex:
		texs.append(createTexTable(tab.values, header=det_header, label=filename, form=["l|", "r", "r", "r", "r", "r", "r", "r"], caption=title))
	else:
		plotTable(tab, title, det_header)

if args.descperf:
	det_methods = [" BRIEF", " SIFT", " SURF", " ORB"]
	desr_header = ["Deskriptor", "celkově[%]", "zoom[%]", "blur[%]", "rot[%]",
				   "angle[%]", "light[%]", "res[%]"]
	cat = "desc"

	title = "Celková výkonnost deskriptorů na datasetech"

	tab = createMeanTable(datasets, cat, det_methods, ["score"])

	filename = "tab_descperf"
	filenames.append(filename)

	if args.tex:
		texs.append(createTexTable(tab.values, header=desr_header, label=filename, form=["l|", "r", "r", "r", "r", "r", "r", "r"], caption=title))
	else:
		plotTable(tab, title, desr_header)

if args.comboperf:
	det_methods = [" Harris", " GFTT", " SIFT", " SURF", " FAST", " MSER", " ORB"]
	desc_methods = [" BRIEF", " SIFT", " SURF", " ORB"]
	# desr_header = [	"methods", "total score [%]", "zoom score [%]", "blur score [%]", "rot score [%]",
				  	# "angle score [%]", "light score [%]", "res score [%]"]
	desr_header = [	"Detektor", "Deskriptor", "celkově[%]", "zoom[%]", "blur[%]", "rot[%]",
				   "angle[%]", "light[%]", "res[%]"]
	title = "Celková výkonnost kombinací detektor -> deskriptor"

	tab = createComboTable(datasets, "det", "desc", det_methods, desc_methods, ["score"])

	filename = "tab_comboperf"
	filenames.append(filename)

	if args.tex:
		texs.append(createTexTable(tab.values, header=desr_header, label=filename, form=["l", "l|", "r", "r", "r", "r", "r", "r", "r"], caption=title))
	else:
		plotTable(tab, title, desr_header)

if args.dettimes:
	det_methods = [" Harris", " GFTT", " SIFT", " SURF", " FAST", " MSER", " ORB"]
	det_header = ["Detektor", "průměrný čas detekce [s]"]

	cat = "det"
	title = "Průměrné časy detekce"

	columns = ["det time"]

	tab = createMeanTable([df], cat, det_methods, columns)

	filename = "tab_dettimes"
	filenames.append(filename)

	if args.tex:
		texs.append(createTexTable(tab.values, header=det_header, label=filename, form=["l|", "r"], caption=title))
	else:
		plotTable(tab, title, det_header)

if args.desctimes:
	det_methods = [" BRIEF", " SIFT", " SURF", " ORB"]
	desr_header = ["Deskriptor", "průměrný čas deskripce [s]"]
	# tmplst = []
	cat = "desc"
	title = "Průměrné časy deskripce"

	columns = ["desc time"]

	tab = createMeanTable([df], cat, det_methods, columns)

	filename = "tab_desctimes"
	filenames.append(filename)

	if args.tex:
		texs.append(createTexTable(tab.values, header=desr_header, label=filename, form=["l|", "r"], caption=title))
	else:
		plotTable(tab, title, desr_header)

if args.matchcount:
	det_methods = [" Harris", " GFTT", " SIFT", " SURF", " FAST", " MSER", " ORB"]
	desc_methods = [" BRIEF", " SIFT", " SURF", " ORB"]
	header = ["Detektor", "Deskriptor", "průměrně párů", "průměrně použitých párů", "průměrné skóre [%]"]
	title = "Počty nalezených párů bodů"

	tab = createComboTable([df], "det", "desc", det_methods, desc_methods, ["matches", "inliers", "score"])

	filename = "tab_matchcount"
	filenames.append(filename)

	if args.tex:
		texs.append(createTexTable(tab.values, header=header, label=filename, form=["l", "l|", "r", "r", "r"], caption=title))
	else:
		plotTable(tab, title, header)

# GRAPHS ======================================================================

if args.graphzoom:

	det_methods = ["Harris", "GFTT", "SIFT", "SURF", "FAST", "MSER", "ORB"]
	desc_methods = ["BRIEF", "SIFT", "SURF", "ORB"]

	header = []
	for dt in det_methods:
		for dsc in desc_methods:
			header.append(dt + " -> " + dsc)

	filename = "graph_zoom"
	filenames.append(filename)

	title = "Střední hodnota a standartní odchylka výkonnosti kombinací metod na datasetu Asterix (zoom)"
	df = graphDataset(df_1zoom)
	if args.tex:
		texs.append(createTexBoxplot(df.values, [0, 100], header=header, caption=title, label=filename))
	else:
		plotGraph(df, title)

if args.graphrot:

	det_methods = ["Harris", "GFTT", "SIFT", "SURF", "FAST", "MSER", "ORB"]
	desc_methods = ["BRIEF", "SIFT", "SURF", "ORB"]

	header = []
	for dt in det_methods:
		for dsc in desc_methods:
			header.append(dt + " -> " + dsc)

	filename = "graph_rot"
	filenames.append(filename)

	title = "Střední hodnota a standartní odchylka výkonnosti kombinací metod na datasetu Monet (rotace)"
	df = graphDataset(df_1rot)
	if args.tex:
		texs.append(createTexBoxplot(df.values, [0, 100], header=header, caption=title, label=filename))
	else:
		plotGraph(df, title)


if args.out is not None:
	for i in range(len(filenames)):
		finame = args.out + filenames[i]
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
