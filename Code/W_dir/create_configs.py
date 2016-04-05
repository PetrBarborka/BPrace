
import argparse
import os
import re
import json

"""
Crawl datapath for .pgm, .ppm, and .png files and
their corresponding homography ground truth matrices.
Generate their respective outconf.json and pics.json
"""

def sortStingListByNumber(inlist, n):
	"""sort list of strings containing numbers by the nth number"""
    indices = []
    outlist = []
    nums = []
    number = re.compile(ur"\d+")
    # print "inlist: ", inlist
    for e in inlist:
        cur_nums = re.findall(number, e)
        cur_nums_list = []
        for s in cur_nums:
            cur_nums_list.append(int(s))
        num = int(re.findall(number, e)[n])
        if indices:
            if indices[0] > num:
                indices = [num] + indices
                outlist = [e] + outlist
                nums = [cur_nums_list] + nums
            elif indices[-1] < num:
                indices.append(num)
                outlist = outlist + [e]
                nums = nums + [cur_nums_list]
            for i in range(1, len(indices)):
                if indices[i-1] < num and indices[i] > num:
                    indices = indices[:i] + [num] + indices[i:]
                    outlist = outlist[:i] + [e] + outlist[i:]
                    nums = nums[:i] + [cur_nums_list] + nums[i:]
        else:
            indices.append(num)
            outlist.append(e)
            nums.append(cur_nums_list)

    return outlist, nums


parser = argparse.ArgumentParser(description='Create pics json and outconf json for given data')
parser.add_argument('datapath', type=str,
                   help='path containing folders with datasets')
parser.add_argument('outpath', type=str,
                   help='path to output files to')

args = parser.parse_args()

# print "datapath: ", args.datapath, " outpath: ", args.outpath

for root, dirs, files in os.walk(args.datapath):
    # print "root: ", root, " dirs: ", dirs, " files: ", files
    dotpgm = re.compile(r"\w*\.(pgm)")
    pgms = []
    dotppm = re.compile(r"\w*\.(ppm)")
    ppms = []
    dotpng = re.compile(r"\w*\.(png)")
    pngs = []
    number = re.compile(ur"\d+")
    numbers = []
    H1 = re.compile(ur"H0to\d+$")
    H2 = re.compile(ur"H1to\d+$")
    H3 = re.compile(ur"H0to\d+p$")
    H4 = re.compile(ur"H1to\d+p$")
    H1s = []
    H2s = []
    H3s = []
    H4s = []

    for f in files:
        nmatch = re.findall(number, f)
        if dotpgm.match(f) and len(nmatch) == 1:
            pgms.append(f)
            numbers.append(int(nmatch[0]))
        elif dotppm.match(f) and len(nmatch) == 1:
            ppms.append(f)
            numbers.append(int(nmatch[0]))
        elif dotpng.match(f) and len(nmatch) == 1:
            pngs.append(f)
            numbers.append(int(nmatch[0]))
        if H1.match(f):
            H1s.append(f)
        elif H2.match(f):
            H2s.append(f)
        elif H3.match(f):
            H3s.append(f)
        elif H4.match(f):
            H4s.append(f)

    Hs = []
    if H1s:
        Hs = H1s
    elif H2s:
        Hs = H2s
    elif H3s:
        Hs = H3s
    elif H4s:
        Hs = H4s

    pics = []
    if pgms:
        pics = pgms
    elif ppms:
        pics = ppms
    elif pngs:
        pics = pngs

    pics_dict = {}

    if pics:

    	outconf_dict = {"picspath" : args.outpath + root + "/", "csvpath" : args.outpath + root + "/data.csv"}

        pics, picsIdc = sortStingListByNumber(pics, 0)
        Hs, HsIdc = sortStingListByNumber(Hs, 1)
        maxn = max(numbers)
        minn = min(numbers)
        if Hs:
            idx = 0
            for hi in HsIdc:
            	key = "pair" + str(idx) + ":"
            	# print "hi: ", hi
            	pic1idx = [i for i, j in enumerate(picsIdc) if j == [hi[0]]]
            	pic2idx = [i for i, j in enumerate(picsIdc) if j == [hi[1]]]
            	if pic1idx and pic2idx and pic1idx[0] < len(pics) and pic2idx[0] < len(pics):
	            	pic1 = root + "/" + pics[pic1idx[0]]
	            	pic2 = root + "/" + pics[pic2idx[0]]
	                pics_dict[key] = [pic1, pic2, root + "/" + Hs[idx]]
                idx += 1
        folder_re = re.compile(ur"[\w\d ]+$")
        folder = folder_re.search(root).group(0)

        if pics_dict:
        	filename_pics = "pics_" + folder.lower() + ".json"
        	filename_outconf = "outconf_" + folder.lower() + ".json"

        	os.mkdir(args.outpath + folder)

        	with open("configs/" + filename_pics, "w") as f:
        		json.dump(pics_dict, f)
        	with open("configs/" + filename_outconf, "w") as f:
        		json.dump(outconf_dict, f)
















