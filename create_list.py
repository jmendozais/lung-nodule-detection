import jsrt
from math import *

overlapped = ['LN060','LN065','LN105','LN108','LN112','LN113','LN115','LN126','LN130','LN133','LN136','LN149','LN151','LN152']

def print_jsrt140():
	paths = jsrt.get_paths()
	sub, siz, loc = jsrt.get_metadata()
	xfactor = 0.25
	yfactor = 0.25

	for i in range(len(paths)):
		count = 1 if siz[i][0] != -1 else 0
		valid = True
		for tok in overlapped:
			if paths[i].find(tok) != -1:
				valid = False
				break
		if not valid:
			continue

		print paths[i], count,
		if count > 0:
			print int(round(loc[i][0] * xfactor)), int(round(loc[i][1] * yfactor)), int(round(xfactor * max(siz[i][0], siz[i][1]))),
		print ''

def jsrt(set=None):
	paths = jsrt.get_paths()
	sub, siz, loc = jsrt.get_metadata()
	xfactor = 0.25
	yfactor = 0.25

	npaths = []
	nloc = []
	rads = []
	for i in range(len(paths)):
		count = 1 if siz[i][0] != -1 else 0
		if set == 'jsrt140':
			valid = True
			for tok in overlapped:
				if paths[i].find(tok) != -1:
					valid = False
					break
			if not valid:
				continue
		npaths.append(paths[i])
		if count > 0:
			nloc.append([int(round(loc[i][0] * xfactor)), int(round(loc[i][1] * yfactor))])
			rads.append(int(round(xfactor * max(siz[i][0], siz[i][1]))))
	
	return npaths, nloc, rads

def left_lung(set=None):
	lpath = '/Users/mac/Projects/data/scr/masks/left-lung.txt'
	f = open(lpath)
	paths = []
	for line in f:
		if set=='jsrt140':
			valid = True
			for tok in overlapped:
				if line.find(tok) != -1:
					valid = False
					break
			if not valid:
				continue
		paths.append(line)
	return paths

def right_lung(set=None):
	lpath = '/Users/mac/Projects/data/scr/masks/right-lung.txt'
	f = open(lpath)
	paths = []
	for line in f:
		if set=='jsrt140':
			valid = True
			for tok in overlapped:
				if line.find(tok) != -1:
					valid = False
					break
			if not valid:
				continue
		paths.append(line)
	return paths
