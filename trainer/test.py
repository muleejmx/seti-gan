from pylab import *
import os
import numpy as np
import pandas as pd
import csv
import pickle
import fitsio

data_loc = '/Users/mulan/desktop/fits_data/'
ext = '.fits'
typ = '_diag'
root_dir = os.path.abspath('.')
data_dir = os.path.join(root_dir, 'Data')

with open('labelled_files.pkl', 'rb') as f:
    data = pickle.load(f)
data = [k for k, v in data.iteritems() if v == 6.0 or v == 7.0]

normalized = []

for d in data:
	max_i = max([max(f) for f in fitsio.read(data_loc + str(d) + ext)])
	min_i = min([min(f) for f in fitsio.read(data_loc + str(d) + ext)])
	normalized.append([2 * (x - min_i) / (max_i - min_i) - 1 for x in fitsio.read(data_loc + str(d) + ext)])

print(len(normalized))
print(len(normalized[0]))
print(len(normalized[0][0]))

normalized = np.asarray(normalized)

with open('normalized_diag.pkl', 'wb') as outfile:
    pickle.dump(normalized, outfile, protocol=pickle.HIGHEST_PROTOCOL)


# max_i = max([max(f) for f in fitsio.read(data_loc + str(data[0]) + ext)])
# min_i = min([min(f) for f in fitsio.read(data_loc + str(data[0]) + ext)])

# a = [2 * (x - min_i) / (max_i - min_i) - 1 for x in fitsio.read(data_loc + str(data[0]) + ext)]
