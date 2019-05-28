#%% IMPORT MODULES

from __future__ import division

import pickle

from src.cell_class import Cell

import sys
sys.path.append('./analysis/gaba_synapses')
import MiniDetector


#%% LOAD DATA

mIPSC_fnames = ['18o23003.abf',
                '18o23004.abf',
                '18o23005.abf']

DATA_PATH = './data/GABA_synapses/'

mini_recs = Cell().read_ABF([DATA_PATH + fname for fname in mIPSC_fnames])


#%% DETECT & INSPECT MINIS

mini_detectors = []

for i, rec in enumerate(mini_recs):

    # Load data and name MiniDetector based on filename.
    tmp = MiniDetector.MiniDetector(rec, remove_n_sweeps = 3)
    tmp.set_name(mIPSC_fnames[i][:-4] + '_MiniDetector')

    # Scrape minis.
    tmp.compute_gradient(100)
    tmp.find_grad_peaks(4.5, 450, width = 10)
    tmp.extract_minis((-50, 400))

    # Inspect scraped minis.
    tmp.plot_signal(10)
    tmp.plot_minis(25)

    # Save data.
    mini_detectors.append(tmp)


#%% DUMP DATA

MINI_SAVE_PATH = './data/GABA_synapses/detected_minis/'

for MinDet in mini_detectors:

    with open(MINI_SAVE_PATH + MinDet.name + '.pyc', 'wb') as f:
        pickle.dump(MinDet, f)
