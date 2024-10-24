import ast
from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib import patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import statistics
from collections import Counter
from scipy.sparse import coo_matrix
import psycopg2
import pandas as pd
import itertools
import copy
from scipy import stats
from scipy.misc import derivative
import time
import sys
import math
import numpy
import json
import scipy.stats
from scipy.spatial.distance import pdist, squareform
import create_map
import os
from scipy.signal import argrelextrema, find_peaks
from numpy.polynomial import Polynomial as P
from numpy.polynomial import Chebyshev as T
from scipy.signal import savgol_filter
import math
from datetime import datetime as dt
import time

################################
##
## features
##
################################

def df_additions(hr_df):

	hr_df = add_num_timesteps(hr_df)
	hr_df = avg_hr(hr_df)
	hr_df = hr_max_raw(hr_df)
	hr_df = hr_min_raw(hr_df)
	hr_df = hr_max_percent(hr_df)
	hr_df = hr_min_percent(hr_df)
	hr_df = direction(hr_df)

	return hr_df


def direction(hr_df):

	def helper_func(timesteps, bpms):

		time_to_bpm = dict(zip(timesteps, bpms))

		max_time = max(time_to_bpm, key=time_to_bpm.get)
		min_time = min(time_to_bpm, key=time_to_bpm.get)

		if max_time > min_time:
			return 1
		if min_time > max_time: 
			return -1
		return 0

	hr_df['direction'] = hr_df.apply(lambda x: helper_func(x['timesteps'],x['bpm']), axis=1)
	return hr_df 

def hr_min_percent(hr_df):

	hr_df['percent_min_hr'] = hr_df.apply(lambda x: 100*(x['raw_min_hr'] - x['avg_hr'])/x['avg_hr'],axis=1)
	return hr_df

def hr_max_percent(hr_df):

	hr_df['percent_max_hr'] = hr_df.apply(lambda x: 100*(x['raw_max_hr'] - x['avg_hr'])/x['avg_hr'],axis=1)
	return hr_df

def hr_min_raw(hr_df):

	hr_df['raw_min_hr'] = hr_df.apply(lambda x: min(x['bpm']),axis=1)
	return hr_df

def hr_max_raw(hr_df):

	hr_df['raw_max_hr'] = hr_df.apply(lambda x: max(x['bpm']),axis=1)
	return hr_df

def avg_hr(hr_df):

	hr_df['avg_hr'] = hr_df.apply(lambda x: np.mean(x['bpm']),axis=1)
	return hr_df

def add_num_timesteps(hr_df):

	hr_df['number_of_timesteps'] = hr_df.apply(lambda x: len(x['timesteps']), axis=1)
	return hr_df


################################
##
## plotting
##
################################

def plot_raw(hr_df):

	timesteps = hr_df['timesteps'].tolist()[0]
	bpm = hr_df['bpm'].tolist()[0]

	plt.figure()
	plt.title('hr vs time (in seconds)')
	plt.plot(timesteps, bpm)

################################
##
## setting up the df
##
################################


def convert_hr_data_to_df(trial_df, r, c):

	sub_df = trial_df[(trial_df['real_r'] == r) & (trial_df['real_c'] == c)]

	hr_data = sub_df.iloc[0]['heart_rate_data']
	hr_data_dicts = ast.literal_eval(hr_data)

	hr = []
	times = []
	for d in hr_data_dicts:
		hr.append(d["hr"])

		t = d['time']
		# Wed Sep 13 15:45:27.000000 2023 UTC
		timestep = t.split(".")[0]
		datetime = dt.strptime(timestep, "%a %b %d %H:%M:%S")
		times.append(datetime)

	t_0 = times[0]
	timesteps = []
	for t in times:
		time_since_beginning = (t-t_0).total_seconds()
		timesteps.append(time_since_beginning)

	hr_df = pd.DataFrame(columns=['bpm','timesteps','real_r','real_c'])

	hr_df.at[1,'bpm'] = hr
	hr_df.at[1,'timesteps'] = timesteps
	hr_df['real_r'] = r
	hr_df['real_c'] = c

	return hr_df

def get_pref_map(pref_dict):
	chosen_pref_dict = create_map.choose_prefs_new(pref_dict)

	chosen_prefs = sorted(chosen_pref_dict, key=chosen_pref_dict.get)
	choice_range = list(range(1,5))
	pref_map = dict(zip(chosen_prefs, choice_range))
	return pref_map

def real_rc(row_lvl, row_prefs):

	row_lvl = int(row_lvl)
	pref_dict = ast.literal_eval(row_prefs)
	pref_map = get_pref_map(pref_dict)

	return pref_map[row_lvl]

if __name__ == "__main__":

	want_plot = 0
	participant_id = sys.argv[1].rstrip()
	tasktypedone = sys.argv[2].rstrip()
	tasktypedone = tasktypedone.split("Git")[1]

	conn = psycopg2.connect(database='live_database', host='129.108.49.137', user='postgres', port='5432', password='1234')
	trial_cursor = conn.cursor()

	qry = f"SELECT subjectidnumber,tasktypedone,reward_prefs,cost_prefs,reward_level,cost_level,decision_made,heart_rate_data from human_dec_making_table_utep where subjectidnumber = '{participant_id}' and tasktypedone = '{tasktypedone}'"

	trial_cursor.execute(qry)
	trial_table = trial_cursor.fetchall()
	trial_df = pd.DataFrame(trial_table)

	trial_df.columns = ['subjectidnumber','tasktypedone','reward_prefs','cost_prefs','reward_level','cost_level','decision_made','heart_rate_data']

	trial_df = trial_df[(trial_df['reward_level'] != '7') & (trial_df['cost_level'] != '7')]
	trial_df = trial_df[(trial_df['reward_level'] != '0') & (trial_df['cost_level'] != '0')]

	trial_df['real_r'] = trial_df.apply(lambda x: real_rc(x['reward_level'],x['reward_prefs']), axis=1)
	trial_df['real_c'] = trial_df.apply(lambda x: real_rc(x['cost_level'],x['cost_prefs']), axis=1)

	dfs = []
	for r in range(1, 5):
		for c in range(1,5):
			task = "_".join(tasktypedone.split("/"))
			tasktype = tasktypedone.split("/")[1]

			try:
				hr_df = convert_hr_data_to_df(trial_df, r, c)
				if want_plot:
					plot_raw(hr_df)
					pathname = f"hr_maps/{participant_id}/{task}"
					if not os.path.exists(pathname):
						os.makedirs(pathname)
					figname = f"{pathname}/{r}_{c}.pdf"
					plt.savefig(figname)
				hr_df = df_additions(hr_df)
				dfs.append(hr_df)
			except:
				continue

	intermed_df = pd.concat(dfs)
	trial_df = trial_df.drop(['reward_prefs','cost_prefs','reward_level','cost_level','heart_rate_data'], axis=1)
	new_df = pd.merge(trial_df, intermed_df, how='outer',on=['real_r','real_c'])
	
	new_df.to_csv('hr_tracking_0415.csv', mode='a')
