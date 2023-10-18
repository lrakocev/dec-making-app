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

################################
##
## helper functions
##
################################

# TODO: this function is currently how i'm modifying the length of the reading box
# this is sort of hack-y so there's definitely a better way to do this
def get_added_q_len(x_coords):

	grad_x = np.gradient(x_coords)
	max_is = [i for i in range(round(.75*len(grad_x))) if grad_x[i] < -15]
	added_q_len = max(len(max_is)-1, 0)

	return added_q_len

def get_reading_box(min_x, max_y, added_q_len = 0):
	return (-5, max_y, 97, 12+added_q_len)

def get_reaction_box(min_x, max_y, added_q_len = 0):
	return (35, max_y-added_q_len-14, 20, 7)

def get_dec_box(min_x, max_y, added_q_len = 0):
	return (40, max_y-added_q_len-21, 10, 15) 

def get_coords(gaze_df, col_name):

	gaze_list = gaze_df[col_name].tolist()

	# x100 just to scale the points - not strictly necessary
	x_coords = [100*pt[0] for pt in gaze_list]
	y_coords = [-100*pt[1] for pt in gaze_list]

	return x_coords, y_coords

def rect_contains(rect, pt):
	logic = rect[0] < pt[0] < rect[0]+rect[2] and rect[1] > pt[1] > rect[1]-rect[3]
	return logic

def gaze_in_box(x_coords, y_coords, added_q_len=0):

	print("xcoords", x_coords)
	print("ycoords", y_coords)

	gaze_list = zip(x_coords, y_coords)

	max_y = np.mean(sorted(y_coords)[-20:])
	min_x = min(x_coords)

	read_rect = get_reading_box(min_x,max_y,added_q_len)
	react_rect = get_reaction_box(min_x,max_y,added_q_len)
	dec_rect = get_dec_box(min_x,max_y,added_q_len)

	color_list = []
	for pt in gaze_list:
		if rect_contains(read_rect, pt):
			color_list.append('r')
		elif rect_contains(react_rect, pt):
			color_list.append('g')
		elif rect_contains(dec_rect, pt):
			color_list.append('y')
		else:
			color_list.append('b')

	return color_list

def get_outliers(y_coords,x_coords = []):

	if len(x_coords) > 0:
		grads = np.gradient(y_coords, x_coords)
	else:
		grads = np.gradient(y_coords)

	avg_diff = np.mean(grads)
	std_diff = min(50,np.std(grads))
	thresh = avg_diff + 2*std_diff

	problem_is = [i-1 for i in range(len(grads)) if np.abs(grads[i]) >= thresh]

	return problem_is

def line_up_data(gaze_df, eye):

	coord_col_name = eye + "_gaze_point_on_display_area"
	pupil_col_name = eye + "_pupil_diameter"

	x_coords, y_coords = get_coords(gaze_df, coord_col_name)
	outlier_idxs = get_outliers(y_coords,x_coords)

	pupil_list = gaze_df[pupil_col_name].tolist()
	pupil_outlier_idxs = get_outliers(pupil_list)

	tot_outlier_idxs = set(outlier_idxs + pupil_outlier_idxs)

	x_coords = [x_coords[i] for i in range(len(x_coords)) if i not in tot_outlier_idxs]
	y_coords = [y_coords[i] for i in range(len(y_coords)) if i not in tot_outlier_idxs]
	pupil_list = [pupil_list[i] for i in range(len(pupil_list)) if i not in tot_outlier_idxs]

	return x_coords, y_coords, pupil_list

# this function isn't used anywhere
def rolling_window_avgs(y_coords, w_size = 5):

	velocities = np.gradient(y_coords)

	avg_velocities = []
	for i in range(len(velocities) - w_size + 1):
		window = velocities[i:i+w_size]
		avg_vel = np.mean(window)
		avg_velocities.append(avg_vel)

	return avg_velocities

################################
##
## plotting
##
################################

def plot_data(x_coords, y_coords, pupil_list, colors, end_read, max_y, min_x, added_q_len):

	top_corner_x_read, top_corner_y_read, width_read, height_read = get_reading_box(min_x, max_y, added_q_len)
	top_corner_x_react, top_corner_y_react, width_react, height_react = get_reaction_box(min_x, max_y, added_q_len)
	top_corner_x_dec, top_corner_y_dec, width_dec, height_dec = get_dec_box(min_x, max_y, added_q_len)

	read_rect = patches.Rectangle((top_corner_x_read, top_corner_y_read-height_read), width_read, height_read, linewidth=1, edgecolor='r', facecolor='none')
	dec_rect = patches.Rectangle((top_corner_x_react, top_corner_y_react-height_react), width_react, height_react, linewidth=1, edgecolor='g', facecolor='none')
	submit_rect = patches.Rectangle((top_corner_x_dec, top_corner_y_dec-height_dec), width_dec, height_dec, linewidth=1, edgecolor='y', facecolor='none')

	x = [i for i in range(len(x_coords))]

	e_drops, poly_y = get_engagement_drops(pupil_list)

	plt.figure()
	ax1 = plt.subplot(211)
	plt.title('position')
	plt.plot(x_coords, y_coords)
	ax1.add_patch(read_rect)
	ax1.add_patch(dec_rect)
	ax1.add_patch(submit_rect)

	plt.subplot(212)
	plt.title('pupil diameter')
	plt.scatter(x,pupil_list,c=colors)
	plt.plot(pupil_list)
	plt.plot(poly_y)
	for i in e_drops:
		plt.plot((i,i), (min(pupil_list), max(pupil_list)),'k')
	plt.plot((end_read, end_read), (min(pupil_list), max(pupil_list)), 'm')

def post_reading_data(x_coords, y_coords, pupil_list, added_q_len=0):

	colors = gaze_in_box(x_coords, y_coords, added_q_len)
	end_read = get_end_reading(x_coords, y_coords, added_q_len)
	return x_coords[end_read:], y_coords[end_read:],pupil_list[end_read:], colors[end_read:]

def plot_init(gaze_df, eye, added_q_len=0):

	x_coords, y_coords, pupil_list = line_up_data(gaze_df, eye)

	colors = gaze_in_box(x_coords, y_coords, added_q_len)
	end_read = get_end_reading(x_coords, y_coords, added_q_len)
	max_y = np.mean(sorted(y_coords)[-20:]) 
	min_x = min(x_coords)

	plot_data(x_coords, y_coords, pupil_list, colors, end_read, max_y, min_x, added_q_len)

def plot_dec_period(gaze_df, eye):

	x_coords_orig, y_coords_orig, pupil_list_orig = line_up_data(gaze_df, eye)
	
	grad_x = np.gradient(x_coords_orig)
	max_is = [i for i in range(len(grad_x)) if grad_x[i] < -10]
	added_q_len = 2*len(max_is)

	max_y = np.mean(sorted(y_coords_orig)[-20:]) 
	min_x = min(x_coords_orig)
	x_coords, y_coords, pupil_list, colors = post_reading_data(x_coords_orig, y_coords_orig, pupil_list_orig, added_q_len)
	plot_data(x_coords, y_coords, pupil_list, colors, 0, max_y, min_x, added_q_len)

def two_eyes(gaze_df):

	left = gaze_df['left_gaze_point_on_display_area'].tolist()
	right = gaze_df['right_gaze_point_on_display_area'].tolist()

	left_pupil = gaze_df['left_pupil_diameter'].tolist()
	right_pupil = gaze_df['right_pupil_diameter'].tolist()

	left_x = [pt[0] for pt in left]
	left_y = [-pt[1] for pt in left]
	right_x = [pt[0] for pt in right]
	right_y = [-pt[1] for pt in right]

	x_cov = np.cov((left_x, right_x))
	y_cov = np.cov((left_y, right_y))

	plt.figure()
	plt.subplot(411)
	plt.plot(left_x, left_y, 'b')
	plt.plot(right_x, right_y, 'r')
	plt.subplot(412)
	plt.plot(left_x, 'b')
	plt.plot(right_x, 'r')
	plt.title('x cov: ' + str(round(x_cov[0][1], 4)))
	plt.subplot(413)
	plt.plot(left_y, 'b')
	plt.plot(right_y, 'r')
	plt.title("y cov: " + str(round(y_cov[0][1],4)))
	plt.subplot(414)
	plt.plot(left_pupil, 'b')
	plt.plot(right_pupil, 'r')
	#plt.show()

	return x_cov, y_cov


################################
##
## features
##
################################

def get_end_reading(x_coords, y_coords, added_q_len):

	colors = gaze_in_box(x_coords, y_coords, added_q_len)
	w = 15

	for i in range(w,len(colors) - w + 1):
		window = colors[i:i+w]
		num_red = window.count('r')
		num_blue = window.count('b')
		prop_red = num_red / len(window)
		prop_blue = num_blue / len(window)
		if prop_red < .1 and prop_blue < .1:
			return i

def get_decision_timing(x_coords, y_coords, added_q_len):

	end_of_reading = get_end_reading(x_coords, y_coords, added_q_len)
	total_length = len(x_coords)

	return total_length - end_of_reading

# TODO: i'm using polyfit as a way to smooth the engagement data, but there's definitely a better function for it
def get_engagement_drops(pupil_list):

	x = [i for i in range(0,len(pupil_list))]

	pupil_list_poly = np.polyfit(x, pupil_list, 100)
	poly_y =  np.poly1d(pupil_list_poly)(x)
	max_ys = argrelextrema(poly_y, np.greater)

	return max_ys[0], poly_y

def eye_on_targets(x_coords, y_coords, added_q_len=0):

	eye_placement = gaze_in_box(x_coords, y_coords, added_q_len=0)

	on_dec_box = [i for i in range(len(eye_placement)) if eye_placement[i] == 'g']
	on_submit_box = [i for i in range(len(eye_placement)) if eye_placement[i] == 'y']

	return on_dec_box, on_submit_box


def get_reaction_time(x_coords, y_coords, added_q_len):
	# how quickly do you move from reading -> decision

	end_read = get_end_reading(x_coords, y_coords, added_q_len)

	on_dec_box, on_submit_box = eye_on_targets(x_coords, y_coords, added_q_len)

	first_on_dec_box = [i for i in on_dec_box if i >= end_read]
	first_on_submit_box = [i for i in on_submit_box if i >= end_read]

	btwn_read_n_decide = first_on_dec_box[0] - end_read
	btwn_read_n_submit = first_on_submit_box[0] - end_read

	return btwn_read_n_decide, btwn_read_n_submit


################################
##
## setting up the df
##
################################

def convert_gaze_data_to_df(trial_df, r, c):

	sub_df = trial_df[(trial_df['real_r'] == r) & (trial_df['real_c'] == c)]

	gaze_data = sub_df.iloc[0]['eye_tracker_data']
	gaze_data = gaze_data.replace('nan', '-9999')
	gaze_data_dict = ast.literal_eval(gaze_data)
	gaze_data = gaze_data_dict['gaze_data']

	gaze_df = pd.DataFrame.from_dict(gaze_data)
	gaze_df = gaze_df[(gaze_df['left_gaze_point_on_display_area'] != (-9999,-9999)) & (gaze_df['right_gaze_point_on_display_area'] != (-9999,-9999))]
	gaze_df = gaze_df[(gaze_df['left_gaze_point_on_display_area'] != (-9999,-9999)) & (gaze_df['right_gaze_point_on_display_area'] != (-9999,-9999))]

	# clean by validity 

	gaze_df = gaze_df[(gaze_df['left_gaze_point_validity'] == 1) & (gaze_df['right_gaze_point_validity']  == 1)]
	gaze_df = gaze_df[(gaze_df['left_pupil_validity'] == 1) & (gaze_df['right_pupil_validity']  == 1)]

	gaze_df['left_x_coords'] = gaze_df['left_gaze_point_on_display_area'].apply(lambda x: x[0])
	gaze_df['left_y_coords'] = gaze_df['left_gaze_point_on_display_area'].apply(lambda x: x[1])

	gaze_df['real_r'] = r
	gaze_df['real_c'] = c

	return gaze_df

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

	participant_id = sys.argv[1].rstrip()
	tasktypedone= sys.argv[2].rstrip()
	init_or_dec = sys.argv[3].rstrip()
	eye = "left"

	checking_eyes = False

	conn = psycopg2.connect(database='live_database', host='10.10.21.128', user='postgres', port='5432', password='1234')
	trial_cursor = conn.cursor()

	qry = f"SELECT tasktypedone,reward_prefs,cost_prefs,reward_level,cost_level,decision_made,eye_tracker_data from human_dec_making_table_utep where subjectidnumber = '{participant_id}' and tasktypedone='{tasktypedone}'"

	trial_cursor.execute(qry)
	trial_table = trial_cursor.fetchall()
	trial_df = pd.DataFrame(trial_table)
	trial_df.columns = ['tasktypedone','reward_prefs','cost_prefs','reward_level','cost_level','decision_made','eye_tracker_data']

	trial_df = trial_df[(trial_df['reward_level'] != '7') & (trial_df['cost_level'] != '7')]

	trial_df['real_r'] = trial_df.apply(lambda x: real_rc(x['reward_level'],x['reward_prefs']), axis=1)
	trial_df['real_c'] = trial_df.apply(lambda x: real_rc(x['cost_level'],x['cost_prefs']), axis=1)
	
	dfs = [trial_df]
	for r in range(1, 2): #5):
		for c in range(1,2):#5):
			task = "_".join(tasktypedone.split("/"))
			tasktype = tasktypedone.split("/")[1]
			gaze_df = convert_gaze_data_to_df(trial_df, r, c)
			dfs.append(gaze_df)

			if checking_eyes:
				two_eyes(gaze_df)
				pathname = f"eyetracking_maps/{participant_id}/eye_comparison/{task}"
				if not os.path.exists(pathname):
					os.makedirs(pathname)
				figname = f"{pathname}/{r}_{c}"
				plt.savefig(figname)

			else:
				if init_or_dec == "init":
					plot_init(gaze_df, eye)
				else:
					plot_dec_period(gaze_df, eye)

				pathname = f"eyetracking_maps/{participant_id}/single_eye_3/{task}"
				if not os.path.exists(pathname):
					os.makedirs(pathname)
				figname = f"{pathname}/{init_or_dec}_{r}_{c}"
				plt.savefig(figname)

	new_df = pd.concat(dfs,keys=['real_r','real_c'])

	## TODO: create a DF with all of the features easily accessible -- there's a bug i'm too lazy to sort out. 
	## might be better to pass in the gaze data as it already is (tuples) instead of dividing into x and y coords 
	## which is done in(convert_gaze_data_to_df)
	
	'''
	new_df['added_q_len'] = new_df.apply(lambda x: get_added_q_len(x['left_x_coords']), axis=1)
	new_df['end_of_reading'] = new_df.apply(lambda x: get_end_reading(x['left_x_coords'], x['left_y_coords'], x['added_q_len']), axis=1)
	new_df['engagement_drops'] = new_df.apply(lambda x: get_engagement_drops(x['left_pupil_diameter'])[1], axis=1)
	new_df['decision_timing'] = new_df.apply(lambda x: get_decision_timing(x['left_x_coords'], x['left_y_coords'], x['added_q_len']), axis=1)
	new_df['eye_on_slider'] = new_df.apply(lambda x: eye_on_targets(x['left_x_coords'], x['left_y_coords'], x['added_q_len'])[0], axis=1)
	new_df['eye_on_submit'] = new_df.apply(lambda x: eye_on_targets(x['left_x_coords'], x['left_y_coords'], x['added_q_len'])[1], axis=1)
	new_df['reaction_time'] = new_df.apply(lambda x: get_reaction_time(x['left_x_coords'], x['left_y_coords'], x['added_q_len'])[1], axis=1)
	'''