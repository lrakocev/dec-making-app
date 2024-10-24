import pandas as pd
from matplotlib import pyplot as plt


def filter_col(df, filter_col, desired_string):

	return df[df[filter_col].str.contains(desired_string)]


if __name__ == "__main__":

	
	data = pd.read_csv('eye_tracking_0415.csv')

	data.columns = ['index', 'subjectidnumber','tasktypedone','decision_made',
	'left_x_coords', 'left_y_coords', 'left_pupil_diameter', 'real_r', 'real_c',
	'added_q_len', 'end_of_reading', 'order', 'num_maxes', 'num_mins','decision_timing', 
	'big_ups', 'big_dips', 'avg_eng', 'eng_before_dec', 'eng_around_dec', 'eng_after_dec']

	social = filter_col(data, 'tasktypedone', 'social')
	social.to_excel('social_eyetracking_0415.xlsx')

	prob = filter_col(data, 'tasktypedone', 'probability')
	prob.to_excel('probability_eyetracking_0415.xlsx')

	
	appr_avoid = filter_col(data, 'tasktypedone', 'approach_avoid')
	appr_avoid.to_excel('approach_avoid_eyetracking_0415.xlsx')
	
	moral = filter_col(data, 'tasktypedone', 'moral')
	moral.to_excel('moral_eyetracking_0415.xlsx')
	

	'''
	read_file = pd.read_csv('probability_eyetracking.csv')
	read_file.to_excel('probability_eyetracking.xlsx', index=None, header=True)
	'''