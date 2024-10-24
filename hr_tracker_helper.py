import pandas as pd
from matplotlib import pyplot as plt


def filter_col(df, filter_col, desired_string):

	return df[df[filter_col].str.contains(desired_string)]


if __name__ == "__main__":

	
	data = pd.read_csv('hr_tracking_0415.csv')

	data.columns = ['index', 'subjectidnumber','tasktypedone','decision_made','real_r','real_c','bpm','timesteps',
	'number_of_timesteps','avg_hr','raw_max_hr','raw_min_hr','percent_max_hr','percent_min_hr','direction']

	appr_avoid = filter_col(data, 'tasktypedone', 'approach_avoid')
	appr_avoid.to_excel('approach_avoid_hr_0415.xlsx')
	
	moral = filter_col(data, 'tasktypedone', 'moral')
	moral.to_excel('moral_hr_0415.xlsx')

	social = filter_col(data, 'tasktypedone', 'social')
	social.to_excel('social_hr_0415.xlsx')

	prob = filter_col(data, 'tasktypedone', 'probability')
	prob.to_excel('probability_hr_0415.xlsx')

	'''
	read_file = pd.read_csv('probability_hr.csv')
	read_file.to_excel('probability_hr.xlsx', index=None, header=True)
	'''