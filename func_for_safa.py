from itertools import product

def write_questions(reward_pref_file, cost_pref_file, output_file):

	rew = open(reward_pref_file, 'r')
	cost = open(cost_pref_file, 'r')

	rew_dict = {}
	cost_dict = {}
	for i in range(1,7):
		rew_line = rew.readline().split(")")[1].strip()
		cost_line = cost.readline().split(")")[1].strip()
		rew_dict[i] = rew_line
		cost_dict[i] = cost_line

	t = [list(range(1,7)) for _ in range(2)]
	index_combos = list(product(*t))
	print(index_combos)
	qs = []
	with open(output_file, 'w') as fp:
		for ind in index_combos:
			r_key, c_key = ind
			rew = rew_dict[r_key]
			cost = cost_dict[c_key]
			q = f"Would you like to {rew} but {cost}? (R{r_key},C{c_key}) \n"
			fp.write(q)

#example
write_questions('stories/Story 12/pref_reward.txt','stories/Story 12/pref_cost.txt', 'stories/Story 12/q2.txt')