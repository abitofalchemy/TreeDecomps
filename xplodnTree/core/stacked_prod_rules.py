import pandas as pd
from isomorph_interxn import listify_rhs
resutls = []

def collect_results(result):
	results.extend(result)
	# results.append(result)

# def probe_propensity_tofire( pddf, origG):


def stack_prod_rules_bygroup_into_list(prs_flst):
	stacked_df = pd.DataFrame()
	# pp = mp.Pool(processes=2)
	for j,f in enumerate(prs_flst):
		df = pd.read_csv(f, delimiter='\t', header='infer', dtype={})
		df.columns=['rnbr', 'lhs', 'rhs', 'prob']
		## df['lhs'] = df['lhs'].apply(lambda x: x.split(','))
		df['rhs'] = df['rhs'].apply(lambda x: listify_rhs(x))
		if j == 0:
			stacked_df = df
			continue
		stacked_df = pd.concat([stacked_df, df])
	return stacked_df
