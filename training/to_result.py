import pandas as pd
from ast import literal_eval
result = pd.read_csv('./2024-10-31T153115_01/eval_result.csv')
behavior = pd.read_csv('behaviors_parsed.csv')
behavior = behavior.drop('clicked_news', axis=1)
# behavior = behavior[behavior['user_id'].isin(result['user_id'])]

result = result.merge(behavior, on='user_id', how='inner')
result['labels'] = None
for idx, row in result.iterrows():
    candidate = literal_eval(row['candidate_news'])
    true_set = literal_eval(row['clicked_candidate'])
    false_set = literal_eval(row['unclicked_candidate'])

    ground_true = [(1 if c in true_set else 0) for c in candidate]

    result.loc[idx, 'labels'] = str(ground_true)
result = result.drop(['num_clicked_news', 'num_candidate_news', 'clicked_candidate', 'unclicked_candidate'], axis=1)




result.to_csv('temp.csv')