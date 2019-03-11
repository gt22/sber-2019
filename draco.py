# %%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import xgboost
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier


# %%


def get_data(name):
    return f'data/dota2_{name}'


# %%
df = pd.read_csv(get_data("skill_train.csv")).set_index('id')


# %%


def map_df(d, columns, m):
    for c in columns:
        d[c] = d[c].map(m)


# %%


def preprocess(d):
    team_map = {'radiant': 0, 'dire': 1, 'other': 2}
    map_df(d, ['player_team', 'winner_team'], team_map)
    d['is_winner'] = (d['player_team'] == d['winner_team']).astype(int)


# %%
preprocess(df)
# %%
grp = df.groupby('is_winner')['skilled']

# %%
target = 'skilled'

cat_features = ['player_team', 'winner_team', 'pre_game_duration', 'first_blood_claimed',
                'hero_id', 'hero_pick_order', 'leaver_status', 'is_winner']

questionable_features = ['party_players', 'level', 'tower_kills', 'roshan_kills',
                         'radiant_tower_status', 'dire_tower_status']

numeric_features = [c for c in df.columns if
                    c not in cat_features and
                    c not in questionable_features and
                    c != target]

treat_questionable_as_cat = True

# %%
(X_train, X_test,
 y_train, y_test) = train_test_split(df.drop('skilled', axis=1), df['skilled'], random_state=6741)

# %%


def cols_to_id(d, cc):
    i = d.columns.tolist()
    return [i.index(c) for c in cc]


cat_id = cols_to_id(X_train, cat_features)
if treat_questionable_as_cat:
    cat_id += cols_to_id(X_train, questionable_features)

# %%

model = CatBoostClassifier(
    iterations=None,
    learning_rate=None,
    depth=None,
    eval_metric='Accuracy',
    random_seed=6741,
    use_best_model=True,
    verbose=True,
)

# %%
model.fit(X_train, y_train, cat_features=cat_id, eval_set=(X_test, y_test))
# %%
tdf = None


def make_submission(predict_func):
    global tdf
    tdf = test_df = pd.read_csv(get_data('skill_test.csv'))
    preprocess(test_df)
    pred = predict_func(test_df.drop('id', axis=1))
    subm = pd.DataFrame({'id': test_df.id, 'skilled': pred.astype(int)})
    subm.to_csv('submission.csv', header=True, index=False)


# %%
make_submission(model.predict)
