# %%
from websocket import create_connection
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
import os
import json
import tqdm
from itertools import chain
from multiprocessing import Pool
from collections import defaultdict
import gc
# %%


def init_websocket():
    ws = create_connection("ws://10.8.0.1:8080/alert")
    return ws


def alert(msg: str='Computation complete', ws: bool=True, usr: str='Admin') -> str:
    if not ws:
        return "Alerted"
    else:
        uadab = init_websocket()
        uadab.send(f"alert${usr}:{msg.replace(' ', ':')}")
        rep = uadab.recv()
        uadab.shutdown()
        return rep


# %%


def get_data(name):
    return f'data/dota2_{name}'


# %%
df = pd.read_csv(get_data("skill_train.csv")).set_index('id')
test_df = pd.read_csv(get_data('skill_test.csv')).set_index('id')
# %%
heroes = pd.read_csv(get_data('heroes.csv')).set_index('hero_id')
abils = pd.read_csv(get_data('abilities.csv')).set_index('ability_id')
abils_behavior = set(chain.from_iterable(abils['behavior']
                                         .apply(lambda x: eval(x.replace('nan', 'None'))).values))
items = pd.read_csv(get_data('items.csv')).set_index('item_id')

with open(get_data('skill_train.jsonlines')) as f:
    json_line = json.loads(f.readline())
    for c in df.columns:
        json_line.pop(c, None)
# %%


def map_df(d, columns, m):
    for c in columns:
        d[c] = d[c].map(m)


def read_single_json():
    with open(get_data('skill_train.jsonlines')) as f:
        line = f.readline()
        rec = json.loads(line)
        for c in df.columns:
            rec.pop(c, None)
        return rec
# %%


def preprocess(d):
    team_map = {'radiant': 0, 'dire': 1}
    d.drop(d[d['winner_team'] == 'other'].index, inplace=True)
    map_df(d, ['player_team', 'winner_team'], team_map)
    d['is_winner'] = (d['player_team'] == d['winner_team']).astype(int)
    for b in abils_behavior:
        d[f'upgrade_beh_{b}'] = 0
    for b in abils['behavior']:
        d[f'upgrade_beh_list_{b}'] = 0
    for item in items.index:
        d[f'item_{item}'] = 0
    for hero in heroes.index:
        d[f'p_hero_{hero}'] = 0
        d[f'e_hero_{hero}'] = 0
    d['item_0'] = 0
    d['ult_time'] = 0


def z():
    return 0


def process_json_line(l):
    d = defaultdict(z)
    rec = json.loads(l)
    d['id'] = rec['id']
    lvl = rec['level_up_times']
    for b in rec['ability_upgrades']:
        d[f'upgrade_beh_list_{abils.loc[b, "behavior"]}'] += 1
        for bh in eval(abils.loc[b, 'behavior'].replace("nan", "None")):
            d[f'upgrade_beh_{bh}'] += 1
    # for item in rec['final_items']:
    #     d.loc[i, f'item_{item}'] += 1
    for item in rec['item_purchase_log']:
        d[f'item_{item["item_id"]}'] += 1
    p_team, e_team = ('radiant', 'dire') if rec['player_team'] == 'radiant' else ('dire', 'radiant')
    for p_hero in rec[f'{p_team}_heroes']:
        d[f'p_hero_{p_hero}'] += 1
    for e_hero in rec[f'{e_team}_heroes']:
        d[f'e_hero_{e_hero}'] += 1
    d['ult_time'] = -1 if len(lvl) < 9 else lvl[8]
    return d


def preprocess_json(d, f):
        print("Starting...")
        p = Pool()
        res = p.map(process_json_line, f)
        p.close()
        del p
        print("Wrapping to df")
        dd = pd.DataFrame(res).set_index('id')
        print("Copying to df")
        d = d.update(dd)
        print("Complete")
        return d


def preprocess_after_json(d):
    d['ult_percent'] = d['ult_time'] / d['duration']


def read_json(f):
    with open(f) as fi:
        ret = fi.readlines()
    return ret


# %%
preprocess(df)
jd = read_json(get_data('skill_train.jsonlines'))
preprocess_json(df, jd)
del jd
gc.collect()
preprocess_after_json(df)
alert("Train json complete")
# %%
preprocess(test_df)
jd = read_json(get_data('skill_test.jsonlines'))
preprocess_json(test_df, jd)
del jd
gc.collect()
preprocess_after_json(test_df)
alert("Test json complete")
# %%
df = pd.read_csv(get_data("processed_train.csv"), index_col='id')
test_df = pd.read_csv(get_data("processed_test.csv"), index_col='id')
# %%
target = 'skilled'

cat_features = ['player_team', 'winner_team', 'pre_game_duration', 'first_blood_claimed',
                'hero_id', 'hero_pick_order', 'leaver_status', 'is_winner']

questionable_features = ['party_players', 'level', 'tower_kills', 'roshan_kills',
                         'radiant_tower_status', 'dire_tower_status', 'dire_barracks_status', 'radiant_barracks_status']

numeric_features = [c for c in df.columns if
                    c not in cat_features and
                    c not in questionable_features and
                    c != target]

treat_questionable_as_cat = True

# %%
(X_train, X_test,
 y_train, y_test) = train_test_split(df.drop(target, axis=1), df[target], random_state=6741)


# %%


def cols_to_id(d, cc):
    i = d.columns.tolist()
    return [i.index(c) for c in cc]


cat_id = cols_to_id(X_train, cat_features)
if treat_questionable_as_cat:
    cat_id += cols_to_id(X_train, questionable_features)


# %%


def get_score_data(name):
    f = f'{name}_score.csv'
    if os.path.exists(f):
        return pd.read_csv(f).set_index('id')
    else:
        ret = pd.DataFrame({'id': [0], 'acc': [0]}).set_index('id')
        save_score_data(name, ret)
        return ret


def save_score_data(name, d):
    f = f'{name}_score.csv'
    d.index.name = 'id'
    d.to_csv(f, index=True, header=True)


def score_model(pred_func, name):
    s_data = get_score_data(name)
    acc_score = accuracy_score(y_test, pred_func(X_test))
    last_acc = s_data.loc[s_data.index[-1]]['acc']
    max_acc = s_data['acc'].max()
    print(f"Scoring {name}")
    print("Accuracy:", acc_score)
    print("Diff from last:", round(acc_score - last_acc, 4))
    print("Diff from max:", round(acc_score - max_acc, 4))

    if abs(acc_score - last_acc) > 1e-5:
        s_data = s_data.append({'acc': acc_score}, ignore_index=True)
        save_score_data(name, s_data)


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

model.fit(X_train, y_train, cat_features=cat_id, eval_set=(X_test, y_test))

score_model(model.predict, 'catboost')
# %%


def make_submission(predict_func):
    pred = predict_func(test_df)
    subm = pd.DataFrame({'id': test_df.index, 'skilled': pred.astype(int)})
    subm.to_csv('submission.csv', header=True, index=False)


# %%
make_submission(model.predict)
