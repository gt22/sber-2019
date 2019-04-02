# %%
from websocket import create_connection
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
import os
from excalibur import tower_map
from shishiga import submit

# %%


def init_websocket():
    ws = create_connection("ws://10.8.0.1:8080/alert")
    return ws


def alert(msg: str = 'Computation complete', ws: bool = True, usr: str = 'Admin') -> str:
    if not ws:
        return "Alerted"
    else:
        uadab = init_websocket()
        uadab.send(f"alert${usr}:{msg.replace(' ', ':')}")
        rep = uadab.recv()
        uadab.shutdown()
        return rep


def get_data(name):
    return f'data/dota2_{name}'


# %%
df = pd.read_csv(get_data("processed_train.csv"), index_col='id')
test_df = pd.read_csv(get_data("processed_test.csv"), index_col='id')
# %%
target = 'skilled'

cat_features = ['player_team', 'winner_team', 'pre_game_duration', 'first_blood_claimed',
                'hero_id', 'hero_pick_order', 'leaver_status', 'is_winner', 'party_players',
                'level', 'tower_kills', 'roshan_kills', 'radiant_tower_status', 'dire_tower_status',
                'dire_barracks_status', 'radiant_barracks_status'] \
               + [f'{p}_tower_{n}' for n in tower_map for p in ('p', 'e')]

numeric_features = [c for c in df.columns if
                    c not in cat_features and
                    c != target]

# %%
(X_train, X_test,
 y_train, y_test) = train_test_split(df.drop(target, axis=1), df[target], random_state=6741, test_size=0.3)


# %%


def cols_to_id(d, cc):
    i = d.columns.tolist()
    return [i.index(c) for c in cc]


cat_id = cols_to_id(X_train, cat_features)


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
    iterations=14000,
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


def make_reversed_submission(predict_func):
    pred = predict_func(test_df)
    subm = pd.DataFrame({'id': test_df.index, 'skilled': (~(pred.astype(bool))).astype(int)})
    subm.to_csv('reversed-submission.csv', header=True, index=False)


make_submission(model.predict)
make_reversed_submission(model.predict)
# %%
# submit('submission', '')
submit('reversed-submission', '')
