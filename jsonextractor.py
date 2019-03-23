import pandas as pd
import numpy as np
import json
from itertools import chain
from multiprocessing import Pool
from collections import defaultdict
import gc
from typing import List


def get_data(name):
    return f'data/dota2_{name}'


heroes = pd.read_csv(get_data('heroes.csv')).set_index('hero_id')
abils = pd.read_csv(get_data('abilities.csv')).set_index('ability_id')
abils_behavior = set(chain.from_iterable(abils['behavior']
                                         .apply(lambda x: eval(x.replace('nan', 'None'))).values))
items = pd.read_csv(get_data('items.csv')).set_index('item_id')


class ProcessingModule:

    def process(self, l: dict, d: dict):
        pass

    def get_cols(self) -> List[str]:
        return []


class Pipeline(ProcessingModule):
    modules: List[ProcessingModule]

    def __init__(self, *args):
        self.modules = args

    def process(self, l: dict, d: dict):
        for m in self.modules:
            m.process(l, d)

    def get_cols(self):
        return sum((m.get_cols() for m in self.modules), [])


class UltTime(ProcessingModule):

    def process(self, l: dict, d: dict):
        lvl = l['level_up_times']
        d['ult_time'] = -1 if len(lvl) < 9 else lvl[8]
        d['ult_percent'] = d['ult_time'] / l['duration'] if d['ult_time'] > 0 else -1

    def get_cols(self):
        return ['ult_time', 'ult_percent']


class AbilityUpgrades(ProcessingModule):

    def process(self, l: dict, d: dict):
        for b in l['ability_upgrades']:
            d[f'upgrade_beh_list_{abils.loc[b, "behavior"]}'] += 1
            for bh in eval(abils.loc[b, 'behavior'].replace("nan", "None")):
                d[f'upgrade_beh_{bh}'] += 1

    def get_cols(self):
        return [f'upgrade_beh_list_{b}' for b in abils['behavior']] + \
               [f'upgrade_beh_{b}' for b in abils_behavior]


class Items(ProcessingModule):

    def process(self, l: dict, d: dict):
        for item in l['item_purchase_log']:
            d[f'item_{item["item_id"]}'] += 1

    def get_cols(self):
        return [f'item_{i}' for i in items.index]


class Heroes(ProcessingModule):

    def process(self, l: dict, d: dict):
        p_team, e_team = ('radiant', 'dire') if l['player_team'] == 'radiant' else ('dire', 'radiant')
        for p_hero in l[f'{p_team}_heroes']:
            d[f'p_hero_{p_hero}'] += 1
        for e_hero in l[f'{e_team}_heroes']:
            d[f'e_hero_{e_hero}'] += 1

    def get_cols(self):
        return [f'p_hero_{i}' for i in items.index] + [f'e_hero_{i}' for i in items.index]


class Series(ProcessingModule):

    def process(self, l: dict, d: dict):
        p_team, e_team = ('radiant', 'dire') if l['player_team'] == 'radiant' else ('dire', 'radiant')
        ser = l['series']
        d['player_gold_mean'] = np.mean(ser['player_gold'])
        d['player_team_gold_mean'] = np.mean(ser[f'{p_team}_gold'])
        d['enemy_team_gold_mean'] = np.mean(ser[f'{e_team}_gold'])

    def get_cols(self):
        return ['player_gold_mean', 'enemy_team_gold_mean', 'player_team_gold_mean']


class DamageTargets(ProcessingModule):

    def process(self, l: dict, d: dict):
        d['dt_sum'] = sum(l['damage_targets'].values())

    def get_cols(self):
        return ['dt_sum']


def z():
    return 0


def process_json_line(m: ProcessingModule, l: str):
    d = defaultdict(z)
    rec = json.loads(l)
    d['id'] = rec['id']
    m.process(rec, d)
    return d


def preprocess_json(d: pd.DataFrame, data: List[str], m: ProcessingModule):
    print("Starting...")
    p = Pool()
    res = p.starmap(process_json_line, [(m, x) for x in data])
    p.close()
    del p
    print("Adding cols to df")
    for c in m.get_cols():
        d[c] = 0
    print("Wrapping to df")
    dd = pd.DataFrame(res).set_index('id')
    print("Copying to df")
    d = d.update(dd)
    print("Complete")
    return d


def map_df(d, columns, m):
    for c in columns:
        d[c] = d[c].map(m)


def preprocess(d):
    team_map = {'radiant': 0, 'dire': 1}
    d.drop(d[d['winner_team'] == 'other'].index, inplace=True)
    map_df(d, ['player_team', 'winner_team'], team_map)
    d['is_winner'] = (d['player_team'] == d['winner_team']).astype(int)


def read_json(f):
    with open(f) as fi:
        ret = fi.readlines()
    return ret


print("Reading data")
df = pd.read_csv(get_data("skill_train.csv")).set_index('id')
test_df = pd.read_csv(get_data('skill_test.csv')).set_index('id')

module = Pipeline(
    UltTime(),
    AbilityUpgrades(),
    Items(),
    Heroes(),
    Series(),
    DamageTargets()
)

print("Preprocessing data")
preprocess(df)
preprocess(test_df)

print("Reading train json")
jd = read_json(get_data('skill_train.jsonlines'))
print("Processing train json")
preprocess_json(df, jd, module)
del jd
gc.collect()
print()
print("Reading test json")
jd = read_json(get_data('skill_test.jsonlines'))
print("Processing test json")
preprocess_json(test_df, jd, module)
del jd
gc.collect()

print("Saving")
df.to_csv(get_data('processed_train.csv'))
test_df.to_csv(get_data('processed_test.csv'))