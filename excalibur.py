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


def parse_list(d):
    return eval(d.replace('nan', 'None'))


def parse_data_list(col):
    return set(chain.from_iterable(col.apply(parse_list).values))


heroes = pd.read_csv(get_data('heroes.csv')).set_index('hero_id')
roles = parse_data_list(heroes['roles'])

abils = pd.read_csv(get_data('abilities.csv')).set_index('ability_id')
behaviors = parse_data_list(abils['behavior'])

items = pd.read_csv(get_data('items.csv')).set_index('item_id')

items_score = pd.read_csv(get_data('items_score.csv')).set_index('id')

tower_map = ['t1', 't2', 't3', 'm1', 'm2', 'm3', 'b1', 'b2', 'b3', 'at', 'ab']


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


def get_level(l: dict, d: dict, lvl: int):
    lup = l['level_up_times']
    d[f'level_{lvl}'] = -1 if len(lup) < (lvl - 1) else lup[lvl - 2]
    d[f'level_{lvl}_percent'] = -1 if d[f'level_{lvl}'] == -1 else d[f'level_{lvl}'] / l['duration']


def get_level_cols(lvl: int):
    return [f'level_{lvl}', f'level_{lvl}_percent']


class UltTime(ProcessingModule):
    levels = [6, 12, 18, 10, 20, 30]

    def process(self, l: dict, d: dict):
        for lvl in self.levels:
            get_level(l, d, lvl)

    def get_cols(self):
        return sum((get_level_cols(i) for i in self.levels), [])


class AbilityUpgrades(ProcessingModule):

    def process(self, l: dict, d: dict):
        for b in l['ability_upgrades']:
            d[f'upgrade_beh_list_{abils.loc[b, "behavior"]}'] += 1
            for bh in parse_list(abils.loc[b, 'behavior']):
                d[f'upgrade_beh_{bh}'] += 1

    def get_cols(self):
        return [f'upgrade_beh_list_{b}' for b in abils['behavior']] + \
               [f'upgrade_beh_{b}' for b in behaviors]


class Items(ProcessingModule):

    def process(self, l: dict, d: dict):
        sum_cost = 0
        for item in l['item_purchase_log']:
            iid = item['item_id']
            iobj = items.loc[iid]
            d[f'item_{iid}'] += 1
            d[f'item_type_{iobj["qual"]}'] += 1
            sum_cost += iobj['cost']
            if iid in items_score.index:
                for c in items_score.columns:
                    d[c] += items_score.loc[iid, c]
        d['sum_cost'] = sum_cost

    def get_cols(self):
        return [f'item_{i}' for i in items.index] + \
               [f'item_type_{q}' for q in items['qual'].unique()] + \
               ['sum_cost'] + items_score.columns.tolist()


class Heroes(ProcessingModule):

    def process(self, l: dict, d: dict):
        p_team, e_team = ('radiant', 'dire') if l['player_team'] == 'radiant' else ('dire', 'radiant')
        for (pref, team) in (('p', p_team), ('e', e_team)):
            for hero in l[f'{team}_heroes']:
                d[f'{pref}_hero_{hero}'] += 1
                for role in parse_list(heroes.loc[hero, 'roles']):
                    d[f'{pref}_role_{role}'] += 1

    def get_cols(self):
        return [f'{p}_hero_{i}' for i in heroes.index for p in ('p', 'e')] + \
               [f'{p}_role_{i}' for i in roles for p in ('p', 'e')]


class Series(ProcessingModule):

    def process(self, l: dict, d: dict):
        p_team, e_team = ('radiant', 'dire') if l['player_team'] == 'radiant' else ('dire', 'radiant')
        ser = l['series']
        for p, t in (('p', p_team), ('e', e_team)):
            g = ser[f'{t}_gold']
            d[f'{p}_gold_mean'] = np.mean(g)
            d[f'{p}_gold_max'] = g[-1]

        pg = ser['player_gold']
        d['player_gold_mean'] = np.mean(pg)
        d['player_gold_max'] = pg[-1]
        d['player_gold_contrib_mean'] = d['player_gold_mean'] / d['p_gold_mean']
        d['player_gold_contrib_max'] = d['player_gold_max'] / d['p_gold_max']

    def get_cols(self):
        return ['player_gold_mean', 'player_gold_contrib_max', 'player_gold_contrib_mean', 'player_gold_max',
                'e_gold_mean', 'p_gold_mean', 'e_gold_max', 'p_gold_max']


class DamageTargets(ProcessingModule):

    def process(self, l: dict, d: dict):
        d['dt_sum'] = sum(l['damage_targets'].values())

    def get_cols(self):
        return ['dt_sum']


class TowerStatus(ProcessingModule):

    def process(self, l: dict, d: dict):
        p_team, e_team = ('radiant', 'dire') if l['player_team'] == 'radiant' else ('dire', 'radiant')
        p_status, e_status = (l[f'{t}_tower_status'] for t in (p_team, e_team))
        for p, s in (('p', p_status), ('e', e_status)):
            for i in range(11):
                d[f'{p}_tower_{tower_map[i]}'] = int((s & (1 << i)) != 0)

    def get_cols(self):
        return [f'{p}_tower_{n}' for n in tower_map for p in ('p', 'e')]


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


if __name__ == '__main__':
    print("Reading data")
    df = pd.read_csv(get_data("skill_train.csv")).set_index('id')
    test_df = pd.read_csv(get_data('skill_test.csv')).set_index('id')

    module = Pipeline(
        UltTime(),
        AbilityUpgrades(),
        Items(),
        Heroes(),
        Series(),
        DamageTargets(),
        TowerStatus()
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