import itertools
import os
import csv
import json
import random
from typing import List, Tuple, Generator
import tqdm

N_truth = 1
data_dir = 'data-ml1m-1-100'

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def sample(n: int, exclude: List[int], count: int = 1):
    exclude_set = set(exclude)
    return random.sample(
        [i for i in range(n) if i not in exclude_set],
        count
    )

def yield_step(user_mapping, item_mapping, record):
    user, item, rating, time = record
    if user not in user_mapping:
        user_mapping[user] = len(user_mapping)
    if item not in item_mapping:
        item_mapping[item] = len(item_mapping)
    return user_mapping[user], item_mapping[item], rating, time

def encode_user_item(records):
    threshold = max(int(N_truth * 1.5), 2)
    print(f'Threshold: {threshold}')
    user_review_pending = {}
    user_mapping = {}
    item_mapping = {}
    record_count = 0

    for line in tqdm.tqdm(records):
        user, item, rating, time = filter(None, line)
        assert all(map(lambda x: x is not None, line))
        if float(rating) < 0:
            continue
        if user not in user_review_pending:
            user_review_pending[user] = {}

        if user_review_pending[user] is not None:
            if len(user_review_pending[user]) < threshold:
                # Add but not yield
                if item not in user_review_pending[user]:
                    user_review_pending[user][item] = (item, rating, time)
                continue
            else:
                for record in user_review_pending[user].values():
                    yield yield_step(user_mapping, item_mapping, [user, *record])
                    record_count += 1
                user_review_pending[user] = None
        else:
            yield yield_step(user_mapping, item_mapping, [user, item, rating, time])
            record_count += 1

    yield -1, user_mapping, item_mapping, record_count

def group_by_user(records: Generator[Tuple[int, int, float, int], None, None]):
    for uid, iid, rating, time in records:
        yield uid, (iid, rating, time)

if __name__ == '__main__':
    f = open('../ml-1m/ratings.dat', 'r')
    reader = csv.reader(f, delimiter=':')
    next(reader)
    sorted_interactions = sorted([
        record for record in group_by_user(encode_user_item(reader))
    ], key=lambda x: x[0])
    data = {
        u: sorted((_[1] for _ in records), key=lambda x: x[-1])
        for u, records in itertools.groupby(sorted_interactions, key=lambda x: x[0])
    }
    f.close()
    user_mapping, item_mapping, record_count = data.pop(-1)[0]
    user_count = len(user_mapping) # type: ignore
    item_count = len(item_mapping) # type: ignore
    print(f'User count: {user_count}')
    print(f'Item count: {item_count}')
    print(f'Record count: {record_count}')
    with open(f'data/{data_dir}/meta.json', 'w') as f:
        json.dump({
            'user': {
                'count': user_count,
                'mapping': user_mapping,
            },
            'item': {
                'count': item_count,
                'mapping': item_mapping,
            },
            'record_count': record_count,
        }, f)

    # Truncate rating field and timestamp field
    data_ui = {
        u: [*{_[0] for _ in records}] for u, records in data.items()
    }
    # for k, v in data_ui.items():
    #     if len(v) <= 10:
    #         print(k, len(v))
    #         assert False

    train = {u: i[:-N_truth] for u, i in tqdm.tqdm(data_ui.items(), total=len(data_ui))}
    test = {
        u: [*record[-N_truth:], *sample(item_count, train[u], 100 - N_truth)]
        for u, record in tqdm.tqdm(data_ui.items(), total=len(data_ui))
    }

    with open(f'data/{data_dir}/train.txt', 'w') as f:
        for u, records in train.items():
            f.write(' '.join(map(str, [u, *records])) + '\n')
    with open(f'data/{data_dir}/test.txt', 'w') as f:
        for u, records in test.items():
            f.write(' '.join(map(str, [u, *records])) + '\n')

    with open(f'data/{data_dir}/train.json', 'w') as f:
        json.dump(train, f)
    with open(f'data/{data_dir}/test.json', 'w') as f:
        json.dump(test, f)
