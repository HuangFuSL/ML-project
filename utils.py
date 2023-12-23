import collections
import json
import os
from typing import List
import torch

os.chdir(os.path.dirname(os.path.abspath(__file__)))

Config = collections.namedtuple('config', [
    'random_seed',
    'data_path',
    'model',
    'model_args',
    'optimizer',
    'optimizer_args',
    'scheduler',
    'scheduler_args',
    'batch_size',
    'epochs',
    'test_Ks',
    'loss_func'
])
TestResult = collections.namedtuple('test_result', ['confighash', 'k', 'recall', 'ndcg'])

def save_results(config: Config, model: torch.nn.Module, loss: List[float], test_results: List[TestResult]):
    assert all(hash(config) == _.confighash for _ in test_results)
    root_dir = f'results/{hash(config)}'
    if not os.path.exists('results'):
        os.mkdir('results')
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)

    with open(f'{root_dir}/config.json', 'w') as f:
        json.dump(config._asdict(), f)

    torch.save(model.state_dict(), f'{root_dir}/model.pt')

    with open(f'{root_dir}/results.json', 'w') as f:
        json.dump({
            'loss': loss,
            'test_results': [_._asdict() for _ in test_results]
        }, f)
