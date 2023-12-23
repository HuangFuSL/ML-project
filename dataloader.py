import json
import os
import torch.utils.data
import random
from typing import Dict, List, overload

class BPR_Loader(torch.utils.data.Dataset):
    # Only available for train data
    
    def __init__(self, path: str, split: str = 'train'):
        super(BPR_Loader, self).__init__()
        self._current_split = split
        self.user_num = 0
        self.item_num = 0
        self.length = {'train': 0, 'test': 0}
        self.data = {}
        self.positive_per_user = {} # Only used in training
        self._graph = None
        records = {'train': {}, 'test': {}}

        if os.path.exists(f'data/{path}/descriptive.json'):
            with open(f'data/{path}/descriptive.json', 'r') as f:
                self.item_description = {int(k): json.dumps(v) for k, v in json.load(f).items()}

        all_items = set()
        
        for _split in ['train', 'test']:
            with open(f'data/{path}/{_split}.txt', 'r') as f:
                while True:
                    line = f.readline()
                    if not line:
                        break

                    user, *items = map(int, line.strip().split(' '))

                    records[_split][user] = items
                    if _split == 'train':
                        all_items.update(items)
                        self.positive_per_user[user] = [*items]

                    self.user_num = max(self.user_num, user)
                    self.item_num = max(self.item_num, *items)
                    self.length[_split] += len(items)

            if _split == 'train':
                self.data[_split] = torch.zeros((self.length[_split], 3), dtype=torch.long)
                
                i = 0
                for user, items in records[_split].items():
                    for item in items:
                        self.data[_split][i] = torch.tensor([user, item, 0])
                        i += 1
            else:
                self.data[_split] = {k: torch.tensor(v).reshape(-1).cuda() for k, *v in records[_split].items()}
                self.length[_split] = len(self.data[_split])

        self.user_num += 1
        self.item_num += 1

        print('Data loaded')

    @property
    def graph(self):
        return None

    def query_user_test_set(self, uid):
        return self.data['test'][uid.item()]

    @overload
    def query_item_description(self, iid: int) -> str:
        ...

    @overload
    def query_item_description(self, iid: List[int]) -> List[str]:
        ...

    def query_item_description(self, iid: int | List[int]) -> Dict[int, str]:
        if isinstance(iid, int):
            return {iid: self.item_description.get(iid, '')}
        else:
            return {i: self.item_description[i] for i in iid if i in self.item_description}

    def query_user_viewed(self, uid):
        return self.positive_per_user[uid.item()]

    def train(self):
        self._current_split = 'train'

    def eval(self):
        self._current_split = 'test'

    def __len__(self):
        return self.length[self._current_split]

    def __getitem__(self, idx):
        if self._current_split == 'train':
            uid, pid, _ = self.data['train'][idx]
            while True:
                nid = random.randint(0, self.item_num - 1)
                if nid not in self.positive_per_user[uid.item()]:
                    break
            self.data['train'][idx][2] = nid
            return self.data[self._current_split][idx]
        else:
            return idx
