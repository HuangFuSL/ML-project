import os
import json
import zhipuai
import tqdm
import time
import torch
import multiprocessing

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def query_embedding(text: str):
    time.sleep(1)
    while True:
        response = zhipuai.model_api.invoke(
            prompt=text,
            model='text_embedding'
        )
        assert response is not None
        if response['success']:
            return torch.tensor(response['data']['embedding'])
        else:
            print(response)
            time.sleep(1)

if __name__ == '__main__':
    with open('data-dm-10-truth/meta.json', 'r') as f:
        meta = json.load(f)
    item_map = meta['item']['mapping']

    descriptive_map = {}
    with open('/workspace/meta_Digital_Music.json', 'r') as f:
        for line in tqdm.tqdm(f):
            item = json.loads(line)
            if item['asin'] in item_map:
                descriptive_map[item_map[item['asin']]] = {
                    'title': item['title'],
                    'rank': item['rank'],
                    'brand': item['brand'],
                }
    with open('data-dm-10-truth/descriptive.json', 'w') as f:
        json.dump(descriptive_map, f)

    # def func(x):
    #     key, value = x
    #     return int(key), query_embedding(json.dumps(value))
    # result_tensor = torch.zeros((len(item_map)), 1024)

    # with multiprocessing.Pool(24) as pool:
    #     for key, value in tqdm.tqdm(pool.imap(func, descriptive_map.items()), total=len(descriptive_map)):
    #         result_tensor[key] = value

    # torch.save(result_tensor, 'data-dm/descriptive-embed.pt')
