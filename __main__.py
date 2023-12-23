import json
from typing import Dict, List
import torch
import tqdm
import os
import torch.utils.data
from . import model, dataloader, utils
import numpy as np
import zhipuai

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def hit(topk_items, truth):
    return len(set(topk_items) & set(truth)) / len(truth)

def ndcg(topk_items, truth):
    ret = 0
    for _ in truth:
        if _ in topk_items:
            index = topk_items.index(_)
            ret += np.reciprocal(np.log2(index + 2))
    return ret / len(truth)

def BPR_loss(positive_scores, negative_scores):
    return -torch.log(torch.sigmoid(positive_scores - negative_scores)).mean()

def NLL_loss(positive_scores, negative_scores):
    return (-torch.log(torch.sigmoid(positive_scores)) - torch.log(1 - torch.sigmoid(negative_scores))).mean()

loss_funcs = {
    'BPR_loss': BPR_loss,
    'NLL_loss': NLL_loss
}

test_ratio = 0.000
zhipuai.api_key = '3a417750decde74eb623193dc4895a30.iwvGRsQcRGQ4AfHX'
prompt_en = \
    'You are serving as a judge in a recommendation system.' \
    'Each time your upstream will send you a user and a list of items' \
    'The list contains 50 items, ranked by the upstream\'s model.' \
    'The list is given in the json list format, e.g. [1, 2, 3, 4, 5]' \
    'You need to rank the items and return the top 5 items.' \
    'The more relevant the item is, the further forward it should be.\n' \
    'The output should also be in the json list format, e.g. [1, 2, 3, 4, 5]' \
    'Keep in mind that you should ONLY return the json representation of ITEMS but not CODE or anything else.' \
    'Do not give any text not in json format.' \
    'To help your decision, for each item, we can provide you with the information of the item.' \
    'The extra information is also given in the json format.' \
    'e.g. {{"title": "Master Collection Volume One", "rank": "58,291 in CDs & Vinyl (", "brand": "John Michael Talbot"}}' \
    'You can use the information to help your decision.' \
    'Remember, you should only return the top 10 items, ranked by relevance' \
    'The recommended list is `{recommended_list}`.' \
    'The key is the item index, and the value is the relevance score.' \
    'Descriptive information of the items is `{descriptive_info}`.' \
    'You only need to return the index of top-10 items in json format, ranked by relevance.' \
    'Do not return more or less than 10 items.' \
    'Do not return duplicate items.' \
    'Do not return items that are not in the recommended list.' \
    'Do not return any unrelated information.'

def query_LLM(recommended_list: Dict[int, float], descriptive_info: List[str]):
    while True:
        response = zhipuai.model_api.invoke(
            prompt=[{"role": "user", "content": prompt_en.format(
                recommended_list=json.dumps(recommended_list),
                descriptive_info=json.dumps(descriptive_info)
            )}],
            temperature=0.95,
            top_p=0.7,
            model='chatglm_turbo'
        )
        assert response is not None
        if response['success']:
            return response['data']['choices'][0]['content']
        else:
            print(response)

def resort(topl_items: Dict[int, float], descriptions: List[str]):
    result = json.loads(json.loads(query_LLM(topl_items, descriptions).strip()))
    return [*map(int, result)]


def test(config: utils.Config, dataset: dataloader.BPR_Loader, m: torch.nn.Module):
    m.eval()
    dataset.eval()
    recalls = []
    ndcgs = []
    test_results = []
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True, pin_memory=True, num_workers=12
    )
    
    for k in config.test_Ks:
        for uids in tqdm.tqdm(loader):
            uids = uids.to('cuda')
            for uid in uids:
                test = dataset.query_user_test_set(uid)
                if len(test) == 0:
                    continue
                truth = test[:1].cpu().numpy().tolist()
                uid_tensor = torch.tensor([uid.item()]).repeat(test.shape[0]).to('cuda')
                scores = m(uid_tensor, test)
                topk_items = test[torch.topk(scores, k).indices].cpu().numpy().tolist()
                recalls.append(hit(topk_items, truth))
                ndcgs.append(ndcg(topk_items, truth))
        test_results.append(utils.TestResult(
            hash(config), k, sum(recalls) / len(recalls), sum(ndcgs) / len(ndcgs)
        ))
        print(f'HR@{k}: {sum(recalls) / len(recalls)}')
        print(f'NDCG@{k}: {sum(ndcgs) / len(ndcgs)}')
    return test_results

def evaluate():
    ...

def train(config: utils.Config):
    dataset = dataloader.BPR_Loader(config.data_path)
    m = model.__dict__[config.model](
        dataset.user_num,
        dataset.item_num,
        **dict(config.model_args)
    )
    if os.path.exists(f'results/{hash(config)}'):
        raise Exception(f'Already trained, result in results/{hash(config)}')
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True, pin_memory=True, num_workers=12
    )
    optimizer = torch.optim.__dict__[config.optimizer](m.parameters(), **dict(config.optimizer_args))
    lr_scheduler = torch.optim.lr_scheduler.__dict__[config.scheduler](optimizer, **dict(config.scheduler_args))
    
    epoch_loss = []
    m.cuda()
    m.train()
    dataset.train()
    for epoch in range(config.epochs):
        losses = []
        for batch in loader:
            optimizer.zero_grad()
            batch = batch.to('cuda')
            positive_scores = m(batch[:, 0], batch[:, 1])
            negative_scores = m(batch[:, 0], batch[:, 2])
            loss = loss_funcs[config.loss_func](positive_scores, negative_scores)
            # Check nan
            if torch.isnan(loss):
                print(positive_scores, negative_scores)
                raise Exception('Loss is NaN')
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
        lr_scheduler.step()
        epoch_loss.append(sum(losses) / len(losses))
        print(f'Epoch {epoch} loss: {sum(losses) / len(losses)}')

    return m, epoch_loss, test(config, dataset, m)

# Train Matrix Factorization
if __name__ == '__main__':

    train_config = utils.Config(
        random_seed=234389259823,
        data_path='data-ml1m-1-100',
        model='NeuMF',
        model_args=(
            ('latent_dim', 32),
            ('hidden_dims', (256, 128, 64))
        ),
        optimizer='Adam',
        optimizer_args=(('lr', 0.001), ('weight_decay', 0.00001)),
        scheduler='StepLR',
        scheduler_args=(('step_size', 15), ('gamma', 0.9)),
        batch_size=256,
        epochs=50,
        test_Ks=(5, 10, 20),
        loss_func='NLL_loss'
    )
    print(hash(train_config))

    m, epoch_loss, test_results = train(train_config)
    utils.save_results(train_config, m, epoch_loss, test_results)
    
    # model_name = 'MLP-ML-1.pt'
    # dataset = dataloader.BPR_Loader('data-ml-1m-1-truth')
    # mf = model.NeuMF(dataset.user_num, dataset.item_num, 64, (32, 16, 8))
    # if os.path.exists(model_name):
    #     mf.load_state_dict(torch.load(model_name))
    # loader = torch.utils.data.DataLoader(dataset, batch_size=2048, shuffle=True, pin_memory=True, num_workers=12)
    # optimizer = torch.optim.Adam(mf.parameters(), lr=0.001)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.9)

    # mf.cuda()
    # mf.train()
    # dataset.train()
    # for epoch in range(100):
    #     losses = []
    #     for batch in loader:
    #         optimizer.zero_grad()
    #         batch = batch.to('cuda')
    #         # positive_embed = dataset.query_item_embed(batch[:, 1]).cuda()
    #         # negative_embed = dataset.query_item_embed(batch[:, 2]).cuda()
    #         # positive_scores = mf(batch[:, 0], batch[:, 1], positive_embed)
    #         # negative_scores = mf(batch[:, 0], batch[:, 2], negative_embed)
    #         positive_scores = mf(batch[:, 0], batch[:, 1])
    #         negative_scores = mf(batch[:, 0], batch[:, 2])
    #         loss = model.BPR_loss(positive_scores, negative_scores)
    #         loss.backward()
    #         losses.append(loss.item())
    #         optimizer.step()
    #     lr_scheduler.step()
    #     print(f'Epoch {epoch} loss: {sum(losses) / len(losses)}')
    # torch.save(mf.state_dict(), model_name)

    # k = 10
    # mf.load_state_dict(torch.load(model_name))
    # mf.cuda()
    # mf.eval()
    # dataset.eval()
    # recalls = []
    # ndcgs = []
    # resorted_recalls = []
    # resorted_ndcgs = []
    # for uids in tqdm.tqdm(loader):
    #     uids = uids.to('cuda')
    #     for uid in uids:
    #         test = dataset.query_user_test_set(uid)
    #         if len(test) == 0:
    #             continue
    #         viewed = dataset.query_user_viewed(uid)
    #         truth = test[:1].cpu().numpy().tolist()
    #         uid_tensor = torch.tensor([uid.item()]).repeat(test.shape[0]).to('cuda')
    #         scores = mf(uid_tensor, test)
    #         topk_items = test[torch.topk(scores, k).indices].cpu().numpy().tolist()
    #         topk_scores = scores[torch.topk(scores, k).indices].cpu().detach().numpy().tolist()
    #         if random.random() < test_ratio:
    #             resorted = resort({
    #                 k: v for k, v in zip(topk_items, topk_scores)
    #                 }, dataset.query_item_description(topk_items)
    #             )
    #             resorted_recalls.append(hit(resorted, truth))
    #             resorted_ndcgs.append(ndcg(resorted, truth))
    #         recalls.append(hit(topk_items, truth))
    #         ndcgs.append(ndcg(topk_items, truth))
    # print(f'HR@{k}: {sum(recalls) / len(recalls)}')
    # print(f'NDCG@{k}: {sum(ndcgs) / len(ndcgs)}')
    # if len(resorted_recalls) > 0:
    #     print(f'Resorted HR@10: {sum(resorted_recalls) / len(resorted_recalls)}')
    #     print(f'Resorted NDCG@10: {sum(resorted_ndcgs) / len(resorted_ndcgs)}')
