import pickle
import gc
import requests
import yaml
from os.path import join

GREEDY_MODE = 'greedy'
PROJECT_PATH = '../../'

with open(join(PROJECT_PATH, 'data', GREEDY_MODE + 'RMHardKeywordList.bin'), mode="rb") as fr:
    sspList = list(pickle.load(fr))

# 525398

reqUrl = 'http://google.com/complete/search'

gCnt = 0

tmpStorage = []

for ssp in sspList:
    groundTruth = (' '.join(ssp[-2].split())).strip()

    if gCnt >= 65103:
        break

    for query in ssp[:-2]:
        query = (' '.join(query.split())).strip()
        getPara = dict(client='chrome', hl='en', q=query)
        response = requests.get(url=reqUrl, params=getPara)
        if response.ok is False:
            print('ERROR Reformulation')
            continue
        try:
            suggestions = yaml.load(response.text, Loader=yaml.BaseLoader)[1]
        except:
            print('ERROR Parsing')
            continue

        gCnt += 1

        choices = [query]
        for suggestion in suggestions:
            choices.append(suggestion)

        choices.append(groundTruth)

        tmpStorage.append(choices)
        if gCnt % 100 == 0:
            with open(join(PROJECT_PATH, 'data', 'googleBaseline', str(gCnt) + '.pkl'), mode='wb')as fw:
                pickle.dump(tmpStorage, fw)
            print('\n', gCnt, sspList.index(ssp), end=' ', flush=True)
            for tmp in tmpStorage:
                print(len(tmp), end=' ', flush=True)
            del tmpStorage
            gc.collect()
            tmpStorage = []

        del choices
        gc.collect()

with open(join(PROJECT_PATH, 'data', 'googleBaseline', str(gCnt) + '.pkl'), mode='wb')as fw:
    pickle.dump(tmpStorage, fw)
