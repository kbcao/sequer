import pickle
import gc
import requests
import json

GREEDY_MODE = 'greedy'

with open(GREEDY_MODE + 'RMHardKeywordList.bin', mode="rb") as fr:
    sspList = list(pickle.load(fr))

gCnt = 0

tmpStorage = []

for ssp in sspList:
    groundTruth = (' '.join(ssp[-2].split())).strip()

    if gCnt >= 65103:
        break

    for query in ssp[:-2]:
        gCnt += 1
        query = (' '.join(query.split())).strip()
        response = requests.get(url='http://localhost:8081/v2/check',
                                params={'language': 'en-US', 'text': query})
        matches = json.loads(response.text, encoding='utf-8').get('matches', [])
        matches = sorted(matches, key=lambda x: int(x['offset']))
        choices = [query]
        for advice in matches:
            candidates = advice['replacements']
            offset = int(advice['offset'])
            length = int(advice['length'])
            badWord = query[offset:offset + length]

            if len(choices) >= 100000:
                print('Too many choice')
                break
            if len(choices) > 10000:
                print(len(choices))

            choicesTemp = []
            for reformulated in choices:
                if reformulated.find(badWord) < 0:
                    # print('NO FOUND ERROR')
                    continue
                for candidate in candidates:
                    choicesTemp.append(reformulated.replace(badWord, candidate['value']))
                    # print('append: {}'.format(reformulated.replace(badWord, candidate['value'])))
            for i in range(len(choicesTemp) - 1, -1, -1):
                if choicesTemp[i][:offset] != groundTruth[:offset]:
                    choicesTemp.pop(i)
            choices = choices + choicesTemp
        choices.append(groundTruth)

        tmpStorage.append(choices)
        if gCnt % 100 == 0:
            with open('./ltProcess/' + str(gCnt) + '.pkl', mode='wb')as fw:
                pickle.dump(tmpStorage, fw)
            print(gCnt)
            for tmp in tmpStorage:
                print(len(tmp), end=' ')
            del tmpStorage
            gc.collect()
            tmpStorage = []

        del choices
        gc.collect()

with open('./ltProcess/' + str(gCnt) + '.pkl', mode='wb')as fw:
    pickle.dump(tmpStorage, fw)
