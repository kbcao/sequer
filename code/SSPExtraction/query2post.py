import pickle
import random
import json
from os.path import join

PROJECT_PATH = '../../'
GREEDY_MODE = 'greedy'

with open(join(PROJECT_PATH, 'data', GREEDY_MODE + 'RMHardKeywordList.bin'), mode="rb") as fr:
    SSPList = list(pickle.load(fr))

with open(join(PROJECT_PATH, 'data', 'questionContent.json'), mode="r", encoding="utf-8") as fr:
    questionContent = dict(json.load(fr))

questionTitle2qid = dict()
for qid, attributes in questionContent.items():
    questionTitle2qid[attributes['Title']] = qid

random.shuffle(SSPList)

train_pair = 0
query2clickedPost = dict()

for ssp in SSPList:
    _groundTruth = ssp[-2]
    postTitle = ssp[-1]
    if questionTitle2qid.get(postTitle, None) is None:
        print('None')
    for query in ssp[:-2]:
        query2clickedPost[query] = {'title': postTitle, 'qid': questionTitle2qid.get(postTitle, None)}
        train_pair += 1

with open(join(PROJECT_PATH, 'result', 'query2clickedPost.json'), mode="w") as fw:
    json.dump(query2clickedPost, fw, indent=2)

print('Training pair: {}'.format(train_pair))
