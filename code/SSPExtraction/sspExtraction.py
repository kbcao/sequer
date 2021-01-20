import json
import pickle

import pandas as pd
import numpy as np
import datetime
from os.path import join
from urllib import parse
from nltk import word_tokenize
from nltk.corpus import stopwords

PROJECT_PATH = '../../'
PREVIEW_MODE = False
GREEDY_MODE = True
MIN_QUERY_LENGTH = 2

with open(join(PROJECT_PATH, 'data', 'REID2IDList.json'), mode="r", encoding="utf-8") as fr:
    REID2IDList = dict(json.load(fr))
with open(join(PROJECT_PATH, 'data', 'ID2ContentSearch.json'), mode="r", encoding="utf-8") as fr:
    ID2ContentSearch = dict(json.load(fr))
with open(join(PROJECT_PATH, 'data', 'ID2ContentPost.json'), mode="r", encoding="utf-8") as fr:
    ID2ContentPost = dict(json.load(fr))
with open(join(PROJECT_PATH, 'data', 'questionContent.json'), mode="r", encoding="utf-8") as fr:
    questionContent = dict(json.load(fr))
with open(join(PROJECT_PATH, 'data', 'answerContent.json'), mode="r", encoding="utf-8") as fr:
    answerContent = dict(json.load(fr))
# with open(join(PROJECT_PATH, 'data', 'allTags.bin'), mode="rb") as fr:
#     allTags = set(pickle.load(fr))

IDSearch = ID2ContentSearch.keys()
IDPost = ID2ContentPost.keys()
questionContentKeys = questionContent.keys()
answerContentKeys = answerContent.keys()


# stopWord = set(stopwords.words('english'))

def calcCharacterSimilarity(str_a, str_b):
    lengthLCS = length_lcs(str_a, str_b)
    lengthTotal = len(str_a) + len(str_b)
    return 2 * lengthLCS / lengthTotal


def length_lcs(str_a, str_b):
    if len(str_a) == 0 or len(str_b) == 0:
        return 0
    dp = [[0 for _ in range(len(str_b) + 1)] for _ in range(len(str_a) + 1)]
    for i in range(1, len(str_a) + 1):
        for j in range(1, len(str_b) + 1):
            if str_a[i - 1] == str_b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max([dp[i - 1][j], dp[i][j - 1]])
    return dp[len(str_a)][len(str_b)]


# def tokenize(doc):
#     return [] if doc is None else word_tokenize(doc)


def getPostTitle(eid):
    postType = ID2ContentPost[eid]['type']
    id1 = ID2ContentPost[eid]['id1']
    if postType[0] == 'q':
        if id1 in questionContentKeys:
            return questionContent[id1]['Title']
        else:
            return None
    elif postType[0] == 'a':
        if id1 in answerContentKeys:
            qid = answerContent[id1]['ParentId']
            if qid in questionContentKeys:
                return questionContent[qid]['Title']
            else:
                return None
        else:
            if postType == 'a0':
                qid = ID2ContentPost[eid]['id2']
                if qid in questionContentKeys:
                    return questionContent[qid]['Title']
                else:
                    return None
            else:
                return None
    else:
        raise KeyError


def getPostDescribe(eid):
    postType = ID2ContentPost[eid]['type']
    id1 = ID2ContentPost[eid]['id1']
    if postType[0] == 'q':
        if id1 in questionContentKeys:
            return 'Q:' + id1 + '  ' + questionContent[id1]['Title']
        else:
            return 'Q:' + id1 + '  ' + 'Missing in Posts.xml'
    elif postType[0] == 'a':
        if id1 in answerContentKeys:
            qid = answerContent[id1]['ParentId']
            if qid in questionContentKeys:
                return 'A:' + id1 + ' ' + 'PQ:' + qid + '  ' + questionContent[qid]['Title']
            else:
                return 'A:' + id1 + ' ' + 'PQ:' + qid + '  ' + 'Missing in Posts.xml'
        else:
            if postType == 'a0':
                qid = ID2ContentPost[eid]['id2']
                if qid in questionContentKeys:
                    return 'A:' + id1 + ' ' + 'PQ:' + qid + '  ' + questionContent[qid]['Title']
                else:
                    return 'A:' + id1 + ' ' + 'PQ:' + qid + '  ' + 'Missing in Posts.xml'
            else:
                return 'A:' + id1 + '  ' + 'Missing in Posts.xml and /a/'
    else:
        raise KeyError


sspList = []

workLength = len(REID2IDList.keys())
print('Work Length: {}'.format(workLength))
workIdx = 0.0
for reid in REID2IDList.keys():
    workIdx += 1
    if workIdx % 50000 == 0:
        print('processing: {:.2f}'.format(100 * workIdx / workLength))

    eventList = REID2IDList[reid]
    i = len(eventList) - 1
    while i >= 0:
        if eventList[i] in IDPost:
            achieveGreedy = False
            if PREVIEW_MODE:
                ssp = [getPostDescribe(eventList[i])]
            else:
                if getPostTitle(eventList[i]) is None:
                    i -= 1
                    continue
                elif i <= 1 or eventList[i - 1] not in IDSearch \
                        or ID2ContentSearch[eventList[i - 1]]['keyword'] is None:
                    i -= 1
                    continue
                else:
                    ssp = [getPostTitle(eventList[i])]
            i -= 1
            lastQuery = ''
            while i >= 0:
                if eventList[i] in IDSearch:
                    keyword = ID2ContentSearch[eventList[i]]['keyword']
                    if keyword is None:
                        if PREVIEW_MODE:
                            ssp.append('Empty Search')
                    else:
                        ssp.append(keyword)
                        lastQuery = keyword
                elif eventList[i] in IDPost:
                    if GREEDY_MODE is False or achieveGreedy is True:
                        break
                    if i == 0 or eventList[i - 1] not in IDSearch:
                        break
                    queryAhead = ID2ContentSearch[eventList[i - 1]]['keyword']
                    if queryAhead is None:
                        break
                    querySimilarity = calcCharacterSimilarity(queryAhead, lastQuery)
                    if querySimilarity < 0.8:
                        break
                    postStartTime = datetime.datetime.strptime(
                        ID2ContentPost[eventList[i]]['time'], '%Y-%m-%d %H:%M:%S')
                    postEndTime = datetime.datetime.strptime(
                        ID2ContentSearch[eventList[i + 1]]['time'], '%Y-%m-%d %H:%M:%S')
                    postDurationSeconds = (postEndTime - postStartTime).seconds
                    if postDurationSeconds >= 30:
                        break

                    achieveGreedy = True
                    if PREVIEW_MODE:
                        ssp.append(getPostDescribe(eventList[i]))
                else:
                    break
                i -= 1
            if len(ssp) >= (MIN_QUERY_LENGTH + 1):
                sspList.append(ssp)
        else:
            i -= 1

for i in range(0, len(sspList)):
    sspList[i].reverse()

# with open(join(PROJECT_PATH, 'result', 'greedyQueryPreview.txt' if GREEDY_MODE else 'strictQueryPreview.txt'), mode="w",
#           encoding="utf-8") as fw:
#     for ssp in sspList:
#         for kp in ssp:
#             fw.write(kp)
#             fw.write('\n')
#         fw.write('\n')

with open(join(PROJECT_PATH, 'data', 'greedyRawKeywordList.bin' if GREEDY_MODE else 'strictRawKeywordList.bin'),
          mode="wb") as fw:
    pickle.dump(sspList, fw)
