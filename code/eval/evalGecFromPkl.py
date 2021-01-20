import pickle
import gc
import os
import Levenshtein
import numpy as np
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction, sentence_bleu
from nltk.translate.chrf_score import corpus_chrf, sentence_chrf
from rouge import Rouge
from os.path import join

PROJECT_PATH = '../../'

path = join(PROJECT_PATH, 'data', 'googleProcess')

smoothFunction = SmoothingFunction()

rouge = Rouge()

totalPair = 0
correctPair = 0

pendingCalcBLEUGroundTruth = []
pendingCalcBLEUCandidate = []

pendingCalcCHRFGroundTruth = []
pendingCalcCHRFCandidate = []

pendingCalcROUGEGroundTruth = []
pendingCalcROUGECandidate = []

editDistanceSimilarityList = []


def get_edit_distance_similarity(sentence1, sentence2):
    s1 = ' '.join(sentence1.split())
    s2 = ' '.join(sentence2.split())
    maxLen = max(len(s1), len(s2))
    return 1 - Levenshtein.distance(s1, s2) / maxLen


def get_bleu_weights(sentence1, sentence2):
    bleuSize = min(len(sentence1.split()), len(sentence2.split()))
    if bleuSize == 1:
        return [1, 0, 0, 0]
    elif bleuSize == 2:
        return [0.5, 0.5, 0, 0]
    elif bleuSize == 3:
        return [1 / 3, 1 / 3, 1 / 3, 0]
    else:
        return [0.25, 0.25, 0.25, 0.25]


for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith('.pkl') is False:
            continue
        with open(os.path.join(root, file), mode='rb') as fr:
            print('processing:{}'.format(file))
            samples100 = list(pickle.load(fr))
            for sample in samples100:
                groundTruth = sample[-1]
                query = sample[0]
                choices = sample[1:-1]
                totalPair += 1

                findMaxBLEU = []
                findMaxCHRF = []
                findMaxROUGE = []
                maxEditDistanceSimilarity = 0
                match = False
                for choice in choices[:10]:
                    # print('choice')
                    # print(choice)
                    if choice == groundTruth:
                        match = True
                        correctPair += 1
                        findMaxBLEU.append((choice, 1))
                        findMaxCHRF.append((choice, 1))
                        findMaxROUGE.append((choice, 1))
                        maxEditDistanceSimilarity = 1
                        break
                    if len(choice.split()) == 0 or len(groundTruth.split()) == 0:
                        print('Zero length!')
                        continue
                    findMaxBLEU.append((choice, sentence_bleu([groundTruth.split()], choice.split(),
                                                              weights=get_bleu_weights(groundTruth, choice),
                                                              smoothing_function=smoothFunction.method3)))
                    findMaxCHRF.append((choice, sentence_chrf(groundTruth.split(), choice.split())))
                    findMaxROUGE.append((choice, rouge.get_scores([choice], [groundTruth])[0]['rouge-l']['f']))
                    maxEditDistanceSimilarity = max(maxEditDistanceSimilarity,
                                                    get_edit_distance_similarity(choice, groundTruth))
                if match is True or (len(findMaxBLEU) != 0 and len(findMaxCHRF) != 0 and len(findMaxROUGE) != 0):
                    pendingCalcBLEUGroundTruth.append([groundTruth.split()])
                    pendingCalcBLEUCandidate.append(sorted(findMaxBLEU, key=lambda x: x[1], reverse=True)[0][0].split())
                    pendingCalcCHRFGroundTruth.append(groundTruth.split())
                    pendingCalcCHRFCandidate.append(sorted(findMaxCHRF, key=lambda x: x[1], reverse=True)[0][0].split())
                    pendingCalcROUGEGroundTruth.append(' '.join(groundTruth.split()))
                    pendingCalcROUGECandidate.append(
                        ' '.join(sorted(findMaxROUGE, key=lambda x: x[1], reverse=True)[0][0].split()))
                    editDistanceSimilarityList.append(maxEditDistanceSimilarity)
                else:
                    if query == groundTruth:
                        correctPair += 1
                    pendingCalcBLEUGroundTruth.append([groundTruth.split()])
                    pendingCalcBLEUCandidate.append(query.split())
                    pendingCalcCHRFGroundTruth.append(groundTruth.split())
                    pendingCalcCHRFCandidate.append(query.split())
                    pendingCalcROUGEGroundTruth.append(' '.join(groundTruth.split()))
                    pendingCalcROUGECandidate.append(' '.join(query.split()))
                    editDistanceSimilarityList.append(get_edit_distance_similarity(query, groundTruth))
            del samples100
            gc.collect()

print('ExactMatch:{} all:{} match:{}'.format(correctPair / totalPair, totalPair, correctPair))

corpusBLEU = corpus_bleu(pendingCalcBLEUGroundTruth, pendingCalcBLEUCandidate, weights=(0.5, 0.5, 0, 0),
                         smoothing_function=smoothFunction.method3)
print('CorpusBLEU: {}'.format(corpusBLEU))

corpusCHRF = corpus_chrf(pendingCalcCHRFGroundTruth, pendingCalcCHRFCandidate)
print('CorpusCHRF: {}'.format(corpusCHRF))

avgROUGE = rouge.get_scores(pendingCalcROUGECandidate, pendingCalcROUGEGroundTruth, avg=True)
avgROUGE1 = avgROUGE['rouge-1']['f']
avgROUGE2 = avgROUGE['rouge-2']['f']
avgROUGEl = avgROUGE['rouge-l']['f']

print('AvgROUGE1:{}'.format(avgROUGE1))
print('AvgROUGE2:{}'.format(avgROUGE2))
print('AvgROUGEl:{}'.format(avgROUGEl))

avgEditDistanceSimilarity = float(np.mean(editDistanceSimilarityList))
print('AvgEditDistanceSimilarity:{}'.format(avgEditDistanceSimilarity))

print('em:{} bleu:{} chrf: {} rg1:{} rg2:{} rgl:{} ED:{}'.format(
    round(correctPair / totalPair, 4),
    round(corpusBLEU, 4),
    round(corpusCHRF, 4),
    round(avgROUGE1, 4),
    round(avgROUGE2, 4),
    round(avgROUGEl, 4),
    round(avgEditDistanceSimilarity, 4)
))
