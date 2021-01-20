import random
import Levenshtein
import numpy as np
import sys
from os.path import join
from nltk.corpus import stopwords
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction, sentence_bleu
from nltk.translate.chrf_score import corpus_chrf, sentence_chrf
from rouge import Rouge

PROJECT_PATH = '../../'
# runNo = '25'
# BEAM_SIZE = 10
runNo = sys.argv[1]
BEAM_SIZE = int(sys.argv[2])
FINAL_BEAM_SIZE = 5 if BEAM_SIZE == 1 or BEAM_SIZE == 5 else 10
BEGIN_BEAM_SIZE = str(BEAM_SIZE) if BEAM_SIZE == 1 or BEAM_SIZE == 10 else ''
# MODEL_NAME = 'transformer'
# MODEL_HYPER_PARAMETER = 'transformer_poetry'
# MODEL_NAME = 'lstm_seq2seq'
# MODEL_HYPER_PARAMETER = 'lstm_seq2seq'
MODEL_NAME = sys.argv[3]
MODEL_HYPER_PARAMETER = sys.argv[4]

rouge = Rouge()

frIn = open(join(PROJECT_PATH, 'model', 'server', runNo, 'QueryReformulation', 'modelOutput',
                 BEGIN_BEAM_SIZE + 'predict_y.bpe.txt.' + MODEL_NAME + '.' + MODEL_HYPER_PARAMETER + '.query_reformulation.beam' + str(
                     FINAL_BEAM_SIZE) + '.alpha0.6.inputs.word'),
            mode="r", encoding="utf-8")
frOut = open(join(PROJECT_PATH, 'model', 'server', runNo, 'QueryReformulation', 'modelOutput',
                  BEGIN_BEAM_SIZE + 'predict_y.bpe.txt.' + MODEL_NAME + '.' + MODEL_HYPER_PARAMETER + '.query_reformulation.beam' + str(
                      FINAL_BEAM_SIZE) + '.alpha0.6.decodes.word'),
             mode="r", encoding="utf-8")
frTruth = open(join(PROJECT_PATH, 'model', 'server', runNo, 'QueryReformulation', 'modelOutput',
                    BEGIN_BEAM_SIZE + 'predict_y.bpe.txt.' + MODEL_NAME + '.' + MODEL_HYPER_PARAMETER + '.query_reformulation.beam' + str(
                        FINAL_BEAM_SIZE) + '.alpha0.6.targets.word'),
               mode="r", encoding="utf-8")

fwPreview = open(join(PROJECT_PATH, 'model', 'server', runNo, 'QueryReformulation', 'eval', 'eval_result.txt'),
                 mode="w",
                 encoding='utf-8')

nltk_stop_words = stopwords.words('english')

so_stop_words = []
with open(join(PROJECT_PATH, 'data', 'so_stop_word.txt'), mode='r', encoding='utf-8') as fr:
    for line in fr.readlines():
        if line.startswith('----') is False:
            so_stop_words.append(line.strip())

smoothFunction = SmoothingFunction()


def match_no_stopwords(doc1, doc2):
    doc1_token = [w for w in doc1.split(' ') if w not in so_stop_words]
    doc2_token = [w for w in doc2.split(' ') if w not in so_stop_words]
    return doc1_token == doc2_token


def exact_match(doc1, doc2):
    doc1_token = doc1.split()
    doc2_token = doc2.split()
    return doc1_token == doc2_token


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


def get_edit_distance_similarity(sentence1, sentence2):
    s1 = ' '.join(sentence1.split())
    s2 = ' '.join(sentence2.split())
    maxLen = max(len(s1), len(s2))
    return 1 - Levenshtein.distance(s1, s2) / maxLen


acc = 0
all = 0
result = []

pendingCalcBLEUGroundTruth = []
pendingCalcBLEUCandidate = []

pendingCalcCHRFGroundTruth = []
pendingCalcCHRFCandidate = []

pendingCalcROUGEGroundTruth = []
pendingCalcROUGECandidate = []

editDistanceSimilarityList = []

finishExtraction = False

while True:
    input = []
    output = []
    truth = []
    for i in range(BEAM_SIZE):
        input.append(frIn.readline().strip())
        output.append(frOut.readline().strip())
        truth.append(frTruth.readline().strip())
        if len(input) == 1 and len(output) == 1 and len(truth) == 1 \
                and input[0] == '' and output[0] == '' and truth[0] == '':
            finishExtraction = True
            break
    if finishExtraction:
        print(all)
        print('end processing')
        break

    assert len(input) == BEAM_SIZE
    assert len(output) == BEAM_SIZE
    assert len(truth) == BEAM_SIZE

    match = False
    resultItem = ['In:    ' + input[0] + '\n']

    findMaxBLEU = []
    findMaxCHRF = []
    findMaxROUGE = []
    maxEditDistanceSimilarity = 0
    for item in output:
        item = item.strip()
        resultItem.append('Out:   ' + item + '\n')

        if exact_match(item, truth[0]):
            match = True
            findMaxBLEU.append((item, 1))
            findMaxCHRF.append((item, 1))
            findMaxROUGE.append((item, 1))
            maxEditDistanceSimilarity = 1
        else:
            if len(item.split()) == 0 or len(truth[0].split()) == 0:
                print('Zero length!')
                continue
            findMaxBLEU.append((item, sentence_bleu([truth[0].split()], item.split(),
                                                    weights=get_bleu_weights(truth[0], item),
                                                    smoothing_function=smoothFunction.method3)))
            findMaxCHRF.append((item, sentence_chrf(truth[0].split(), item.split())))
            findMaxROUGE.append((item, rouge.get_scores([item], [truth[0]])[0]['rouge-l']['f']))
            maxEditDistanceSimilarity = max(maxEditDistanceSimilarity,
                                            get_edit_distance_similarity(item, truth[0]))
    if len(findMaxBLEU) != 0 and len(findMaxCHRF) != 0 and len(findMaxROUGE) != 0:
        pendingCalcBLEUGroundTruth.append([truth[0].split()])
        pendingCalcBLEUCandidate.append(sorted(findMaxBLEU, key=lambda x: x[1], reverse=True)[0][0].split())
        pendingCalcCHRFGroundTruth.append(truth[0].split())
        pendingCalcCHRFCandidate.append(sorted(findMaxCHRF, key=lambda x: x[1], reverse=True)[0][0].split())
        pendingCalcROUGEGroundTruth.append(' '.join(truth[0].split()))
        pendingCalcROUGECandidate.append(' '.join(sorted(findMaxROUGE, key=lambda x: x[1], reverse=True)[0][0].split()))
        editDistanceSimilarityList.append(maxEditDistanceSimilarity)
    else:
        print('Candidate Empty!')
    resultItem.append('Truth: ' + truth[0] + '\n')
    if match:
        acc += 1
        resultItem.append('---Match---' + '\n')
    all += 1
    resultItem.append('\n')
    result.append(resultItem)

print('ExactMatch:{} all:{} match:{}'.format(acc / all, all, acc))

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
    round(acc / all, 4),
    round(corpusBLEU, 4),
    round(corpusCHRF, 4),
    round(avgROUGE1, 4),
    round(avgROUGE2, 4),
    round(avgROUGEl, 4),
    round(avgEditDistanceSimilarity, 4)
))

random.shuffle(result)

for item in result:
    fwPreview.write(''.join(item))
