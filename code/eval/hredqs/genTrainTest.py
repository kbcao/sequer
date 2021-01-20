import pickle
import random
import json
from os.path import join

PROJECT_PATH = '../../../'
GREEDY_MODE = 'greedy'

fr_x = open(join(PROJECT_PATH, 'model', 'tensor2tensor-master', 'QueryReformulation', 'bpedata', 'train_x.bpe.txt'),
            mode="r", encoding='utf-8')

fr_y = open(join(PROJECT_PATH, 'model', 'tensor2tensor-master', 'QueryReformulation', 'bpedata', 'train_y.bpe.txt'),
            mode="r", encoding='utf-8')

fw_train = open(join(PROJECT_PATH, 'data', 'hredqs_train.txt'), mode="w", encoding='utf-8')
fw_test = open(join(PROJECT_PATH, 'data', 'hredqs_test.txt'), mode="w", encoding='utf-8')
fw_val = open(join(PROJECT_PATH, 'data', 'hredqs_val.txt'), mode="w", encoding='utf-8')

write_data = []
for x, y in zip(fr_x.readlines(), fr_y.readlines()):
    x = x.strip()
    y = y.strip()
    sessionId = ''.join([chr(random.randrange(97, 122)) for _ in range(10)])
    originalQueryId = ''.join([chr(random.randrange(97, 122)) for _ in range(10)])
    reformedQueryId = ''.join([chr(random.randrange(97, 122)) for _ in range(10)])
    data = {"session_id": sessionId,
            "query": [
                {"id": originalQueryId, "text": x, "tokens": x.split(' '), "type": "DESCRIPTION", "candidates": []},
                {"id": reformedQueryId, "text": y, "tokens": y.split(' '), "type": "DESCRIPTION", "candidates": []}
            ]}
    write_data.append(json.dumps(data))

random.shuffle(write_data)
for item in write_data[:int(len(write_data) * 0.9)]:
    fw_train.write(item + '\n')
for item in write_data[int(len(write_data) * 0.9):int(len(write_data) * 0.95)]:
    fw_test.write(item + '\n')
for item in write_data[int(len(write_data) * 0.95):]:
    fw_val.write(item + '\n')
