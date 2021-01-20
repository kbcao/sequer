import json
from os.path import join

PROJECT_PATH = '../../../'

fr = open(join(PROJECT_PATH, 'result', 'hredqs', 'v3hredqs_test.json'), mode="r", encoding='utf-8')

fw_input = open(join(PROJECT_PATH, 'result', 'hredqs', 'v3hredqs_input.txt'), mode="w", encoding='utf-8')
fw_output = open(join(PROJECT_PATH, 'result', 'hredqs', 'v3hredqs_output.txt'), mode="w", encoding='utf-8')
fw_truth = open(join(PROJECT_PATH, 'result', 'hredqs', 'v3hredqs_truth.txt'), mode="w", encoding='utf-8')

for line in fr.readlines():
    data = json.loads(line.strip())
    _sessionId = data['session_id']
    _input = data['previous_queries'].replace('@@ ', '')
    _outputs = [item.replace('@@ ', '') for item in data['predictions']]
    _truth = data['references'][0].replace('@@ ', '')

    for i in range(5):
        fw_input.write(_input + '\n')
        fw_output.write(_outputs[i] + '\n')
        fw_truth.write(_truth + '\n')
