import json
import pandas as pd
from os.path import join

PROJECT_PATH = '../../'

event = pd.read_csv(join(PROJECT_PATH, 'data', 'LinearSearchThreadEvent.csv'),
                    header=None,
                    names=['Id', 'RootEventId', 'UserIdentifier', 'CreationDate', 'DiffSeconds', 'EventSource',
                           'EventTarget',
                           'Referrer', 'Url', 'Query', 'FragmentIdentifier'],
                    dtype={
                        'Id': 'int64',
                        'RootEventId': pd.Int64Dtype(),
                        'UserIdentifier': 'str',
                        'CreationDate': 'str',
                        'DiffSeconds': pd.Int64Dtype(),
                        'EventSource': 'str',
                        'EventTarget': 'str',
                        'Referrer': 'str',
                        'Url': 'str',
                        'Query': 'str',
                        'FragmentIdentifier': 'str'
                    })
event['CreationDate'] = pd.to_datetime(event['CreationDate'], format='%Y-%m-%d %H:%M:%S')
event = event.sort_values(by=['RootEventId', 'CreationDate'], ascending=True)

REID2IDList = {}
for _, row in event.iterrows():
    if pd.isna(row['RootEventId']):
        REID2IDList[str(row['Id'])] = [str(row['Id'])]
    else:
        if str(row['RootEventId']) not in REID2IDList.keys():
            REID2IDList[str(row['RootEventId'])] = [str(row['Id'])]
        else:
            REID2IDList[str(row['RootEventId'])].append(str(row['Id']))

with open(join(PROJECT_PATH, 'data', 'REID2IDList.json'), "w", encoding="utf-8") as fw:
    json.dump(REID2IDList, fw)
