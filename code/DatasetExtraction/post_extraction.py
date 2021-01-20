
import json
import re
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

ID2ContentPost = {}
for _, row in event.iterrows():
    assert str(row['Id']) not in ID2ContentPost.keys()
    if row['EventTarget'] == 'Post':
        url = str(row['Url'])
        questionsMatch = re.search(r'/questions/[0-9]{1,9}', url)
        if questionsMatch is not None:
            answerMatch = re.search(r'/questions/[0-9]{1,9}/.+[/,#][0-9]{1,9}', url)
            if answerMatch is not None:
                linkType = 'a0'
                id1 = int(re.findall(r'[/,#][0-9]{1,9}', answerMatch.group()[11:])[-1][1:])
                id2 = int(questionsMatch.group()[11:])
            else:
                linkType = 'q0'
                id1 = int(questionsMatch.group()[11:])
                id2 = None
        else:
            linkType = ''
            aMatch = re.search(r'/a/', url)
            qMatch = re.search(r'/q/', url)
            if aMatch is None:
                if qMatch is None:
                    print('Error url: {}'.format(url))
                    raise LookupError('/a/ and /q/ both do not exist')
                else:
                    linkType = 'q'
            else:
                linkType = 'a'
                assert qMatch is None, '/a/ /q/ both exist'
            twoIdMatch = re.search(r'/[a,q]/[0-9]{1,9}/[0-9]{1,9}', url)
            if twoIdMatch is None:
                OneIdMatch = re.search(r'/[a,q]/[0-9]{1,9}', url)
                if OneIdMatch is None:
                    print('Error url: {}'.format(url))
                    raise LookupError('/a|q/xxx/xxx /a|q/xxx both can not be found')
                else:
                    linkType = linkType + '1'
                    id1 = int(OneIdMatch.group()[3:])
                    id2 = None
                    # print('LinkType: {}  ID: {}  URL: {}'.format(linkType, id1, url))
            else:
                TwoIdStr = twoIdMatch.group()
                linkType = linkType + '2'
                id1 = int(TwoIdStr[3:TwoIdStr.find('/', 3)])
                id2 = int(TwoIdStr[TwoIdStr.find('/', 3) + 1:])
                # print('LinkType: {}  ID: {} {}  URL: {}'.format(linkType, id1, id2, url))
        ID2ContentPost[str(row['Id'])] = {'reid': None if pd.isna(row['RootEventId']) else str(row['RootEventId']),
                                          'user': str(row['UserIdentifier']),
                                          'time': str(row['CreationDate']),
                                          'source': str(row['EventSource']),
                                          'target': str(row['EventTarget']),
                                          'type': str(linkType),
                                          'id1': str(id1),
                                          'id2': None if id2 is None else str(id2)
                                          }
    else:
        # not the post action
        pass

with open(join(PROJECT_PATH, 'data', 'ID2ContentPost.json'), "w", encoding="utf-8") as fw:
    json.dump(ID2ContentPost, fw)
