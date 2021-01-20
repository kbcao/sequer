
import json
import pandas as pd
from os.path import join
from urllib import parse

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

ID2ContentSearch = {}
for _, row in event.iterrows():
    assert str(row['Id']) not in ID2ContentSearch.keys()
    if row['EventTarget'] == 'Search':
        validQuery = True
        if pd.isna(row['Query']) is True:
            validQuery = False
        else:
            testURL = str(row['Url'])
            testIdxQ = testURL.find('q=')
            if testIdxQ == -1:
                validQuery = False
            else:
                if testIdxQ + 2 >= len(testURL):
                    validQuery = False
                else:
                    if testURL[testIdxQ + 2] == '&':
                        validQuery = False
                    else:
                        pass
        if validQuery is True:
            url = str(row['Url'])
            keyword = ''
            page = 1
            tab = 'Relevance'
            # ATTENTION: find action must be done before the parser
            idxKeyword = url.find('q=')
            assert idxKeyword != -1
            if url.find('&', idxKeyword + 2) == -1:
                keyword = parse.unquote(url.replace('+', ' ')[idxKeyword + 2:])
            else:
                keyword = parse.unquote(url.replace('+', ' ')[idxKeyword + 2:url.find('&', idxKeyword + 2)])
            idxPage = url.find('page=')
            if idxPage != -1:
                if url.find('&', idxPage + 5) == -1:
                    page = int(url[idxPage + 5:])
                else:
                    page = int(url[idxPage + 5:url.find('&', idxPage + 5)])
            idxTab = url.find('tab=')
            if idxTab != -1:
                if url.find('&', idxTab + 4) == -1:
                    tab = url[idxTab + 4:]
                else:
                    tab = url[idxTab + 4:url.find('&', idxTab + 4)]
            assert len(keyword) >= 1, 'Illegal length of keyword'
            assert page >= 1, 'Illegal page'
            assert len(tab) >= 4, 'Illegal length of tab'
        else:
            keyword = None
            page = None
            tab = None
        ID2ContentSearch[str(row['Id'])] = {'reid': None if pd.isna(row['RootEventId']) else str(row['RootEventId']),
                                            'user': str(row['UserIdentifier']),
                                            'time': str(row['CreationDate']),
                                            'source': str(row['EventSource']),
                                            'target': str(row['EventTarget']),
                                            'keyword': None if keyword is None else str(keyword),
                                            'page': None if page is None else str(page),
                                            'tab': None if tab is None else str(tab)
                                            }
    else:
        # not the search action
        pass

with open(join(PROJECT_PATH, 'data', 'ID2ContentSearch.json'), "w", encoding="utf-8") as fw:
    json.dump(ID2ContentSearch, fw)
