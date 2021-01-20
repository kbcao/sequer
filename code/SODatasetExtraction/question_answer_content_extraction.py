import json
from os.path import join
from lxml import etree

PROJECT_PATH = '../../'

postsXML = etree.iterparse(join(PROJECT_PATH, 'data', 'Posts.xml'), encoding="utf-8")
with open(join(PROJECT_PATH, 'data', 'ID2ContentPost.json'), mode="r", encoding="utf-8") as fr:
    ID2ContentPost = dict(json.load(fr))

with open(join(PROJECT_PATH, 'data', 'AID2QID.json'), mode="r", encoding="utf-8") as fr:
    AID2QID = dict(json.load(fr))

AID2QIDKeys = AID2QID.keys()

questionAttribute = ["AcceptedAnswerId",
                     "CreationDate",
                     "Score",
                     "ViewCount",
                     "Body",
                     "Title",
                     "Tags",
                     "AnswerCount",
                     "CommentCount",
                     "FavoriteCount"]
answerAttribute = ["ParentId",
                   "CreationDate",
                   "Score",
                   "Body",
                   "CommentCount"]

questionID = set()
answerID = set()

for key in ID2ContentPost.keys():
    linkType = ID2ContentPost[key]['type'][0]
    if linkType == 'q':
        questionID.add(ID2ContentPost[key]['id1'])
    elif linkType == 'a':
        aid = ID2ContentPost[key]['id1']
        answerID.add(aid)
        if aid in AID2QIDKeys:
            questionID.add(AID2QID[aid])
    else:
        raise KeyError

questionContent = {}
answerContent = {}

workIdx = 0
for _, elem in postsXML:
    workIdx += 1
    if workIdx % 10000 == 0:
        print('processing: {}'.format(workIdx))
    ID = str(elem.get('Id'))
    typeID = str(elem.get('PostTypeId'))
    if ID in questionID and typeID == '1':
        questionContent[ID] = {i: elem.get(i) for i in questionAttribute}
    if ID in answerID and typeID == '2':
        answerContent[ID] = {i: elem.get(i) for i in answerAttribute}
    elem.clear()
    for ancestor in elem.xpath('ancestor-or-self::*'):
        while ancestor.getprevious() is not None:
            del ancestor.getparent()[0]

with open(join(PROJECT_PATH, 'data', 'questionContent.json'), "w", encoding="utf-8") as fw:
    json.dump(questionContent, fw)

with open(join(PROJECT_PATH, 'data', 'answerContent.json'), "w", encoding="utf-8") as fw:
    json.dump(answerContent, fw)

print('Length of question provided: {}'.format(len(questionContent.keys())))
print('Length of question needed: {}'.format(len(questionID)))
print('Length of answer provided: {}'.format(len(answerContent.keys())))
print('Length of answer needed: {}'.format(len(answerID)))

print('qid can not be found')
providedQID = questionContent.keys()
for qid in questionID:
    if qid not in providedQID:
        print(qid)

print('aid can not be found')
providedAID = answerContent.keys()
for aid in answerID:
    if aid not in providedAID:
        print(aid)
