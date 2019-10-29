import numpy as np
import matplotlib.pyplot as plt
import re
from collections import Counter
import collections as clts
import plotly.graph_objects as go

def PrepareText(filename):
    with open(filename, 'r') as file:
        text = file.read().replace(',', ' ').replace('.', ' ').replace('-', ' ').replace('\n', ' ')
    text = re.sub(r'[a-z]([A-Z])', r'-\1', text).lower()
    return re.compile('\w+').findall(text)

def nGramsVocab(words,n):
    vocab = dict()
    for i in range(len(words)-n+1):
        if " ".join(words[i:i+n]) in vocab:
            vocab[" ".join(words[i:i+n])]+=1
        else:
            vocab[" ".join(words[i:i+n])]=1
    return vocab

def VisualizeVocab(vocab, metrics):
    sortedVocab = sorted(vocab.items(), key=lambda pair: pair[1], reverse=True)
    vocabKeys, vocabValues = zip(*sortedVocab)

    if metrics == 'qnt':
         fig = go.Figure(data=[go.Table(header=dict(values=['N-грамма, n='+str(len(re.compile('\w+').findall(vocabKeys[1]))), 'Количество']),
                                                    cells=dict(values=[vocabKeys, vocabValues]))])
         fig.update_layout(width=500, height=3000)
         fig.show()

    if metrics == 'prob':
        vocabProb = list(vocabValues)
        length = sum(vocabValues)
        for i in range(len(vocabProb)):
            vocabProb[i]=round(vocabProb[i]/length,3)
        fig = go.Figure(data=[go.Table(header=dict(values=['N-грамма, n='+str(len(re.compile('\w+').findall(vocabKeys[1]))), 'Вероятность']),
                                                   cells=dict(values=[vocabKeys, vocabProb]))])
        fig.update_layout(width=500, height=3000)
        fig.show()

    if metrics == 'qnt':
         fig = go.Figure(data=[go.Table(header=dict(values=['N-грамма, n='+str(len(re.compile('\w+').findall(vocabKeys[1]))), 'Количество']),
                                                    cells=dict(values=[vocabKeys, vocabValues]))])
         fig.update_layout(width=500, height=3000)
         fig.show()

    if metrics == 'qntprob':
        vocabProb = list(vocabValues)
        length = sum(vocabValues)
        for i in range(len(vocabProb)):
            vocabProb[i]=round(vocabProb[i]/length,3)
        fig = go.Figure(data=[go.Table(header=dict(values=['N-грамма, n='+str(len(re.compile('\w+').findall(vocabKeys[1]))), 'Количество', 'Вероятность']),
                                                   cells=dict(values=[vocabKeys, vocabValues, vocabProb]))])
        fig.update_layout(width=500, height=3000)
        fig.show()

def LaplaceSmoothing1(train, words):
    trainVocab = nGramsVocab(train,1)
    VisualizeVocab(trainVocab, 'qntprob')
    n = len(re.compile('\w+').findall(list(trainVocab.keys())[1]))
    lengthTrain = sum(list(trainVocab.values()))
    testVocab = dict()

    #если слово из тестовой было в тренировочной, то берем частоту, если нет то ставим 0
    for i in range(len(test)):
        if " ".join(test[i:i+n]) in trainVocab:
            testVocab[" ".join(test[i:i+n])]=trainVocab[" ".join(test[i:i+n])]
        else:
            testVocab[" ".join(test[i:i+n])]=0

    testVocab = sorted(testVocab.items(), key=lambda pair: pair[1], reverse=True)
    testVocabKeys, testVocabValues = zip(*testVocab)

    prob = list(testVocabValues)
    addOne = list(testVocabValues)
    addOneProb = list(testVocabValues)

    #аддитивное сглаживание: в числителе +1, в знаменателе + V=|trainVocab|
    for i in range(len(prob)):
         prob[i]=round(prob[i]/len(train),3)
         addOne[i]+=1
         addOneProb[i]=round(addOne[i]/(len(train)+len(trainVocab)),3)

    fig = go.Figure(data=[go.Table(header=dict(values=['N-грамма, n='+str(n), 'Количество', 'Вероятность P', 'Сглаж Лапласа', 'Сглаж Лапласа Р']),
                                               cells=dict(values=[testVocabKeys, testVocabValues, prob, addOne, addOneProb]))])
    fig.update_layout(width=600, height=3000)
    fig.show()

    #подсчет перплексии
    pp = 1
    for i in range(len(addOneProb)):
        pp *= addOneProb[i]

    print('Сглаживание Лапласа. Перплексия для униграмм = ' + str(1/(pp**(1/len(addOneProb)))))

def LaplaceSmoothing2(train, test):
    prob = dict()
    qntVocab1 = dict()
    qntVocab2 = dict()
    #добавляем 2 unknown-слова к тренировочной выборке, чтобы впоследствии можно было вычислить перплексию
    train+=['UNK', 'UNK']

    #количественные словари униграмм и биграмм
    for i in range(len(train)):
        if " ".join(train[i:i+1]) in qntVocab1:
            qntVocab1[" ".join(train[i:i+1])]+=1
        else:
            qntVocab1[" ".join(train[i:i+1])]=1

    for i in range(len(train)-1):
        if " ".join(train[i:i+2]) in qntVocab2:
            qntVocab2[" ".join(train[i:i+2])]+=1
        else:
            qntVocab2[" ".join(train[i:i+2])]=1

    #матрица биграмм
    matrix = []
    k1 = 0

    for i in range(len(qntVocab1)+1):
        if i==0:
            matrix.append(list(qntVocab1.keys()))
        else:
            matrix.append(list(np.zeros(len(qntVocab1))))

    #сглаживание Лапласа: добавляем всем биграммам +1 и считаем сглаженные вероятности
    for w1 in qntVocab1:
        k2=0
        for w2 in qntVocab1:
            if w1+' '+w2 in qntVocab2:
                prob[w1+' '+w2] = round((qntVocab2[w1+' '+w2]+1)/(len(qntVocab1)),4)
            else:
                prob[w1+' '+w2] = round(1/(len(qntVocab1)),4)
            matrix[k1+1][k2] = str(prob[w1+' '+w2])
            k2+=1
        k1+=1

    fig = go.Figure(data=[go.Table(header=dict(values=list(' ')+list(qntVocab1.keys())),
                                               cells=dict(values=matrix))])
    fig.update_layout(width=1500, height=3000)
    fig.show()

    #подсчет перплексии
    for i in range(len(test)):
        if not(test[i] in list(qntVocab1.keys())):
            test[i] = 'UNK'

    pp=1
    for i in range(len(test)-1):
        pp *= prob[" ".join(test[i:i+2])]

    print('Сглаживание Лапласа. Перплексия для биграмм = ' + str(1/(pp**(1/len(test)))))

def WittenBell2Unk(train, test):
    n1Vocab = dict()
    lambdas = dict()
    prob = dict()

    #два наименее редких слова заменяем unknown-словами
    trainVocab1 = nGramsVocab(train,1)
    sortedtrainVocab1 = sorted(trainVocab1.items(), key=lambda pair: pair[1], reverse=True)
    sortedtrainVocab1Keys, sortedtrainVocab1Values = zip(*sortedtrainVocab1)
    toUnknown = sortedtrainVocab1Keys[len(sortedtrainVocab1Keys)-2:len(sortedtrainVocab1Keys)]

    for i in range(len(train)):
        if train[i] in toUnknown:
            train[i] = 'UNK'

    #либо можно просто добавить 2 unknown-слова в конец тренировочной выборки
    #train+=['UNK', 'UNK']

    trainVocab1 = nGramsVocab(train,1)
    trainVocab2 = nGramsVocab(train,2)
    n = len(re.compile('\w+').findall(list(trainVocab1.keys())[1]))

    #находим кол-во возможных продолжений для каждого слова из текста
    for key2 in trainVocab2:
            key1 = re.compile('\w+').findall(key2)[0]
            if key1 in n1Vocab:
                cnt=n1Vocab[key1]
            else:
                cnt=0
            n1Vocab[key1] = cnt+1

    #считаем коэфф lambda для всех униграмм
    for key1 in trainVocab1:
        lambdas[key1] = 1 - n1Vocab[key1]/(n1Vocab[key1]+trainVocab1[key1])

    #считаем вероятности всевозможных биграмм
    matrix = []

    for i in range(len(trainVocab1)+1):
        if i==0:
            matrix.append(list(trainVocab1.keys()))
        else:
            matrix.append(list(np.zeros(len(trainVocab1))))

    k1 = k2 = 0

    #строим матрицу вероятностей слов
    for w1 in trainVocab1:
        k2=0
        for w2 in trainVocab1:
            if w1+' '+w2 in trainVocab2:
                prob[w1+' '+w2] = round(lambdas[w1]*trainVocab2[w1+' '+w2]/trainVocab1[w1]+(1-lambdas[w1])*trainVocab1[w2]/(sum(list(trainVocab1.values()))), 3)
            else:
                prob[w1+' '+w2] = round((1-lambdas[w1])*trainVocab1[w2]/(sum(list(trainVocab1.values()))),3)
            matrix[k1+1][k2] = str(prob[w1+' '+w2])
            k2+=1
        k1+=1

    fig = go.Figure(data=[go.Table(header=dict(values=list(' ')+list(trainVocab1.keys())),
                                               cells=dict(values=matrix))])
    fig.update_layout(width=1500, height=3000)
    fig.show()

    #подсчет перплексии
    for i in range(len(test)):
        if not(test[i] in list(trainVocab1.keys())):
            test[i] = 'UNK'

    pp=1
    for i in range(len(test)-1):
        pp *= prob[" ".join(test[i:i+2])]

    print('Сглаживание Уиттена-Белла. Перплексия для биграмм = ' + str(1/(pp**(1/len(test)))))

def WittenBell2Lap(train, test):
    n1Vocab = dict()
    lambdas = dict()
    prob = dict()
    probtest = dict()

    trainVocab1 = nGramsVocab(train,1)
    trainVocab2 = nGramsVocab(train,2)

    n = len(re.compile('\w+').findall(list(trainVocab1.keys())[1]))

    #находим кол-во возможных продолжений для каждого слова из текста
    for key2 in trainVocab2:
            key1 = re.compile('\w+').findall(key2)[0]
            if key1 in n1Vocab:
                cnt=n1Vocab[key1]
            else:
                cnt=0
            n1Vocab[key1] = cnt+1

    #считаем коэфф lambda для всех униграмм
    for key1 in trainVocab1:
        lambdas[key1] = round(1 - n1Vocab[key1]/(n1Vocab[key1]+trainVocab1[key1]),3)

    lambdasSorted = sorted(lambdas.items(), key=lambda pair: pair[1], reverse=True)
    lambdasSortedKeys, lambdasSortedValues = zip(*lambdasSorted)
    fig = go.Figure(data=[go.Table(header=dict(values=['Униграмма', 'lambda_i']),
                                               cells=dict(values=[lambdasSortedKeys, lambdasSortedValues]))])
    fig.update_layout(width=600, height=3000)
    fig.show()

    #считаем вероятности всевозможных биграмм из тестовой выборки
    for i in range(len(test)-1):
        if " ".join(test[i:i+2]) in trainVocab2:#[kn kn]
            probtest[" ".join(test[i:i+2])]=round(lambdas[test[i:i+2][0]]*(trainVocab2[test[i:i+2][0]+' '+test[i:i+2][1]]+1)/(trainVocab1[test[i:i+2][0]]+len(trainVocab1)**2)+
                                      (1-lambdas[test[i:i+2][0]])*(trainVocab1[test[i:i+2][1]]+1)/(sum(list(trainVocab1.values()))+len(trainVocab1)), 3)
        elif (((test[i:i+2])[0] in trainVocab1) and (not(test[i:i+2])[1] in trainVocab1)):#[kn unk]
            probtest[" ".join(test[i:i+2])]=round(lambdas[test[i:i+2][0]]/(trainVocab1[test[i:i+2][0]]+len(trainVocab1)**2)+
                                      (1-lambdas[test[i:i+2][0]])/(sum(list(trainVocab1.values()))+len(trainVocab1)), 3)
        elif (not((test[i:i+2])[0] in trainVocab1) and ((test[i:i+2])[1] in trainVocab1)):#[unk kn]
            probtest[" ".join(test[i:i+2])]=round((trainVocab1[test[i:i+2][1]]+1)/(sum(list(trainVocab1.values()))+len(trainVocab1)), 3)
        elif not(" ".join(test[i:i+2]) in trainVocab2):#[unk unk]
            probtest[" ".join(test[i:i+2])]=round(1/len(trainVocab1)**2,3)

    probtestSorted = sorted(probtest.items(), key=lambda pair: pair[1], reverse=True)
    probtestSortedKeys, probtestSortedValues = zip(*probtestSorted)
    fig = go.Figure(data=[go.Table(header=dict(values=['Биграмма', 'Вероятность']),
                                               cells=dict(values=[probtestSortedKeys, probtestSortedValues]))])
    fig.update_layout(width=600, height=3000)
    fig.show()

    #подсчет перплексии
    pp=1
    for element in probtest:
        pp *= probtest[element]

    print('Сглаживание Уиттена-Белла. Перплексия для биграмм = ' + str(1/(pp**(1/len(test)))))

#подготовка данных
train = PrepareText('train.txt')
test = PrepareText('test.txt')

#аддитивное сглаживание для униграмм + перплексия
LaplaceSmoothing1(train, test)

#сглаживание Уиттена-Белла для биграмм + перплексия
WittenBell2Lap(train, test)
