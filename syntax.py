# coding: utf-8

import os
import shutil
import wget
from ufal.udpipe import Model, Pipeline


# URL of the UDPipe model
udpipe_model_url = 'https://rusvectores.org/static/models/udpipe_syntagrus.model'
udpipe_filename = "models/udpipe_syntagrus.model"


print('Загрузка синтаксической модели...')
if not os.path.isfile(udpipe_filename):
    wget.download(udpipe_model_url)
    shutil.move("udpipe_syntagrus.model", "models/udpipe_syntagrus.model")


model = Model.load(udpipe_filename)
process_pipeline = Pipeline(model, 'tokenize', Pipeline.DEFAULT, Pipeline.DEFAULT, 'conllu')


datadict = {}


class UdpipeFeatues:
    def __init__(self, lemmas, syntax_bigrams, syntax_trigrams, syntax_pos, only_poses, words_poses):
        self.lemmas = lemmas
        self.syntax_bigrams = syntax_bigrams
        self.syntax_trigrams = syntax_trigrams
        self.syntax_pos = syntax_pos
        self.only_poses = only_poses
        self.words_poses = words_poses


def num_replace(word):
    newtoken = 'x' * len(word)
    return newtoken


def clean_lemma(lemma, pos):
    """
    :param lemma: лемма (строка)
    :param pos: часть речи (строка)
    :return: очищенная лемма (строка)
    """
    out_lemma = lemma.strip().replace(' ', '').replace('_', '').lower()
    if '|' in out_lemma:
        return None
    if pos != 'PUNCT':
        if out_lemma.startswith('«') or out_lemma.startswith('»'):
            out_lemma = ''.join(out_lemma[1:])

        if out_lemma.endswith('«') or out_lemma.endswith('»'):
            out_lemma = ''.join(out_lemma[:-1])

        if out_lemma.endswith('!') or out_lemma.endswith('?') or out_lemma.endswith(',') \
                or out_lemma.endswith('.'):
            out_lemma = ''.join(out_lemma[:-1])

    return out_lemma


def clean_token(token, misc):
    """
    :param token:  токен (строка)
    :param misc:  содержимое поля "MISC" в CONLLU (строка)
    :return: очищенный токен (строка)
    """
    out_token = token.strip().replace(' ', '')
    if token == 'Файл' and 'SpaceAfter=No' in misc:
        return None
    return out_token


def list_replace(search, replacement, text):
    search = [el for el in search if el in text]
    for c in search:
        text = text.replace(c, replacement)
    return text


def clean_text(text):

    allowed = list("\t\n\r") \
              + list("абвгдеёзжийклмнопрстуфхцчшщьыъэюяАБВГДЕЁЗЖИЙКЛМНОПРСТУФХЦЧШЩЬЫЪЭЮЯ") \
              + list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ') \
              + list(' ,.[]{}()=+-−*&^%$#@!?~;:0123456789§/\|"') \
              + list("'")

    allowed = set(allowed)

    cleaned_text = [sym for sym in text if sym in allowed]
    cleaned_text = ''.join(cleaned_text)

    return cleaned_text


def process(pipeline, text) -> UdpipeFeatues:

    keep_punct = False

    tagged_res = []

    # обрабатываем текст, получаем результат в формате conllu:
    processed = pipeline.process(text)

    # пропускаем строки со служебной информацией:
    content = [l for l in processed.split('\n') if not l.startswith('#')]

    # извлекаем из обработанного текста леммы, тэги и морфологические характеристики
    tagged = [w.split('\t') for w in content if w]

    for t in tagged:
        if len(t) != 10:
            continue
        (word_id, token, lemma, pos, xpos, feats, head, deprel, deps, misc) = t
        lemma = clean_lemma(lemma, pos)
        head = int(head)
        if head == 0:
            parent = None
        else:
            parent = (tagged[head-1][2], tagged[head-1][3], deprel)

        tagged_res.append((lemma, pos, parent))

    only_words = []
    syntax_bigrams = []
    syntax_trigrams = []
    syntax_pos = []
    only_poses = []
    words_poses = []
    for t in tagged_res:
        if (not keep_punct) and (t[1] == "PUNCT" or ((t[2] is not None) and t[2][1] == "PUNCT")):
            continue
        only_words.append(t[0])
        syntax_bigrams.append(t[0] if t[2] is None else f"{t[2][0]}_{t[0]}")
        syntax_trigrams.append(t[0] if t[2] is None else f"{t[2][0]}_{t[0]}_{t[2][2]}")
        syntax_pos.append(t[1] if t[2] is None else f"{t[2][1]}_{t[1]}")
        only_poses.append(t[1])
        words_poses.append(f"{t[0]}_{t[1]}")


    return UdpipeFeatues(" ".join(only_words),
                         " ".join(syntax_bigrams),
                         " ".join(syntax_trigrams),
                         " ".join(syntax_pos),
                         " ".join(only_poses),
                         " ".join(words_poses))


def process_data(data):
    result = []
    for text in data:
        if text in datadict:
            result.append(datadict[text])
        else:
            res = clean_text(text.strip())
            res = process(process_pipeline, text=res)
            datadict[text] = res
            result.append(res)
    return result
