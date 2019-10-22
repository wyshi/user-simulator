# -*- coding: utf-8 -*-
import copy
import json
import os
import re
import shutil
import urllib
from collections import OrderedDict
from io import BytesIO
from zipfile import ZipFile
from tqdm import tqdm

import numpy as np

from simulator.multiwoz_utils.utils import dbPointer
from simulator.multiwoz_utils.utils import delexicalize

from simulator.multiwoz_utils.utils.nlp import normalize

np.set_printoptions(precision=3)

np.random.seed(2)

# GLOBAL VARIABLES
DICT_SIZE = 400
MAX_LENGTH = 50


def is_ascii(s):
    return all(ord(c) < 128 for c in s)


def fixDelex(filename, data, data2, idx, idx_acts):
    """Given system dialogue acts fix automatic delexicalization."""
    try:
        turn = data2[filename.strip('.json')][str(idx_acts)]
    except:
        return data

    if not isinstance(turn, str) and not isinstance(turn, unicode):
        for k, act in turn.items():
            if 'Attraction' in k:
                if 'restaurant_' in data['log'][idx]['text']:
                    data['log'][idx]['text'] = data['log'][idx]['text'].replace("restaurant", "attraction")
                if 'hotel_' in data['log'][idx]['text']:
                    data['log'][idx]['text'] = data['log'][idx]['text'].replace("hotel", "attraction")
            if 'Hotel' in k:
                if 'attraction_' in data['log'][idx]['text']:
                    data['log'][idx]['text'] = data['log'][idx]['text'].replace("attraction", "hotel")
                if 'restaurant_' in data['log'][idx]['text']:
                    data['log'][idx]['text'] = data['log'][idx]['text'].replace("restaurant", "hotel")
            if 'Restaurant' in k:
                if 'attraction_' in data['log'][idx]['text']:
                    data['log'][idx]['text'] = data['log'][idx]['text'].replace("attraction", "restaurant")
                if 'hotel_' in data['log'][idx]['text']:
                    data['log'][idx]['text'] = data['log'][idx]['text'].replace("hotel", "restaurant")

    return data


def delexicaliseReferenceNumber(sent, turn):
    """Based on the belief state, we can find reference number that
    during data gathering was created randomly."""
    domains = ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'hospital']  # , 'police']
    if turn['metadata']:
        for domain in domains:
            if turn['metadata'][domain]['book']['booked']:
                for slot in turn['metadata'][domain]['book']['booked'][0]:
                    if slot == 'reference':
                        val = '[' + domain + '_' + slot + ']'
                    else:
                        val = '[' + domain + '_' + slot + ']'
                    key = normalize(turn['metadata'][domain]['book']['booked'][0][slot])
                    sent = (' ' + sent + ' ').replace(' ' + key + ' ', ' ' + val + ' ')

                    # try reference with hashtag
                    key = normalize("#" + turn['metadata'][domain]['book']['booked'][0][slot])
                    sent = (' ' + sent + ' ').replace(' ' + key + ' ', ' ' + val + ' ')

                    # try reference with ref#
                    key = normalize("ref#" + turn['metadata'][domain]['book']['booked'][0][slot])
                    sent = (' ' + sent + ' ').replace(' ' + key + ' ', ' ' + val + ' ')
    return sent


def addBookingPointer(task, turn, pointer_vector):
    """Add information about availability of the booking option."""
    # Booking pointer
    rest_vec = np.array([1, 0])
    if task['goal']['restaurant']:
        if turn['metadata']['restaurant'].has_key("book"):
            if turn['metadata']['restaurant']['book'].has_key("booked"):
                if turn['metadata']['restaurant']['book']["booked"]:
                    if "reference" in turn['metadata']['restaurant']['book']["booked"][0]:
                        rest_vec = np.array([0, 1])

    hotel_vec = np.array([1, 0])
    if task['goal']['hotel']:
        if turn['metadata']['hotel'].has_key("book"):
            if turn['metadata']['hotel']['book'].has_key("booked"):
                if turn['metadata']['hotel']['book']["booked"]:
                    if "reference" in turn['metadata']['hotel']['book']["booked"][0]:
                        hotel_vec = np.array([0, 1])

    train_vec = np.array([1, 0])
    if task['goal']['train']:
        if turn['metadata']['train'].has_key("book"):
            if turn['metadata']['train']['book'].has_key("booked"):
                if turn['metadata']['train']['book']["booked"]:
                    if "reference" in turn['metadata']['train']['book']["booked"][0]:
                        train_vec = np.array([0, 1])

    pointer_vector = np.append(pointer_vector, rest_vec)
    pointer_vector = np.append(pointer_vector, hotel_vec)
    pointer_vector = np.append(pointer_vector, train_vec)

    return pointer_vector


def addDBPointer(turn):
    """Create database pointer for all related domains."""
    domains = ['restaurant', 'hotel', 'attraction', 'train']
    pointer_vector = np.zeros(6 * len(domains))
    for domain in domains:
        num_entities = dbPointer.queryResult(domain, turn)
        pointer_vector = dbPointer.oneHotVector(num_entities, domain, pointer_vector)

    return pointer_vector


def get_summary_bstate(bstate):
    """Based on the mturk annotations we form multi-domain belief state"""
    domains = [u'taxi', u'restaurant', u'hospital', u'hotel', u'attraction', u'train', u'police']
    summary_bstate = []
    for domain in domains:
        domain_active = False

        booking = []
        # print(domain,len(bstate[domain]['book'].keys()))
        for slot in sorted(bstate[domain]['book'].keys()):
            if slot == 'booked':
                if bstate[domain]['book']['booked']:
                    booking.append(1)
                else:
                    booking.append(0)
            else:
                if bstate[domain]['book'][slot] != "":
                    booking.append(1)
                else:
                    booking.append(0)
        if domain == 'train':
            if 'people' not in bstate[domain]['book'].keys():
                booking.append(0)
            if 'ticket' not in bstate[domain]['book'].keys():
                booking.append(0)
        summary_bstate += booking

        for slot in bstate[domain]['semi']:
            slot_enc = [0, 0, 0]  # not mentioned, dontcare, filled
            if bstate[domain]['semi'][slot] == 'not mentioned':
                slot_enc[0] = 1
            elif bstate[domain]['semi'][slot] == 'dont care' or bstate[domain]['semi'][slot] == 'dontcare' or \
                    bstate[domain]['semi'][slot] == "don't care":
                slot_enc[1] = 1
            elif bstate[domain]['semi'][slot]:
                slot_enc[2] = 1
            if slot_enc != [0, 0, 0]:
                domain_active = True
            summary_bstate += slot_enc

        # quasi domain-tracker
        if domain_active:
            summary_bstate += [1]
        else:
            summary_bstate += [0]

    # print(len(summary_bstate))
    assert len(summary_bstate) == 94
    return summary_bstate


def analyze_dialogue(dialogue, maxlen):
    """Cleaning procedure for all kinds of errors in text and annotation."""
    d = dialogue
    # do all the necessary postprocessing
    if len(d['log']) % 2 != 0:
        # print path
        print
        'odd # of turns'
        return None  # odd number of turns, wrong dialogue
    d_pp = {}
    d_pp['goal'] = d['goal']  # for now we just copy the goal
    usr_turns = []
    sys_turns = []
    for i in range(len(d['log'])):
        if len(d['log'][i]['text'].split()) > maxlen:
            print
            'too long'
            return None  # too long sentence, wrong dialogue
        if i % 2 == 0:  # usr turn
            if 'db_pointer' not in d['log'][i]:
                print
                'no db'
                return None  # no db_pointer, probably 2 usr turns in a row, wrong dialogue
            text = d['log'][i]['text']
            if not is_ascii(text):
                print
                'not ascii'
                return None
            # d['log'][i]['tkn_text'] = self.tokenize_sentence(text, usr=True)
            usr_turns.append(d['log'][i])
        else:  # sys turn
            text = d['log'][i]['text']
            if not is_ascii(text):
                print
                'not ascii'
                return None
            # d['log'][i]['tkn_text'] = self.tokenize_sentence(text, usr=False)
            belief_summary = get_summary_bstate(d['log'][i]['metadata'])
            d['log'][i]['belief_summary'] = belief_summary
            sys_turns.append(d['log'][i])
    d_pp['usr_log'] = usr_turns
    d_pp['sys_log'] = sys_turns

    return d_pp


def get_dial(dialogue):
    """Extract a dialogue from the file"""
    dial = []
    d_orig = analyze_dialogue(dialogue, MAX_LENGTH)  # max turn len is 50 words
    if d_orig is None:
        return None
    usr = [t['text'] for t in d_orig['usr_log']]
    db = [t['db_pointer'] for t in d_orig['usr_log']]
    bs = [t['belief_summary'] for t in d_orig['sys_log']]
    sys = [t['text'] for t in d_orig['sys_log']]
    for u, d, s, b in zip(usr, db, sys, bs):
        dial.append((u, s, d, b))

    return dial


def createDict(word_freqs):
    words = word_freqs.keys()
    freqs = word_freqs.values()

    sorted_idx = np.argsort(freqs)
    sorted_words = [words[ii] for ii in sorted_idx[::-1]]

    # Extra vocabulary symbols
    _GO = '_GO'
    EOS = '_EOS'
    UNK = '_UNK'
    PAD = '_PAD'
    extra_tokens = [_GO, EOS, UNK, PAD]

    worddict = OrderedDict()
    for ii, ww in enumerate(extra_tokens):
        worddict[ww] = ii
    for ii, ww in enumerate(sorted_words):
        worddict[ww] = ii + len(extra_tokens)

    for key, idx in worddict.items():
        if idx >= DICT_SIZE:
            del worddict[key]

    return worddict


def loadData():
    data_url = "data/multiwoz-master/data/multi-woz/data.json"
    dataset_url = "https://www.repository.cam.ac.uk/bitstream/handle/1810/280608/MULTIWOZ2.zip?sequence=3&isAllowed=y"
    if not os.path.exists("data"):
        os.makedirs("data")
        os.makedirs("data/multi-woz")

    if not os.path.exists(data_url):
        print("Downloading and unzipping the MultiWOZ dataset")
        resp = urllib.urlopen(dataset_url)
        zip_ref = ZipFile(BytesIO(resp.read()))
        zip_ref.extractall("data/multi-woz")
        zip_ref.close()
        shutil.copy('data/multi-woz/MULTIWOZ2 2/data.json', 'data/multi-woz/')
        shutil.copy('data/multi-woz/MULTIWOZ2 2/valListFile.json', 'data/multi-woz/')
        shutil.copy('data/multi-woz/MULTIWOZ2 2/testListFile.json', 'data/multi-woz/')
        shutil.copy('data/multi-woz/MULTIWOZ2 2/dialogue_acts.json', 'data/multi-woz/')


def createDelexData():
    """Main function of the script - loads delexical dictionary,
    goes through each dialogue and does:
    1) data normalization
    2) delexicalization
    3) addition of database pointer
    4) saves the delexicalized data
    """
    # download the data
    loadData()

    # create dictionary of delexicalied values that then we will search against, order matters here!
    dic = delexicalize.prepareSlotValuesIndependent()


    fin1 = file('data/multi-woz/data.json')
    data = json.load(fin1)

    fin2 = file('data/multi-woz/dialogue_acts.json')
    data2 = json.load(fin2)

    for dialogue_name in tqdm(data):
        dialogue = data[dialogue_name]
        # print dialogue_name

        idx_acts = 1

        for idx, turn in enumerate(dialogue['log']):
            # normalization, split and delexicalization of the sentence
            sent = normalize(turn['text'])

            words = sent.split()
            sent = delexicalize.delexicalise(' '.join(words), dic)

            # parsing reference number GIVEN belief state
            sent = delexicaliseReferenceNumber(sent, turn)

            # changes to numbers only here
            digitpat = re.compile('\d+')
            sent = re.sub(digitpat, '[value_count]', sent)

            # delexicalized sentence added to the dialogue
            dialogue['log'][idx]['text'] = sent

            if idx % 2 == 1:  # if it's a system turn
                # add database pointer
                pointer_vector = addDBPointer(turn)
                # add booking pointer
                pointer_vector = addBookingPointer(dialogue, turn, pointer_vector)

                # print pointer_vector
                dialogue['log'][idx - 1]['db_pointer'] = pointer_vector.tolist()

            # FIXING delexicalization:
            dialogue = fixDelex(dialogue_name, dialogue, data2, idx, idx_acts)
            idx_acts += 1

        delex_data[dialogue_name] = dialogue

    with open('data/multi-woz/delex.json', 'w') as outfile:
        json.dump(delex_data, outfile)

    return delex_data


def main():
    print('Create delexicalized dialogues. Get yourself a coffee, this might take a while.')
    delex_data = createDelexData()


def delexicalize_one_sent0(sent, dic):
    sent, kv_dic = normalize(sent)

    words = sent.split()
    sent, kv_dic_tmp = delexicalize.delexicalise(' '.join(words), dic)

    kv_dic.update(kv_dic_tmp)
    # parsing reference number GIVEN belief state
    # sent = delexicaliseReferenceNumber(sent, turn)

    # changes to numbers only here
    # digitpat = re.compile('(?<!looking for)(?<=for) \d+ (?!of)|(?<=party of) \d+ | \d+ (?=people|person|of us)')
    # # digitpat = re.compile(' \d+ ')
    # value_count = re.findall(digitpat, sent)
    # while value_count:
    #
    #     index = sent.find(value_count[0])
    #
    #     if not delexicalize.check_balance(sent[:index]):
    #         # pdb.set_trace()
    #         value_count.pop(0)
    #         continue
    #
    #     sent = sent[:index] + \
    #            ' [value_count|' + value_count[0][1:-1] + '] ' + \
    #            sent[index + len(value_count[0]):]
    #     value_count = re.findall(digitpat, sent)




    digitpat = re.compile('\d+')
    digits = re.findall(digitpat, sent)
    if len(digits):
        kv_dic.update({'[value_count]': digits})
    sent = re.sub(digitpat, '[value_count]', sent)





    return sent, kv_dic

import sys, os, re, pdb
import pickle as pkl

import sys, os, re, pdb
from nltk import word_tokenize


def delexicalize_one_sent(sent):
    def normalize(text):
        def insertSpace(token, text):
            sidx = 0
            while True:
                sidx = text.find(token, sidx)
                if sidx == -1:
                    break
                if sidx + 1 < len(text) and re.match('[0-9]', text[sidx - 1]) and \
                        re.match('[0-9]', text[sidx + 1]):
                    sidx += 1
                    continue
                if text[sidx - 1] != ' ':
                    text = text[:sidx] + ' ' + text[sidx:]
                    sidx += 1
                if sidx + len(token) < len(text) and text[sidx + len(token)] != ' ':
                    text = text[:sidx + 1] + ' ' + text[sidx + 1:]
                sidx += 1
            return text

        # lower case every word
        text = text.lower()

        # replace white spaces in front and end
        text = re.sub(r'^\s*|\s*$', '', text)

        # hotel domain pfb30
        text = re.sub(r"b&b", "bed and breakfast", text)
        text = re.sub(r"b and b", "bed and breakfast", text)

        # normalize phone number
        ms = re.findall('\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4,5})', text)
        if ms:
            sidx = 0
            for m in ms:
                sidx = text.find(m[0], sidx)
                if text[sidx - 1] == '(':
                    sidx -= 1
                eidx = text.find(m[-1], sidx) + len(m[-1])
                text = text.replace(text[sidx:eidx], ''.join(m))

        # normalize postcode
        ms = re.findall(
            '([a-z]{1}[\. ]?[a-z]{1}[\. ]?\d{1,2}[, ]+\d{1}[\. ]?[a-z]{1}[\. ]?[a-z]{1}|[a-z]{2}\d{2}[a-z]{2})',
            text)
        if ms:
            sidx = 0
            for m in ms:
                sidx = text.find(m, sidx)
                eidx = sidx + len(m)
                text = text[:sidx] + re.sub('[,\. ]', '', m) + text[eidx:]

        # weird unicode bug
        text = re.sub(u"(\u2018|\u2019)", "'", text)

        text = ' ' + text + ' '
        # # replace time and and price
        timepat = re.compile(
            " \d{1,2}[:]\d{1,2}[ \.,\?]| \d{4}[ \.,\?]| \d{1,2}[ap][m\. ]+| \d{1,2} [ap][m\. ]+| \d{1,2}[:]\d{1,2}[ap]m[ \.,\?]")
        # # some utterances just miss the ":"
        # timepat_noise = re.compile(" \d{4}[ \.,\?]")
        pricepat = re.compile("\d{1,3}[\.]\d{1,2}")

        value_time = re.findall(timepat, text)
        # pdb.set_trace()
        while value_time:
            index = text.find(value_time[0])
            text = text[:index] + \
                   ' [value_time|' + value_time[0][1:-1] + ']' + \
                   text[index + len(value_time[0]) - 1:]
            value_time = re.findall(timepat, text)

        value_price = re.findall(pricepat, text)

        if value_price:
            text = re.sub(pricepat, ' [value_price|' + value_price[0] + '] ', text)

        text = text[1:-1]

        # replace st.
        text = text.replace(';', ',')
        text = re.sub('$\/', '', text)
        text = text.replace('/', ' and ')

        # replace other special characters
        text = text.replace('-', ' ')
        text = re.sub('[\"\<>@\(\)]', '', text)

        # insert white space before and after tokens:
        for token in ['?', '.', ',', '!']:
            text = insertSpace(token, text)

        # insert white space for 's
        text = insertSpace('\'s', text)

        # replace it's, does't, you'd ... etc
        text = re.sub('^\'', '', text)
        text = re.sub('\'$', '', text)
        text = re.sub('\'\s', ' ', text)
        text = re.sub('\s\'', ' ', text)

        fin = open('/data/qkun/simulator/data/multiwoz-master/utils/mapping.pair')
        replacements = []
        for line in fin.readlines():
            tok_from, tok_to = line.replace('\n', '').split('\t')
            replacements.append((' ' + tok_from + ' ', ' ' + tok_to + ' '))

        for fromx, tox in replacements:
            text = ' ' + text + ' '
            text = text.replace(fromx, tox)[1:-1]

        # remove multiple spaces
        text = re.sub(' +', ' ', text)

        # concatenate numbers
        tmp = text
        tokens = text.split()
        i = 1
        while i < len(tokens):
            if re.match(u'^\d+$', tokens[i]) and \
                    re.match(u'\d+$', tokens[i - 1]):
                tokens[i - 1] += tokens[i]
                del tokens[i]
            else:
                i += 1
        text = ' '.join(tokens)

        return text

    def check_balance(string):
        # open_tup = tuple('[')
        # close_tup = tuple(']')
        # map = dict(zip(open_tup, close_tup))
        queue = 0

        for i in string:
            if i == '[':
                queue += 1
            elif i == ']':
                if not queue:
                    return False
                else:
                    queue -= 1
        if not queue:
            return True
        else:
            return False

    def delexicalise(utt, dictionary):
        for key, val in dictionary:
            utt = (' ' + utt + ' ')
            if key in utt:
                idx = 0
                while utt[idx:].find(' ' + key + ' ') != -1:
                    idx += utt[idx:].find(' ' + key + ' ')
                    # # to exclude the case that 'ask' is a verb
                    if key == 'ask' and idx > 2 and utt[idx - 2:idx] == ' i':
                        idx += 1
                        continue
                    if check_balance(utt[:idx]):
                        utt = utt[:idx] + ' ' + val[:-1] + '|' + key + '] ' + utt[idx + len(key) + 2:]
                        idx += len(key) + 4 + len(val[:-1])
                    else:
                        idx += len(key)
            utt = utt[1:-1]

        return utt

    def delex_people_count(sent):
        sent = ' ' + sent + ' '
        digitpat = re.compile('(?<!looking for)(?<=for) \d+ (?!of)|(?<=party of) \d+ | \d+ (?=people|person|of us)')
        value_count = re.findall(digitpat, sent)
        while value_count:
            index = sent.find(value_count[0])
            if not check_balance(sent[:index]):
                value_count.pop(0)
                continue

            sent = sent[:index] + \
                   ' [value_count|' + value_count[0][1:-1] + '] ' + \
                   sent[index + len(value_count[0]):]
            value_count = re.findall(digitpat, sent)
        sent = sent[1:-1]
        return sent

    # # # extract slots
    # # replace time, date, specific price
    sent = ' '.join(word_tokenize(sent))

    sent = normalize(sent)
    # pdb.set_trace()
    # # replace info in db
    db_entity_file = open('/data/qkun/simulator/data/multiwoz-master/db_entity_file.pkl', 'rb')
    db_entity_list = pkl.load(db_entity_file)
    db_entity_file.close()
    sent = delexicalise(sent, db_entity_list)

    # # replace # of people for reservation
    sent = delex_people_count(sent)

    # # # replace and generate dic
    slotpat = re.compile('\[.*?\]')
    slots = re.findall(slotpat, sent)
    slot_dic = {}

    for slot in slots:
        if '|' in slot:
            [slot_name, slot_val] = slot[1:-1].split('|')
            slot_name = '[' + slot_name + ']'
            slot_dic[slot_name] = slot_val
            sent = sent.replace(slot, slot_name)

    sent = sent.replace('[', "").replace(']', "")
    return sent, slot_dic

if __name__ == "__main__":
    delexicalize_one_sent()