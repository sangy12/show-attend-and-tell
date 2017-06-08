#-*- coding:utf-8 -*-
from collections import Counter
# from core.vggnet import Vgg19
from core.utils import *

import random
# import tensorflow as tf
import os


def _process_caption_data(caption_file, set_type):
    count = 0
    captions = []
    cap_per_image = []
    with open(caption_file, 'r') as infile:
        for line in infile:
            if not count:
                count += 1
                continue
            if line.strip().isdigit():
                captions.append(cap_per_image)
                cap_per_image = []
                count += 1
                continue
            count += 1
            line = line.decode('utf8')
            cap = line.lower().strip().replace('/', ' ').split()
            cap_per_image.append(cap)
        captions.append(cap_per_image)


    print('read in line num = %d, image num = %d' % (count, len(captions)))
    save_pickle(captions, os.path.join('new_data/', set_type, 'caption_%s.pickle' % set_type))
    print ('%s caption word cut pickle saved' % set_type)
    return captions

def _build_vocab(annotations, threshold=1):
    counter = Counter()
    max_length = 0

    print('\n*** debug: caption when max_length is getting larger')
    for  caption_per_img in annotations:
        for caption in caption_per_img:
            # debug
            if (max_length <  len(caption)):
                max_length = len(caption)
                print("%d: %s" %(max_length, " ".join(caption)))
            for w in caption:
                counter[w] +=1
    print('*********\n')


    print ('max word num is %d' % max_length)
    vocab = [word for word in counter if counter[word] >= threshold]
    print ('Filtered %d words to %d words with word count threshold %d.' % (len(counter), len(vocab), threshold))

    word_to_idx = {u'<NULL>': 0, u'<START>': 1, u'<END>': 2}
    idx = 3
    for word in vocab:
        word_to_idx[word] = idx
        idx += 1

    return word_to_idx

def _build_caption_vector(annotations, word_to_idx, max_length=15):
    captions_vec = []

    for caption_per_image in annotations:
        vec_per_image = []
        for caption in caption_per_image:
            cap_vec = []
            cap_vec.append(word_to_idx['<START>'])
            count = 0
            for word in caption:
                if (count < max_length) and (word in word_to_idx):
                    cap_vec.append(word_to_idx[word])
                    count += 1
            cap_vec.append(word_to_idx['<END>'])
            # pad short caption with the special null token '<NULL>' to make it fixed-size vector
            if len(cap_vec) < (max_length + 2):
                for j in range(max_length + 2 - len(cap_vec)):
                    cap_vec.append(word_to_idx['<NULL>'])
            vec_per_image.append(cap_vec)

        captions_vec.append(vec_per_image)

    print "Finished building caption vectors"
    return captions_vec

def _process_validate_reference(caption_file):
    count = 0
    captions = {}
    cap_per_image = []
    set_type = 'val'
    with open(caption_file, 'r') as infile:
        for line in infile:
            if not count:
                count += 1
                continue
            if line.strip().isdigit():
                captions[count-1] = cap_per_image
                cap_per_image = []
                count += 1
                continue
            # count += 1
            line = line.decode('utf8')
            cap = line.strip().lower()
            cap = list(cap)
            cap = " ".join(cap)
            cap_per_image.append(cap)
        captions[count-1] = cap_per_image

    save_pickle(captions, os.path.join('new_data/', set_type, '%s.references.pkl' % set_type))
    print ('%s caption references pickled' % set_type)

if __name__ == "__main__":

    # if word occurs less than word_count_threshold in training dataset, the word index is special unknown token.
    word_count_threshold = 4

    ### Preprocessing step 1: use the jieba chinese word cut tool to cut the captions

    ### Preprocessing step 2: collect the word-cut to vacabulary
    train_captions = _process_caption_data(caption_file='new_data/train_cut.txt', set_type='train')
    val_captions = _process_caption_data(caption_file='new_data/valid_cut.txt', set_type='val')

    #debug
    print('\n*** debug: random choose some captions')
    rand_idx = random.sample(xrange(len(train_captions)),10)
    rand_cap = [train_captions[idx][0] for idx in rand_idx]
    for k in range(10):
        print ' '.join(rand_cap[k])
    print('******\n')


    ### Preprocessing step 3: collect the word-cut to vacabulary
    word_to_idx = _build_vocab(annotations=train_captions, threshold=word_count_threshold)
    save_pickle(word_to_idx, 'new_data/train/word_to_idx.pkl')
    #debug
    print('\n*** debug: print vocabulary')
    for word,vec in word_to_idx.iteritems():
        print("%d: %s" %(vec, word))
    print('******\n')


    # max length of word-cut in a sentence
    max_length = 15
    ### Preprocessing step 4: convert the captions to vectors
    captions = _build_caption_vector(annotations=train_captions, word_to_idx=word_to_idx, max_length=max_length)
    save_pickle(captions, './new_data/train/train_captions_vec.pkl')
    #debug
    print('\n***debug: reconstruct from vec')
    vec_to_word = {}
    for key, value in word_to_idx.iteritems():
        vec_to_word[value] = key
    rand_vec = [captions[idx][0] for idx in rand_idx]
    for k in range(10):
        print rand_vec[k]
        for v in rand_vec[k]:
            print vec_to_word[v],
        print ('\n')
    print('******\n')

    captions = _build_caption_vector(annotations=val_captions, word_to_idx=word_to_idx, max_length=max_length)
    save_pickle(captions, './new_data/val/val_captions_vec.pkl')

    ### Preprocessing step 4: build the reference validation captions for bleu measurement
    _process_validate_reference('./new_data/valid.txt')


    print('\n***debug: check some word')
    word = u"t恤"
    print word, word_to_idx[word]

