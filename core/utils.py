import numpy as np
import cPickle as pickle
import h5py
import time
import os


def load_data(data_path='./new_data', split = 'train', cap_length = 27):
    data_path = os.path.join(data_path, split)

    start_t = time.time()
    data = {}
    val_data = {}

    f = h5py.File(os.path.join('./new_data/train/', 'image_vgg19_block5_pool_feature.h5'))

    data['features'] = np.asarray(f['train_set']).reshape(-1,49,512)
    val_data['features'] = np.asarray(f['validation_set']).reshape(-1,49,512)

    # with open(os.path.join(data_path, '%s.file.names.pkl' %split), 'rb') as f:
        # data['file_names'] = pickle.load(f)
    with open(os.path.join(data_path, '%s_captions_vec.pkl' %split), 'rb') as f:
        captions_vec = pickle.load(f)

    captions = []
    img_idx = []
    count = 0
    for cap_per_img in captions_vec:
        captions += cap_per_img
        img_idx += [count] * len(cap_per_img)
        count += 1

        for words in cap_per_img:
            if len(words)!=cap_length:
                print('\n*** debug ***')
                print(words)
                print('Error')
                print('********\n')
                os.exist(0)

    data['captions'] = np.asarray(captions).astype(np.int32)
    data['image_idxs'] = np.asarray(img_idx)

    if split == 'train':
        with open(os.path.join(data_path, 'word_to_idx.pkl'), 'rb') as f:
            data['word_to_idx'] = pickle.load(f)

    for k, v in data.iteritems():
        if type(v) == np.ndarray:
            print k, type(v), v.shape, v.dtype
        else:
            print k, type(v), len(v)
    end_t = time.time()
    print "Elapse time: %.2f" %(end_t - start_t)
    return data, val_data

def load_test_data(data_path='./new_data', cap_length = 27):
    data_path = os.path.join(data_path, 'train')

    start_t = time.time()
    test_data = {}

    f = h5py.File(os.path.join('./new_data/train/', 'image_vgg19_block5_pool_feature.h5'))

    test_data['features'] = np.asarray(f['test_set']).reshape(-1,49,512)

    with open(os.path.join(data_path, 'word_to_idx.pkl'), 'rb') as f:
        test_data['word_to_idx'] = pickle.load(f)

    for k, v in test_data.iteritems():
        if type(v) == np.ndarray:
            print k, type(v), v.shape, v.dtype
        else:
            print k, type(v), len(v)
    end_t = time.time()
    print "Elapse time: %.2f" %(end_t - start_t)
    return test_data

def decode_captions(captions, idx_to_word):
    if captions.ndim == 1:
        T = captions.shape[0]
        N = 1
    else:
        N, T = captions.shape

    decoded = []
    for i in range(N):
        words = []
        for t in range(T):
            if captions.ndim == 1:
                word = idx_to_word[captions[t]]
            else:
                word = idx_to_word[captions[i, t]]
            if word == u'<END>':
                # words.append('.')
                break
            if word != u'<NULL>' and word != u'<START>':
                word = list(word)
                word = " ".join(word)
                words.append(word)
        decoded.append(' '.join(words))
    return decoded

def sample_coco_minibatch(data, batch_size):
    data_size = data['features'].shape[0]
    mask = np.random.choice(data_size, batch_size)
    features = data['features'][mask]
    file_names = data['file_names'][mask]
    return features, file_names

def write_bleu(scores, path, epoch):
    if epoch == 0:
        file_mode = 'w'
    else:
        file_mode = 'a'
    with open(os.path.join(path, 'val.bleu.scores.txt'), file_mode) as f:
        f.write('Epoch %d\n' %(epoch+1))
        f.write('Bleu_1: %f\n' %scores['Bleu_1'])
        f.write('Bleu_2: %f\n' %scores['Bleu_2'])
        f.write('Bleu_3: %f\n' %scores['Bleu_3'])
        f.write('Bleu_4: %f\n' %scores['Bleu_4'])
        f.write('METEOR: %f\n' %scores['METEOR'])
        f.write('ROUGE_L: %f\n' %scores['ROUGE_L'])
        f.write('CIDEr: %f\n\n' %scores['CIDEr'])

def load_pickle(path):
    with open(path, 'rb') as f:
        file = pickle.load(f)
        print ('Loaded %s..' %path)
        return file

def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        print ('Saved %s..' %path)
