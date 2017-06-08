from core.utils import load_pickle

def gen_caption(ifname, ofname):
    f = load_pickle(ifname)
    with open(ofname, 'w') as of:
        for i,caption in enumerate(f):
            line = str(9000+i) + ' ' + caption + '\n'
            line = line.encode('utf8')
            of.write(line)


if __name__ == '__main__':
    ofname = './new_data/test/model_small.captions.txt'
    ifname = './new_data/test/test_small.candidate.captions.pkl'
    gen_caption(ifname, ofname)
