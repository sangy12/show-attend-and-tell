from core.utils import load_pickle

def gen_caption(ifname, ofname):
    f = load_pickle(ifname)
    with open(ofname, 'w') as of:
        for i,caption in enumerate(f):
            line = str(i) + ' ' + caption + '\n'
            line = line.encode('utf8')
            of.write(line)


if __name__ == '__main__':
    ofname = './new_data/val/model_small_nocut.captions.txt'
    ifname = './new_data/val/val_snocut.candidate.captions.pkl'
    gen_caption(ifname, ofname)
