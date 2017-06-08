from core.solver import CaptioningSolver
from core.model import CaptionGenerator
from core.utils import load_test_data
from gen_captions_from_pickle import gen_caption


if __name__ == "__main__":
    cap_len = 17
    # load train dataset
    test_data= load_test_data(data_path='./new_data', cap_length=cap_len)
    word_to_idx = test_data['word_to_idx']

    # load val dataset to print out bleu scores every epoch
    # val_data = load_data(data_path='./new_data', split = 'val')


    for model_idx in range(1,21):
        model_name = 'snocut'+str(model_idx)
        model = CaptionGenerator(word_to_idx, dim_feature=[7*7, 512], dim_embed=512/2,
                                       dim_hidden=1024/2, n_time_step=cap_len-1, prev2out=True,
                                                 ctx2out=True, alpha_c=1.0, selector=True, dropout=True)

        solver = CaptioningSolver(model, None, None, n_epochs=60, batch_size=128/2, update_rule='adam',
                                         learning_rate=0.001, print_every=300, save_every=1, image_path='./image/',
                                    pretrained_model=None, print_bleu=True,
                              model_path='model/snocut/', test_model='model/snocut/model-'+str(model_idx),
                                      log_path='log/snocut/', model_name=model_name)

        solver.test_all(test_data)
        infile = './new_data/test/test_%s.captions.pickle' % model_name
        ofile = './new_data/test/captions_%s.txt' % model_name
        gen_caption(ifname=infile, ofname=ofile)

