from core.solver import CaptioningSolver
from core.model import CaptionGenerator
from core.utils import load_data


def main():
    # load train dataset
    data, val_data = load_data(data_path='./new_data')
    word_to_idx = data['word_to_idx']

    # load val dataset to print out bleu scores every epoch
    # val_data = load_data(data_path='./new_data', split = 'val')

    model = CaptionGenerator(word_to_idx, dim_feature=[7*7, 512], dim_embed=512/2,
                                       dim_hidden=1024/2, n_time_step=26, prev2out=True,
                                                 ctx2out=True, alpha_c=1.0, selector=True, dropout=True)

    solver = CaptioningSolver(model, data, val_data, n_epochs=20, batch_size=128/2, update_rule='adam',
                                          learning_rate=0.001, print_every=300, save_every=1, image_path='./image/',
                                    pretrained_model=None, model_path='model/snocut/', test_model='model/snocut/model-10',
                                     print_bleu=True, log_path='log/snocut/', model_name='snocut')

    solver.train()

if __name__ == "__main__":
    main()
