from tensor2tensor.data_generators import problem, text_encoder, generator_utils
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry
from tensor2tensor.models import transformer
import os

FILE_PATH = os.path.dirname(os.path.abspath(__file__)) + os.path.sep


@registry.register_problem
class QueryReformulation(text_problems.Text2TextProblem):

    # @property
    # def approx_vocab_size(self):
    #     return 2 ** 15  # 32k

    @property
    def is_generate_per_split(self):
        return False

    @property
    def dataset_splits(self):
        return [{
            "split": problem.DatasetSplit.TRAIN,
            "shards": 8,
        }, {
            "split": problem.DatasetSplit.EVAL,
            "shards": 1,
        }, {
            "split": problem.DatasetSplit.TEST,
            "shards": 1,
        }]

    @property
    def vocab_type(self):
        return text_problems.VocabType.TOKEN

    @property
    def vocab_filename(self):
        return "manual.vocab.bpe.txt"

    @property
    def oov_token(self):
        return 'UNK'

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        del data_dir
        del tmp_dir
        del dataset_split

        train_x_txt = open(os.path.abspath(FILE_PATH + 'bpedata' + os.path.sep + 'train_x.bpe.txt'), 'r')
        train_y_txt = open(os.path.abspath(FILE_PATH + 'bpedata' + os.path.sep + 'train_y.bpe.txt'), 'r')

        # return text_problems.text2text_txt_iterator(bad_txt, good_txt)
        train_x_list = train_x_txt.readlines()
        train_y_list = train_y_txt.readlines()
        train_x_txt.close()
        train_y_txt.close()

        for x, y in zip(train_x_list, train_y_list):
            x = x.strip()
            y = y.strip()
            yield {
                "inputs": x,
                "targets": y
            }


@registry.register_hparams
def transformer_poetry():
    hparams = transformer.transformer_base()
    hparams.num_heads = 4
    hparams.num_hidden_layers = 4
    hparams.hidden_size = 512
    hparams.batch_size = 256
    hparams.attention_dropout = 0.2
    hparams.layer_prepostprocess_dropout = 0.2
    hparams.learning_rate_schedule = ("linear_warmup*constant*cosdecay")
    hparams.learning_rate = 0.0001
    hparams.learning_rate_warmup_steps = 100000
    hparams.learning_rate_constant = 0.0001
    hparams.learning_rate_decay_steps = 350000
    return hparams
