from data.utils.load_sogou_news import PAD
import tensorflow as tf
import logging

from model.news_summary import Seq2Seq
from data.utils.load_sogou_news import SogouNews
from data.utils.data_helper import gen_batch

if __name__ == '__main__':
    sogou_news = SogouNews(len_words=2, freq=20, top_words=20000, save_name='full_preprocess.pkl')
    source_input, target_input, target_output, word_to_idx, idx_to_word = sogou_news.input_data()
    model = Seq2Seq(rnn_size=64,
                    rnn_layer=2,
                    truncate_times=10,
                    share_vocab=True,
                    src_vocab_size=len(word_to_idx),
                    tgt_vocab_size=len(word_to_idx),
                    src_embed_size=128,
                    tgt_embed_size=128,
                    model_path='./MODEL',
                    batch_size=64
                    )
    model.train(source_input, target_input, target_output, word_to_idx, word_to_idx, gen_batch)
