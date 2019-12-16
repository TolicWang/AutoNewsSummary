import tensorflow as tf
from data.utils.data_helper import padding
from data.utils.load_sogou_news import SogouNews
from data.utils.load_sogou_news import special_tokens
from model.news_summary import Seq2Seq
import os

if __name__ == '__main__':
    sogou_news = SogouNews(file_name='news_sohusite_xml.dat', len_words=1, freq=20)
    # sogou_news = SogouNews(file_name='mini_sogou_news.dat', len_words=1, freq=20)
    _, _, _, word_to_idx, idx_to_word = sogou_news.input_data()
    x, titles, contents = sogou_news.load_inference_data(word_to_idx)
    source_input, source_length = padding(x, word_to_idx)
    model = Seq2Seq(rnn_size=128,
                    rnn_layer=2,
                    share_vocab=True,
                    src_vocab_size=len(word_to_idx),
                    tgt_vocab_size=len(word_to_idx),
                    src_embed_size=128,
                    tgt_embed_size=128,
                    inference=True,
                    model_path='MODEL',
                    batch_size=len(source_input)
                    )
    results= model.infer(source_input,source_length)[0]
    summary = sogou_news.index_transform_to_data(results,idx_to_word)


    for item in zip(titles,summary):
        print("真实标签：",item[0])
        print("预测标签：",item[1])
        print('================================')
