from model.news_summary import Seq2Seq
from data.utils.load_sogou_news import SogouNews
from data.utils.data_helper import gen_batch

if __name__ == '__main__':
    len_words = 1
    freq = 20
    rnn_size = 128
    rnn_layer = 2
    share_vocab = True
    src_embed_size = 128
    tgt_embed_size = 128
    use_attention = True
    truncate_times = 15
    batch_size = 128

    sogou_news = SogouNews(file_name='news_sohusite_xml.dat', len_words=len_words, freq=freq)
    source_input, target_input, target_output, word_to_idx, idx_to_word = sogou_news.input_data()
    model = Seq2Seq(rnn_size=rnn_size,
                    rnn_layer=rnn_layer,
                    truncate_times=truncate_times,
                    share_vocab=share_vocab,
                    src_vocab_size=len(word_to_idx),
                    tgt_vocab_size=len(word_to_idx),
                    src_embed_size=src_embed_size,
                    tgt_embed_size=tgt_embed_size,
                    model_path='MODEL',
                    use_attention=use_attention,
                    batch_size=batch_size
                    )
    model.train(source_input, target_input, target_output, word_to_idx, word_to_idx, gen_batch, idx_to_word)
