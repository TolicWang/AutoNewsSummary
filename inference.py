from data.utils.data_helper import padding
from data.utils.load_sogou_news import SogouNews
from model.news_summary import Seq2Seq
from utils.evaluation_utils import evaluate, save_result
import os

if __name__ == '__main__':
    is_compute_bleu = True
    len_words = 1
    freq = 20
    rnn_size = 128
    rnn_layer = 2
    share_vocab = True
    src_embed_size = 128
    tgt_embed_size = 128
    use_attention = False

    sogou_news = SogouNews(file_name='news_sohusite_xml.dat', len_words=len_words, freq=freq)
    # sogou_news = SogouNews(file_name='mini_sogou_news.dat', len_words=1, freq=20)
    _, _, _, word_to_idx, idx_to_word = sogou_news.input_data()
    x, titles, cut_titles, contents = sogou_news.load_inference_data(word_to_idx)
    source_input, source_length = padding(x, word_to_idx)

    model = Seq2Seq(rnn_size=rnn_size,
                    rnn_layer=rnn_layer,
                    share_vocab=share_vocab,
                    src_vocab_size=len(word_to_idx),
                    tgt_vocab_size=len(word_to_idx),
                    src_embed_size=src_embed_size,
                    tgt_embed_size=tgt_embed_size,
                    inference=True,
                    use_attention=use_attention,
                    model_path='MODEL',
                    batch_size=len(source_input)
                    )
    results = model.infer(source_input, source_length)[0]
    summary, cut_summary = sogou_news.index_transform_to_data(results, idx_to_word)

    for item in zip(titles, summary):
        print("真实标签：", item[0])
        print("预测标签：", item[1])
        print('================================')

    if is_compute_bleu:
        ref = cut_titles
        out = cut_summary
        save_result(ref, 'infer_reference')
        save_result(out, 'infer_output')

        output = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'temp', 'result', 'infer_reference')
        reference = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'temp', 'result', 'infer_output')
        bpe_bleu_score = evaluate(reference, output, "bleu")
        print("bleu: ", bpe_bleu_score)
