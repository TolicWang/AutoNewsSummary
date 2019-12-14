from data.utils.load_sogou_news import PAD
from data.utils.load_sogou_news import SogouNews
# from load_sogou_news import PAD
# from load_sogou_news import SogouNews


def truncate_sentences(src_in, tgt_in, truncate_times=10):
    """

    :param src_in:
    :param tgt_in:
    :param truncate_times:
    在文本摘要中，由于输入序列太长，和标签序列长度相差很大；
    并且在这个任务中，也不一定需要这么长的输入（一段新闻的标题，往往可以根据前面的部分内容得出而不是全部）
    因此为了更加有效的训练模型，可以只选择部分（例如标签序列长度的truncate_times倍）
    :return:
    """
    sources = []
    for item in zip(src_in, tgt_in):
        truncate_len = len(item[1]) * truncate_times
        if len(item[0]) <= truncate_len:
            sources.append(item[0])
        else:
            tmp = item[0][:truncate_len] + [item[0][-1]]
            sources.append(tmp)
    return sources


def padding(sentences, vocab_table):
    """
    对每个样本进行补齐
    :param sentences:
    :param vocab_table:
    :return:
    """
    new_sens = []
    lengths = [len(item) for item in sentences]
    max_len = max(lengths)
    for sentence in sentences:
        s_len = len(sentence)
        if s_len < max_len:
            sentence += [vocab_table[PAD]] * (max_len - s_len)
        new_sens.append(sentence)
    return new_sens, lengths


def gen_batch(src_in, tgt_in, tgt_out, src_vocab_table, tgt_vocab_table, truncate_times=0, batch_size=64):
    s_index, e_index, batches = 0, 0 + batch_size, len(src_in) // batch_size
    if truncate_times > 0:
        print("截断处理……")
        src_in = truncate_sentences(src_in, tgt_in, truncate_times)
    for i in range(batches):
        if e_index > len(src_in):
            e_index = len(src_in)
        batch_src_in = src_in[s_index: e_index]
        batch_tgt_in = tgt_in[s_index: e_index]
        batch_tgt_out = tgt_out[s_index: e_index]
        batch_src_in, source_lengths = padding(batch_src_in, src_vocab_table)
        batch_tgt_in, target_lengths = padding(batch_tgt_in, tgt_vocab_table)
        batch_tgt_out, _ = padding(batch_tgt_out, tgt_vocab_table)
        s_index, e_index = e_index, e_index + batch_size
        yield batch_src_in, batch_tgt_in, batch_tgt_out, source_lengths, target_lengths


if __name__ == '__main__':
    src_in = [[1, 3, 4, 5, 3, 5, 5, 0], [5, 6, 4, 5, 7, 9, 0, 0, 0, 6, 5, 0]]
    tgt_in = [[1, 3, 4], [0, 0, 6, 5]]
    sources = truncate_sentences(src_in, tgt_in, 2)
    print(sources)
    print("{:.3f}".format(1.43223))
    sogou_news = SogouNews(len_words=2, freq=20, top_words=20000, save_name='full_preprocess.pkl')
    source_input, target_input, target_output, word_to_idx, idx_to_word = sogou_news.input_data()
    batches = gen_batch(source_input, target_input, target_output, word_to_idx, word_to_idx,truncate_times=10,batch_size=16)
    n_chunk = len(source_input)//16
    for i, batch in enumerate(batches):
        print("\n-----batch [{}/{}]: ".format(n_chunk,i))
        a = max(max(batch[0]))
        b = max(max(batch[1]))
        c= max(max(batch[2]))
        if a >= len(word_to_idx) or b >= len(word_to_idx) or c >= len(word_to_idx):
            print('here')
            break
        print(len(word_to_idx))
