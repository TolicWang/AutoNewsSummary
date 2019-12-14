
from load_sogou_news import SogouNews

if __name__ == '__main__':
    sogou_news = SogouNews()
    source_input, target_input, target_output, word_to_idx, idx_to_word = sogou_news.input_data()

    num_samples = len(source_input)
    num_words = len(word_to_idx)-1
    length_samples = [len(item) for item in source_input]
    length_targets = [len(item) for item in target_output]

    print("样本总数为：", num_samples)
    print("字典大小为：", num_words)
    print("样本长度情况分布：", length_samples[:20],"……")
    print("标签长度情况分布：", length_targets[:20],"……")
    print("最长样本长度为：{},其对应标签长度为：{}".
          format(max(length_samples), length_targets[length_samples.index(max(length_samples))]))
