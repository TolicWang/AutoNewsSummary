import jieba
from collections import Counter
import os
import pickle
from tqdm import tqdm

UNK = '<UNK>'
SOS = '<SOS>'
EOS = '<EOS>'
PAD = '<PAD>'
special_tokens = [UNK, SOS, EOS, PAD]


def strQ2B(ustring):
    """
    中文全角转半角
    :param ustring:
    :return:
    Example:
    # print(strQ2B('Ａｎｇｅｌａｂａｂｙ'))
    >> Angelababy
    """
    ss = ""
    for s in ustring:
        rstring = ""
        for uchar in s:
            inside_code = ord(uchar)
            if inside_code == 12288:  # 全角空格直接转换
                inside_code = 32
            elif (inside_code >= 65281 and inside_code <= 65374):  # 全角字符（除空格）根据关系转化
                inside_code -= 65248
            rstring += chr(inside_code)
        ss += rstring
    return ss


class SogouNews:
    def __init__(self, file_name='mini_sogou_news.dat', len_words=1, freq=10, top_words=20000):
        self.len_words = len_words
        self.top_words = top_words  # 如果freq == 0, 则以top_wrods为准
        self.freq = freq  # 过滤掉出现频率小于10的词
        # 如果freq 和top_words同时指定的话，以freq优先
        self.save_name = file_name.split('.')[0] + '_' + str(len_words) + '_' + str(freq) + '_preprocess.pkl'
        self.data_set_desc = ""
        self.file_name = file_name

    def load_raw_data(self, ):
        # def load_raw_data(self, file_name='mini_sogou_news.dat'):
        """
        清洗数据
        :param file_path:
        :return:
        titles[1]: 中国西部是地球上主要干旱带之一,妇女是当地劳动力...
        contents[1]: 同心县地处宁夏中部干旱带的核心区, 冬寒长,春暖迟,夏热短,秋凉早,干旱少雨,蒸发强烈,风大沙多。.....
        """
        file_path = os.path.join(os.path.abspath(__file__)[:-25], 'SogouNews', self.file_name)
        print("载入原始数据……", file_path)
        f = open(file_path, 'r', encoding='gb18030')
        raw_data = f.read().split('</doc>')
        f.close()
        titles, contents = [], []
        for line in tqdm(raw_data[:-1]):
            line = line.strip('\n')
            line = line.split('\n')
            title = strQ2B(line[3][14:-15])
            content = strQ2B(line[4][9:-10]).replace('\ue40c', ' ')
            if len(title) < 2 or len(content) < 50:
                continue
            titles.append(title)
            contents.append(content)
        self.data_set_desc += "1.样本总数为:{}\n".format(len(contents))

        # for item in zip(titles, contents):
        #     print(item[0])
        #     print(item[1])
        #     print('--------')
        return titles, contents

    def load_inference_data(self, word_to_idx):
        file_path = os.path.join(os.path.abspath(__file__)[:-25], 'SogouNews', 'inference_data.txt')
        print("载入预测数据成功！", file_path, '\n')
        f = open(file_path, 'r', encoding='utf-8')
        raw_data = f.readlines()
        f.close()
        titles, contents = [], []
        line_num = 0
        while line_num < len(raw_data):
            title = raw_data[line_num].strip('\n')[14:-15]
            titles.append(strQ2B(title))
            content = raw_data[line_num + 1].strip('\n')[9:-10]
            contents.append(strQ2B(content).replace('\ue40c', ' '))
            line_num += 2
        cut_contents = []

        for item in contents:
            seg = jieba.cut(item, cut_all=False)
            seg = " ".join(seg)
            cut_contents.append(seg.split())
        x = self.data_transform_to_index(word_to_idx, cut_contents)

        return x,titles,contents

    def get_vocab_and_dict(self, ):
        """
        分词，建立字典
        :return:
        word_to_idx: {'UNK': 0, 'SOS': 1, 'EOS': 2, 'PAD': 3, ',': 4, '的': 5, '。': 6, '、': 7}
        idx_to_word: {0: 'UNK', 1: 'SOS', 2: 'EOS', 3: 'PAD', 4: ',', 5: '的', 6: '。', 7: '、'}
        x: [['南', '都', '讯', '记者', '刘凡', '周昌', '和', '任笑', '一',...],[...],]
        y: [['深圳', '地铁', '将', '设立', 'VIP', '头等', '车厢', '买', ...], ['中国', '西部'...]]
        """
        print("\n ### 构造字典……")
        titles, contents = self.load_raw_data()
        cut_words = ""
        x, y = [], []
        for item in tqdm(zip(titles, contents)):
            seg_title = jieba.cut(item[0], cut_all=False)
            seg_content = jieba.cut(item[1], cut_all=False)
            str_title = (" ".join(seg_title))
            str_content = (" ".join(seg_content))
            cut_words += (str_title + str_content)
            y.append(str_title.split())
            x.append(str_content.split())

        cut_words = cut_words.split()
        self.data_set_desc += "2.整个数据一共包含{}个词语\n".format(len(cut_words))

        print("\n ### 分词完成，整个数据一共包含{}个词语！".format(len(cut_words)))
        print("\n ### 词频统计中……")
        c = Counter()
        for word in tqdm(cut_words):
            if len(word) >= self.len_words \
                    or (word == ',' or word == '。' or word == '、' or word == '“' or
                        word == '”' or word == '《' or word == '》' or word == '！') or word.isdigit():
                c[word] += 1
        word_to_idx = {UNK: 0, SOS: 1, EOS: 2, PAD: 3}
        idx_to_word = {0: UNK, 1: SOS, 2: EOS, 3: PAD}
        dicts = sorted(c.items(), key=lambda x: x[1], reverse=True)

        self.data_set_desc += "3.整个数据一共包含{}个不同的词语\n".format(len(dicts))
        if self.freq > 0:
            lens = len(c.items())
            for i in range(len(c.items()) - 1, -1, -1):
                if dicts[i][1] < self.freq:
                    lens -= 1
                    continue
                else:
                    break
        else:
            lens = self.top_words if self.top_words <= len(c.items()) else len(c.items())

        dicts = dicts[:lens]
        print(dicts)
        print("\n ### 按指定条件筛选后，整个数据还包含{}个不同的词语！\n".format(lens))
        self.data_set_desc += "4.按指定条件筛选后，整个数据还包含{}个不同的词语！\n".format(lens)
        for i, idx in enumerate(range(len(word_to_idx), lens + len(word_to_idx))):
            word_to_idx[dicts[i][0]] = idx
            idx_to_word[idx] = dicts[i][0]

        return x, y, word_to_idx, idx_to_word

    def data_transform_to_index(self, word_to_idx, sentence, marks=None):
        """

        :param word_to_idx:
        :param sentence: [['南', '都', '讯', '记者', '刘凡', '周昌', '和', '任笑', '一',...],[...],]
        :param marks:
        :return: [[0, 26, 0, 44, 0, 0, 12, 0, 157, 0],[ 0, 0, 63, 4, 0, 0, 18, 0, 97, 0, 0, 0, 4]]
        """
        res = []
        for s in tqdm(sentence):
            tmp = []
            for word in s:
                if word not in word_to_idx:
                    tmp.append(word_to_idx[UNK])
                else:
                    tmp.append(word_to_idx[word])

            if marks is None:
                res.append(tmp)
            elif marks == SOS:
                res.append([word_to_idx[marks]] + tmp)
            elif marks == EOS:
                res.append(tmp + [word_to_idx[marks]])
            else:
                raise ValueError("Padding Error")
        return res

    def index_transform_to_data(self, sentences, idx_to_word):
        results = []
        for s in sentences:
            tmp = []
            for word in s:
                word = idx_to_word[word]
                if word == EOS:
                    break
                tmp.append(word)
            results.append("".join(tmp))
        return results

    def random_sample(self, sources, targets, num_samples, idx_to_word):
        import random
        tmp = list(zip(sources, targets))
        random.shuffle(tmp)
        samples = tmp[:num_samples]
        src, tgt = [], []
        for item in samples:
            t = [idx_to_word[idx] for idx in item[0]]
            src.append("".join(t))
            t = [idx_to_word[idx] for idx in item[1]]
            tgt.append("".join(t))
        return src, tgt

    def input_data(self):
        save_path = os.path.join(os.path.abspath(__file__)[:-25], 'SogouNews', self.save_name)

        if os.path.exists(save_path):
            f = open(save_path, 'rb')
            data = pickle.load(f)
            f.close()
            source_input, target_input, target_output, word_to_idx, idx_to_word, desc = data['source_input'], data[
                'target_input'], data['target_output'], data['word_to_idx'], data['idx_to_word'], data['desc']
            print("缓存载入成功！ {}".format(save_path))
            print(desc)
        else:
            x, y, word_to_idx, idx_to_word = self.get_vocab_and_dict()
            source_input = self.data_transform_to_index(word_to_idx, x)
            target_input = self.data_transform_to_index(word_to_idx, y, marks=SOS)
            target_output = self.data_transform_to_index(word_to_idx, y, marks=EOS)
            tmp = {'word_to_idx': word_to_idx,
                   'idx_to_word': idx_to_word,
                   'source_input': source_input,
                   'target_input': target_input,
                   'target_output': target_output,
                   'desc': self.data_set_desc}
            print(self.data_set_desc)
            f = open(save_path, 'wb')
            pickle.dump(tmp, f)
            f.close()
        return source_input, target_input, target_output, word_to_idx, idx_to_word


if __name__ == '__main__':
    sogou_news = SogouNews(file_name='mini_sogou_news.dat', len_words=1, freq=20)
    # sogou_news = SogouNews(file_name='news_sohusite_xml.dat', len_words=1, freq=20)
    source_input, target_input, target_output, word_to_idx, idx_to_word = sogou_news.input_data()
    sogou_news.load_inference_data(word_to_idx)
