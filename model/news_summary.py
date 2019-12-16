import tensorflow as tf
import tensorflow.contrib as contrib
from utils.evaluation_utils import evaluate, save_result
from data.utils.data_helper import padding
from utils.logs import Logger
import logging
import os


def _create_or_load_embed(embed_name, vocab_size, embed_size, dtype):
    """
    初始化（或者载入已有）词向量函数，后面的编码各解码部分用到的词向量均通过这个函数来获得
    :param embed_name:  参数名称
    :param vocab_size:  词向量个数
    :param embed_size:  词向量维度
    :param dtype:
    :return:
    """
    embedding = tf.get_variable(embed_name, shape=[vocab_size, embed_size], dtype=dtype)
    return embedding


def _create_emb_for_encoder_and_decoder(share_vocab,
                                        src_vocab_size,
                                        tgt_vocab_size,
                                        src_embed_size,
                                        tgt_embed_size,
                                        dtype=tf.float32):
    """
    为编码各解码部分分别创建词向量
    但一般只有在类似翻译模型中，这两部分的词向量才会不一样（一个是中文的，一个对应的英文的）
    在本示例新闻摘要中，词向量是一样的，编码和解码部分都共享一个词典，词向量，因为都是中文
    但是，也可以不一样
    :param share_vocab:  是否共享两者的词向量
    :param src_vocab_size:  词向量的大小
    :param tat_vocab_size:
    :param src_embed_size:  词向量的维度
    :param tgt_embed_size:
    :param dtype:
    :return:
    """
    if share_vocab:
        if src_vocab_size != tgt_vocab_size:
            raise ValueError("若要共享词向量，需保持两者词表大小相等 %d vs. %d" % (src_vocab_size, tgt_vocab_size))
        assert src_embed_size == tgt_embed_size
        print("编码和解码部分共享同样的词嵌入矩阵！")
        embedding_encoder = _create_or_load_embed("embedding_share", vocab_size=src_vocab_size,
                                                  embed_size=src_embed_size, dtype=dtype)
        embedding_decoder = embedding_encoder
    else:
        with tf.variable_scope("encoder"):
            embedding_encoder = _create_or_load_embed('embedding_encoder', embed_size=src_embed_size,
                                                      vocab_size=src_vocab_size, dtype=dtype)
        with tf.variable_scope("decoder"):
            embedding_decoder = _create_or_load_embed('embedding_decoder', embed_size=tgt_embed_size,
                                                      vocab_size=tgt_vocab_size, dtype=dtype)
    return embedding_encoder, embedding_decoder


class Seq2Seq():
    def __init__(self,
                 truncate_times=10,
                 rnn_size=32,
                 rnn_layer=2,
                 share_vocab=True,
                 src_vocab_size=5000,
                 tgt_vocab_size=5000,
                 src_embed_size=128,
                 tgt_embed_size=128,
                 use_attention=False,
                 inference=False,
                 learning_rate=1e-3,
                 batch_size=64,
                 epochs=100,
                 model_path='MODEL',
                 log_dir='log_train'):
        """
        :param rnn_size: # rnn 维度，即 num_units
        :param rnn_layer: # rnn 网络的层数
        :param share_vocab:
        :param src_vocab_size:
        :param tgt_vocab_size:
        :param src_embed_size:
        :param tgt_embed_size:
        :param use_attention:
        :param inference:
        :param learning_rate:
        :param batch_size:
        :param epochs:
        """
        self.truncate_times = truncate_times
        self.rnn_size = rnn_size
        self.rnn_layer = rnn_layer
        self.share_vocab = share_vocab
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.src_embed_size = src_embed_size
        self.tgt_embed_size = tgt_embed_size
        self.use_attention = use_attention
        self.inference = inference
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.model_path = model_path
        self.logger = Logger(log_file_name=log_dir, log_level=logging.DEBUG, logger_name='train').get_log()

        self.logger.info('### 建立网络模型……')
        self._build_placeholder()
        self._build_embedding()
        self._build_encoder()
        self._build_decoder()
        self._build_loss()

    def _build_placeholder(self):
        self.source_input = tf.placeholder(dtype=tf.int32, shape=[None, None],
                                           name='src_in')  # shape: [batch_size,time_step]
        self.target_input = tf.placeholder(dtype=tf.int32, shape=[None, None], name='tgt_in')
        self.target_output = tf.placeholder(dtype=tf.int32, shape=[None, None], name='tgt_out')
        self.encoder_input = tf.transpose(self.source_input)  # time_major = True ==> [time_step,batch_size]
        self.decoder_input = tf.transpose(self.target_input)
        self.decoder_output = tf.transpose(self.target_output)

        self.source_lengths = tf.placeholder(dtype=tf.int32, shape=[None],
                                             name='source_seq_lengths')  # 一个batch中，每个样本（序列）的长度
        self.target_lengths = tf.placeholder(dtype=tf.int32, shape=[None], name='target_seq_lengths')
        self.max_source_length = tf.reduce_max(self.source_lengths)
        self.max_target_length = tf.reduce_max(self.target_lengths)

    def _build_embedding(self):
        self.embedding_encoder, self.embedding_decoder = _create_emb_for_encoder_and_decoder(
            share_vocab=self.share_vocab,
            src_vocab_size=self.src_vocab_size,
            tgt_vocab_size=self.tgt_vocab_size,
            src_embed_size=self.src_embed_size,
            tgt_embed_size=self.tgt_embed_size)
        # self.embedding_encoder: [vocab_size, embed_size]
        # self.embedding_decoder
        print(self.embedding_encoder)
        print(self.embedding_decoder)

    def _build_encoder(self):
        def get_encoder_cell(rnn_size):
            lstm_cell = tf.nn.rnn_cell.LSTMCell(rnn_size)
            return lstm_cell

        encoder_emb_inp = tf.nn.embedding_lookup(self.embedding_decoder,
                                                 self.encoder_input)  # [time_step,batch_size,embed_size]
        encoder_cell = tf.nn.rnn_cell.MultiRNNCell(
            [get_encoder_cell(self.rnn_size) for _ in range(self.rnn_layer)])
        self.encoder_outputs, self.encoder_final_state = tf.nn.dynamic_rnn(cell=encoder_cell,
                                                                           inputs=encoder_emb_inp,
                                                                           sequence_length=self.source_lengths,
                                                                           time_major=True,
                                                                           dtype=tf.float32)
        # encoder_outputs: # [time_step,batch_size,rnn_size]
        # encoder_final_state: [batch_size,rnn_size]

    def _build_decoder(self):
        def get_decoder_cell(rnn_size):
            decoder_cell = tf.nn.rnn_cell.LSTMCell(rnn_size)
            return decoder_cell

        decoder_cell = tf.nn.rnn_cell.MultiRNNCell(
            [get_decoder_cell(self.rnn_size) for _ in range(self.rnn_layer)])
        decoder_init_state = self.encoder_final_state  # thought vector

        if self.use_attention:
            # encoder_outputs: # [time_step,batch_size,rnn_size]
            attention_states = tf.transpose(self.encoder_outputs, [1, 0, 2])  # 如果之前没有设置time_major，则不用转换
            # attention_states 指的就是编码阶段每个时刻的编码状态
            attention_mechanism = contrib.seq2seq.LuongAttention(num_units=self.rnn_size,
                                                                 memory=attention_states,
                                                                 memory_sequence_length=self.source_lengths)
            decoder_cell = contrib.seq2seq.AttentionWrapper(decoder_cell, attention_states,
                                                            attention_layer_size=self.rnn_size)
            decoder_init_state = decoder_cell.zero_state(self.batch_size, dtype=tf.float32).clone(
                cell_state=self.encoder_final_state)

        if self.inference:  # 预测
            helper = contrib.seq2seq.GreedyEmbeddingHelper(self.embedding_decoder,
                                                           tf.fill(dims=[self.batch_size], value=1), 2)  # 1指 SOS, 2指EOS
            maximum_iterations = tf.round(self.max_source_length * 2)
        else:  # 训练
            self.decoder_emb_inp = tf.nn.embedding_lookup(self.embedding_decoder, self.decoder_input)
            helper = contrib.seq2seq.TrainingHelper(inputs=self.decoder_emb_inp,
                                                    sequence_length=self.target_lengths,
                                                    time_major=True)
            maximum_iterations = self.max_target_length
        projection_layer = tf.layers.Dense(self.tgt_vocab_size, use_bias=False)
        decoder = contrib.seq2seq.BasicDecoder(cell=decoder_cell,
                                               helper=helper,
                                               initial_state=decoder_init_state,
                                               output_layer=projection_layer)
        outputs, _, _ = contrib.seq2seq.dynamic_decode(decoder=decoder,
                                                       output_time_major=True,
                                                       maximum_iterations=maximum_iterations)
        self.logits = outputs.rnn_output
        self.pred = tf.transpose(outputs.sample_id)

    def _build_loss(self):
        masks = tf.sequence_mask(self.target_lengths, self.max_target_length, dtype=tf.float32,
                                 name='masks')  # [batch_size, max_length/time_step]
        target_weights = tf.transpose(masks)
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.decoder_output, logits=self.logits)
        self.train_loss = (tf.reduce_sum(crossent * target_weights) / self.batch_size)

    def train(self, source_input, target_input, target_output, src_vocab_table, tgt_vocab_table, gen_batch,
              random_samples, index_transform_to_data, idx_to_word):

        self.logger.info("### 开始训练网络……")
        self.model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'temp', self.model_path)
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        params = tf.trainable_variables()

        gradients = tf.gradients(self.train_loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, 2)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        update_step = optimizer.apply_gradients(zip(clipped_gradients, params))
        saver = tf.train.Saver(params, max_to_keep=5)
        model_name = 'train'

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            start_epoch = 0
            check_point = tf.train.latest_checkpoint(self.model_path)
            if check_point:
                saver.restore(sess, check_point)
                start_epoch += int(check_point.split('-')[-1])
                if start_epoch % 2 != 0:
                    start_epoch += 1
                self.logger.info("### 成功载入已有模型：<{}>……".format(check_point))
            total_loss = 0
            try:
                for epoch in range(start_epoch, self.epochs):
                    n_chunk = len(source_input) // self.batch_size
                    ave_loss = total_loss / n_chunk
                    total_loss = 0
                    batches = gen_batch(source_input, target_input, target_output, src_vocab_table, tgt_vocab_table,
                                        self.truncate_times, self.batch_size)
                    for step, batch in enumerate(batches):
                        if max(max(batch[0])) >= len(src_vocab_table):
                            self.logger.info("soruces:{}\ntarget:{}".format(batch[0], batch[1]))
                        feed_dict = {self.source_input: batch[0],
                                     self.target_input: batch[1],
                                     self.target_output: batch[2],
                                     self.source_lengths: batch[3],
                                     self.target_lengths: batch[4]}
                        loss, _ = sess.run([self.train_loss, update_step], feed_dict=feed_dict)
                        total_loss += loss
                        if step % 50 == 0:
                            self.logger.info(
                                "# Epoch:[{}/{}]-batch:[{}/{}]-Loss:{:.2f}-last epoch ave loss:{:.2f} max src len:{}".format(
                                    epoch, self.epochs, step, n_chunk,
                                    loss, ave_loss, max(batch[3])))
                        if step % 500 == 0:
                            nums = 10
                            feed_dict = {self.source_input: batch[0][:nums],
                                         self.target_input: batch[1][:nums],
                                         self.target_output: batch[2][:nums],
                                         self.source_lengths: batch[3][:nums],
                                         self.target_lengths: batch[4][:nums]}
                            pred = sess.run([self.pred], feed_dict=feed_dict)
                            self.evaluate_bleu(pred[0], idx_to_word, batch[2])

                    if epoch % 3 == 0:
                        self.logger.info("### 模型保持成功 {}...".format(model_name + '-' + str(epoch)))
                        saver.save(sess, os.path.join(self.model_path, model_name),
                                   global_step=epoch, write_meta_graph=False)
            except KeyboardInterrupt:
                self.logger.warning("手动终止训练，保存中……")
                saver.save(sess, os.path.join(self.model_path, model_name), global_step=epoch - 1,
                           write_meta_graph=False)

    def evaluate_bleu(self, outputs,idx_to_word,targets):

        ref, out = [], []
        samples = zip(outputs,targets)
        for item in samples:
            t = [idx_to_word[idx] for idx in item[0]]
            out.append(" ".join(t))
            t = [idx_to_word[idx] for idx in item[1]]
            ref.append(" ".join(t))
        save_result(ref,'reference')
        save_result(out,'output')

        output = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'temp', 'result', 'output')
        reference = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'temp', 'result', 'reference')
        bpe_bleu_score = evaluate(reference, output, "bleu")
        for item in zip(ref, out):
            self.logger.info("\n标签值：{}\n预测值：{}".format(item[0], item[1]))
        self.logger.info("bleu:{}".format(bpe_bleu_score))
if __name__ == '__main__':
    model = Seq2Seq()
