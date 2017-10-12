# -*- coding: utf-8 -*-
# file: tang_poems.py
# author: JinTian
# time: 08/03/2017 7:45 PM
# Copyright 2017 JinTian. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
import collections
import os
import sys
import numpy as np
import tensorflow as tf
from models.model import rnn_model
from dataset.poems import process_poems, generate_batch
import heapq

tf.app.flags.DEFINE_integer('batch_size', 64, 'batch size.')
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'learning rate.')

# set this to 'main.py' relative path
tf.app.flags.DEFINE_string('checkpoints_dir', os.path.abspath('./checkpoints/poems/'), 'checkpoints save path.')
tf.app.flags.DEFINE_string('file_path', os.path.abspath('./dataset/data/poems.txt'), 'file name of poems.')

tf.app.flags.DEFINE_string('model_prefix', 'poems', 'model save prefix.')

tf.app.flags.DEFINE_integer('epochs', 50, 'train how many epochs.')

FLAGS = tf.app.flags.FLAGS

start_token = 'G'
end_token = 'E'


def run_training():
    # 检测模型参数文件夹及父文件夹, 不存在则新建
    if not os.path.exists(os.path.dirname(FLAGS.checkpoints_dir)):
        os.mkdir(os.path.dirname(FLAGS.checkpoints_dir))
    if not os.path.exists(FLAGS.checkpoints_dir):
        os.mkdir(FLAGS.checkpoints_dir)

    # 读取诗集文件
    # 依次得到数字ID表示的诗句、汉字-ID的映射map、所有的汉字的列表
    poems_vector, word_to_int, vocabularies = process_poems(FLAGS.file_path)
    # 按照batch读取输入和输出数据
    batches_inputs, batches_outputs = generate_batch(FLAGS.batch_size,
                                                     poems_vector, word_to_int)

    # 声明输入、输出的占位符
    input_data = tf.placeholder(tf.int32, [FLAGS.batch_size, None])
    output_targets = tf.placeholder(tf.int32, [FLAGS.batch_size, None])

    # 通过rnn模型得到结果状态集
    end_points = rnn_model(model='lstm', input_data=input_data,
                           output_data=output_targets, vocab_size=len(vocabularies),
                           rnn_size=128, num_layers=2, batch_size=64,
                           learning_rate=FLAGS.learning_rate)

    # 初始化saver和session
    saver = tf.train.Saver(tf.global_variables())
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_op)
        start_epoch = 0
        # 加载上次的模型参数(如果有)
        checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoints_dir)
        if checkpoint:
            saver.restore(sess, checkpoint)
            print("[INFO] restore from the checkpoint {0}".format(checkpoint))
            # 如果有模型参数, 则取出对应的epoch, 训练从该epoch开始训练
            start_epoch += int(checkpoint.split('-')[-1])
        # 开始训练
        print('[INFO] start training...')
        try:
            for epoch in range(start_epoch, FLAGS.epochs):
                n = 0
                # 计算一个epoch需要多少次batch训练完, 有余数则忽略掉末尾部分
                n_chunk = len(poems_vector) // FLAGS.batch_size
                for batch in range(n_chunk):
                    # 训练并计算loss
                    # batches_inputs[n]: 第n个batch的输入数据
                    # batches_outputs[n]: 第n个batch的输出数据
                    loss, _, _ = sess.run([
                        end_points['total_loss'],
                        end_points['last_state'],
                        end_points['train_op']
                    ], feed_dict={
                        input_data: batches_inputs[n],
                        output_targets: batches_outputs[n]
                    })
                    n += 1
                    print('[INFO] Epoch: %d , batch: %d , training loss: %.6f' % (epoch, batch, loss))
                # 每训练6个epoch进行一次模型保存
                if epoch % 6 == 0:
                    saver.save(sess, os.path.join(FLAGS.checkpoints_dir,
                                                  FLAGS.model_prefix), global_step=epoch)
        except KeyboardInterrupt:
            # 用户手动退出时, 尝试保存模型参数
            print('[INFO] Interrupt manually, try saving checkpoint for now...')
            saver.save(sess, os.path.join(FLAGS.checkpoints_dir,
                                          FLAGS.model_prefix), global_step=epoch)
            print('[INFO] Last epoch were saved, next time will start from epoch {}.'.format(epoch))


def to_word(predict, vocabs):
    # 取词逻辑
    # 将predict累加求和
    t = np.cumsum(predict)
    # 求出预测可能性的总和
    s = np.sum(predict)
    # 返回将0~s的随机值插值到t中的索引值
    # 由于predict各维度对应的词向量是按照训练数据集的频率进行排序的
    # 故P(x|predict[i]均等时) > P(x + δ), 即达到了权衡优先取前者和高概率词向量的目的
    sample = int(np.searchsorted(t, np.random.rand(1) * s))
    if sample >= len(vocabs):
        sample = len(vocabs) - 1
    return vocabs[sample]


def gen_poem(begin_word):
    # 根据首个汉字作诗
    # 作诗时, batch_size设为1
    batch_size = 1
    print('[INFO] loading corpus from %s' % FLAGS.file_path)
    # 读取诗集文件
    # 依次得到数字ID表示的诗句、汉字-ID的映射map、所有的汉字的列表
    poems_vector, word_int_map, vocabularies = process_poems(FLAGS.file_path)

    # 声明输入的占位符
    input_data = tf.placeholder(tf.int32, [batch_size, None])

    # 通过rnn模型得到结果状态集
    end_points = rnn_model(model='lstm', input_data=input_data,
                           output_data=None, vocab_size=len(vocabularies),
                           rnn_size=128, num_layers=2, batch_size=64,
                           learning_rate=FLAGS.learning_rate)

    # 初始化saver和session
    saver = tf.train.Saver(tf.global_variables())
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_op)

        # 加载上次的模型参数
        checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoints_dir)
        # 注: 无模型参数时, 该步直接crash, 强制有训练好的模型参数
        saver.restore(sess, checkpoint)

        # 取出诗文前缀(G)对应的索引值所谓初始输入
        x = np.array([list(map(word_int_map.get, start_token))])

        # 得出预测值和rnn的当前状态
        [predict, last_state] = sess.run([end_points['prediction'], end_points['last_state']],
                                         feed_dict={input_data: x})
        if begin_word:
            # 用户输入值赋值给word
            word = begin_word
        else:
            # 若未输入, 则取初始预测值的词向量
            word = to_word(predict, vocabularies)
        # 初始化作诗结果变量
        poem = ''
        # 未到结束符时, 一直预测下一个词
        while word != end_token:
            # 没预测一个则追加到结果上
            poem += word
            # 初始化输入为[[0]]
            x = np.zeros((1, 1))
            # 赋值为当前word对应的索引值
            x[0, 0] = word_int_map[word]
            # 根据当前词和当前的上下文状态(last_state)进行预测
            # 返回的结果是预测值和最新的上下文状态
            [predict, last_state] = sess.run([end_points['prediction'],
                                              end_points['last_state']],
                                             feed_dict={
                                                 input_data: x,
                                                 end_points['initial_state']: last_state
                                             })
            # 根据预测值得出词向量
            word = to_word(predict, vocabularies)
        return poem


def pretty_print_poem(poem):
    poem_sentences = poem.split('。')
    for s in poem_sentences:
        if s != '' and len(s) > 10:
            print(s + '。')


def main(is_train):
    if is_train:
        print('[INFO] train tang poem...')
        run_training()
    else:
        print('[INFO] write tang poem...')

        begin_word = raw_input('全新藏头诗上线，输入起始字:')
        poem2 = gen_poem(begin_word)
        pretty_print_poem(poem2)


if __name__ == '__main__':
    tf.app.run()
