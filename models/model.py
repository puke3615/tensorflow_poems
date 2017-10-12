# -*- coding: utf-8 -*-
# file: model.py
# author: JinTian
# time: 07/03/2017 3:07 PM
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
import tensorflow as tf
import numpy as np


def rnn_model(model, input_data, output_data, vocab_size, rnn_size=128, num_layers=2, batch_size=64,
              learning_rate=0.01):
    """
    构造rnn的序列模型
    :param model: model class
    :param input_data: 输入数据占位符
    :param output_data: 输出数据占位符
    :param vocab_size: words的总长度
    :param rnn_size: rnn的units数
    :param num_layers: rnn中cell的层数
    :param batch_size: 每个batch的样本数量
    :param learning_rate: 学习率
    :return: 模型状态集
    """
    # 声明模型状态集, 由于模型需要返回多个相关值, 故以map集合的形式向外部返回
    end_points = {}

    # 选择rnn的具体cell类型, 提供了rnn、gru、lstm三种
    if model == 'rnn':
        cell_fun = tf.contrib.rnn.BasicRNNCell
    elif model == 'gru':
        cell_fun = tf.contrib.rnn.GRUCell
    elif model == 'lstm':
        cell_fun = tf.contrib.rnn.BasicLSTMCell

    # 构造具体的cell
    cell = cell_fun(rnn_size, state_is_tuple=True)
    # 将单层的cell变为更深的cell, 以表征更复杂的关联关系
    cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers, state_is_tuple=True)

    # 初始化cell的状态
    if output_data is not None:
        # 训练时batch容量为batch_size
        initial_state = cell.zero_state(batch_size, tf.float32)
    else:
        # 使用时batch容量为1
        initial_state = cell.zero_state(1, tf.float32)

    # tensorflow对于lookup_embedding的操作只能再cpu上进行
    with tf.device("/cpu:0"):
        # 构造(vocab_size + 1, run_size)的Tensor
        embedding = tf.get_variable('embedding', initializer=tf.random_uniform(
            [vocab_size + 1, rnn_size], -1.0, 1.0))

        # embedding_lookup函数
        # output = embedding_lookup(embedding, ids): 将ids里的element替换为embedding中对应element位的值
        # 即: embedding: [[1, 2], [3, 4], [5, 6]]  ids: [1, 2]  则outputs: [[3, 4], [5, 6]]
        # 类比one_hot, 只是这里是x_hot
        # embedding: (3, 2)  ids: (10, )  outputs: (10, 2)

        # 处理之后的shape为(batch_size, n_steps, rnn_size)
        inputs = tf.nn.embedding_lookup(embedding, input_data)

    # (batch_size, n_steps, rnn_size) => (batch_size, n_steps, rnn_size)
    outputs, last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state)
    # (batch_size, n_steps, rnn_size) => (batch_size x n_steps, rnn_size)
    output = tf.reshape(outputs, [-1, rnn_size])

    # (batch_size x n_steps, rnn_size) => (batch_size x n_steps, vocab_size + 1)
    weights = tf.Variable(tf.truncated_normal([rnn_size, vocab_size + 1]))
    bias = tf.Variable(tf.zeros(shape=[vocab_size + 1]))
    logits = tf.nn.bias_add(tf.matmul(output, weights), bias=bias)
    # [?, vocab_size+1]

    if output_data is not None:
        # output_data must be one-hot encode
        labels = tf.one_hot(tf.reshape(output_data, [-1]), depth=vocab_size + 1)
        # should be [?, vocab_size+1]

        loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        # loss shape should be [?, vocab_size+1]
        total_loss = tf.reduce_mean(loss)
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

        end_points['initial_state'] = initial_state
        end_points['output'] = output
        end_points['train_op'] = train_op
        end_points['total_loss'] = total_loss
        end_points['loss'] = loss
        end_points['last_state'] = last_state
    else:
        prediction = tf.nn.softmax(logits)

        end_points['initial_state'] = initial_state
        end_points['last_state'] = last_state
        end_points['prediction'] = prediction

    return end_points
