# -*- coding: utf-8 -*-
# file: poems.py
# author: JinTian
# time: 08/03/2017 7:39 PM
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

start_token = 'G'
end_token = 'E'


def process_poems(file_name):
    # 诗集
    poems = []
    with open(file_name, "r") as f:
        for line in f.readlines():
            try:
                # 取出title和content
                title, content = line.strip().split(':')
                # 移除content中的所有空格
                content = content.replace(' ', '')
                # 过滤掉包含特殊字符的诗
                if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content or \
                                start_token in content or end_token in content:
                    continue
                # 过滤掉过长或过短的诗句
                if len(content) < 5 or len(content) > 79:
                    continue
                # 将内容加上前缀(G)和后缀(E)
                content = start_token + content + end_token
                # 处理后的添加到诗集中
                poems.append(content)
            # 处理过程出错则跳过, 忽略掉
            except ValueError as e:
                pass
    # 按诗的字数排序
    poems = sorted(poems, key=lambda l: len(line))

    # 统计每个字出现次数
    all_words = []
    for poem in poems:
        all_words += [word for word in poem]
    # 计算每个字对应的频率
    counter = collections.Counter(all_words)
    # 按照文字频率进行倒序排列
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    # 取出排列后的字集, 赋值给words
    words, _ = zip(*count_pairs)

    # 将words最后追加一位空格
    words = words[:len(words)] + (' ',)
    # 每个字映射为一个数字ID
    word_int_map = dict(zip(words, range(len(words))))
    # 将诗句中的每个word都注意映射为对应的数字ID
    poems_vector = [list(map(lambda word: word_int_map.get(word, len(words)), poem)) for poem in poems]

    # 依次返回数字ID表示的诗句、汉字-ID的映射map、所有的汉字的列表
    return poems_vector, word_int_map, words


def generate_batch(batch_size, poems_vec, word_to_int):
    # 每次取batch_size首诗进行训练
    n_chunk = len(poems_vec) // batch_size
    x_batches = []
    y_batches = []
    for i in range(n_chunk):
        # 求得每个batch中start和end的索引值
        start_index = i * batch_size
        end_index = start_index + batch_size

        # 取出batch的数据
        batches = poems_vec[start_index:end_index]
        # 找到这个batch的所有poem中最长的poem的长度
        length = max(map(len, batches))
        # 填充一个这么大小的空batch，空的地方放空格对应的index标号
        x_data = np.full((batch_size, length), word_to_int[' '], np.int32)
        for row in range(batch_size):
            # 每一行就是一首诗，在原本的长度上把诗还原上去
            x_data[row, :len(batches[row])] = batches[row]
        y_data = np.copy(x_data)
        # y的话就是x向左边也就是前面移动一个
        y_data[:, :-1] = x_data[:, 1:]
        """
        x_data             y_data
        [6,2,4,6,9]       [2,4,6,9,9]
        [1,4,2,8,5]       [4,2,8,5,5]
        """
        x_batches.append(x_data)
        y_batches.append(y_data)
    return x_batches, y_batches
