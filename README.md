*Tensorflow Poems是一款基于RNN（循环神经网络）的 [Github开源项目](https://github.com/jinfagang/tensorflow_poems)，它能通过学习大量古诗文和歌词然后能够自己来模仿创造诗文和歌词。*

[TOC]

## 简介

就项目本身，其意义不是很大，权当娱乐就行。这里的重点是解读如何从最原始的诗句，到数据的读取，再到数据预处理，再到模型的构建，再到最后的训练和使用流程。

## 源数据

通过上面的github地址可以下载到项目源码，其中古诗文的数据集是`dataset/data/poems.txt`，打开可以看到如下的内容（篇幅限制，只显示了前两行诗句）。

```tex
首春:寒随穷律变，春逐鸟声开。初风飘带柳，晚雪间花梅。碧林青旧竹，绿沼翠新苔。芝田初雁去，绮树巧莺来。
初晴落景:晚霞聊自怡，初晴弥可喜。日晃百花色，风动千林翠。池鱼跃不同，园鸟声还异。寄言博通者，知予物外志。
...
```

整体的格式还是蛮清晰的，每行就代表一首诗，每首诗由标题和内容两部分组成，中间以冒号分割。

## 预处理

代码位于`poems.py`文件的`process_peoms`方法

```python
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
```

## 模型

代码位于`model.py`的`rnn_model`方法

```python
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
```

## 训练

获取数据batch的代码位于`poem.py`的`generate_batch`方法

```python
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
```

训练代码位于`tang_poems.py`的`run_training`方法

```python
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
```

## 使用

取词代码位于`tang_poems.py`的`to_word`方法

```python
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
```

作诗代码位于`tang_peoms.py`的`gen_poem`方法

```python
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
        [predict, last_state] = sess.run([end_points['prediction'], end_points['last_state']],feed_dict={input_data: x})
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
```

