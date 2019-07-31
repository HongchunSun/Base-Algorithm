import tensorflow as tf
import os

# 定义命令行参数：名字，默认值，说明
tf.app.flags.DEFINE_integer('max_step', 100, '模型训练步数')
FLAGS = tf.app.flags.FLAGS


def mylr():
    """
    实现一个线性回归
    :return: None
    """
    # 1.准备数据
    with tf.variable_scope("data"):  # 变量作用域
        x = tf.random_normal([100, 1], mean=1.75, stddev=0.5, name='x_data')
        y_true = tf.matmul(x, [[0.7]]) + 0.8

    # 2.建立回归模型
    weight = tf.Variable(tf.random_normal([1, 1], mean=0.0, stddev=1.0), name='w')
    bias = tf.Variable(0.0, name='b')

    y_predict = tf.matmul(x, weight) + bias

    # 3.建立损失函数，均方误差
    loss = tf.reduce_mean(tf.square(y_true - y_predict))

    # 4.梯度下降优化损失
    train_op = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

    # 收集tensor
    tf.summary.scalar('losses', loss)
    tf.summary.histogram('weights', weight)

    # 合并
    merged = tf.summary.merge_all()

    # 保存模型
    saver = tf.train.Saver()

    # 5.定义一个初始化变量的op
    init_op = tf.global_variables_initializer()

    # 6.开启会话
    with tf.Session() as sess:
        # 初始化变量
        sess.run(init_op)

        # 打印初始化的随机权重和偏置
        print("随机初始化参数权重：%f，偏置为：%f" % (weight.eval(), bias.eval()))

        # 建立事件
        filewriter = tf.summary.FileWriter('./events/', graph=sess.graph)

        # 加载模型，覆盖模型中随机定义的参数，从上次训练参数的结果
        if os.path.exists('./checkpoints'):
            saver.restore(sess, './checkpoints/test')

        # 循环训练，运行优化
        for i in range(FLAGS.max_step):
            sess.run(train_op)

            # 运行合并的tensor
            summary = sess.run(merged)

            filewriter.add_summary(summary, i)

            print("第%d次优化参数权重：%f，偏置为：%f" % (i, weight.eval(), bias.eval()))
        saver.save(sess, './checkpoints/test')

    return None


if __name__ == '__main__':
    mylr()
