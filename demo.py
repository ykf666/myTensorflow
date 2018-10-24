#!/usr/bin/env python
# coding=utf-8


# 在我们的线性模型 y=W*x+b 中，输入xx可以用占位 Tensor 表示，
# 输出yy可以用线性模型的输出表示，我们需要不断的改变WW和bb的值，来找到一个使lossloss最小的值。
# 这里WW和bb可以用变量 Tensor 表示。
# 使用tf.Variable()可以创建一个变量Tensor


import tensorflow as tf

# 创建变量 W 和 b 节点，并设置初始值
W = tf.Variable([.1], dtype=tf.float32)
b = tf.Variable([-.1], dtype=tf.float32)
# 创建 x 节点，用来输入实验中的输入数据
x = tf.placeholder(tf.float32)
# 创建线性模型
linear_model = W * x + b

# 创建 y 节点，用来输入实验中得到的输出数据，用于损失模型计算
y = tf.placeholder(tf.float32)
# 创建损失模型
loss = tf.reduce_sum(tf.square(linear_model - y))

# 创建 Session 用来计算模型
sess = tf.Session()

init_op = tf.global_variables_initializer()
sess.run(init_op)

# print(sess.run(W))
# print(sess.run(linear_model, {x: [1, 2, 3, 6, 8]}))
# print(sess.run(loss, {x: [1, 2, 3, 6, 8], y: [4.8, 8.5, 10.4, 21.0, 25.3]}))

# 创建一个梯度下降优化器，学习率为0.001
optimizer = tf.train.GradientDescentOptimizer(0.001)
train = optimizer.minimize(loss)

# 用两个数组保存训练数据
# x_train = [1, 2, 3, 6, 8]
# y_train = [4.8, 8.5, 10.4, 21.0, 25.3]
x_train = [1, 2, 3, 4, 5, 6, 7, 8]
y_train = [3, 5, 7, 9, 11, 13, 15, 17]

# 训练10000次
for i in range(10000):
    sess.run(train, {x: x_train, y: y_train})

# 打印一下训练后的结果
print('W: %s b: %s loss: %s' % (sess.run(W), sess.run(
    b), sess.run(loss, {x: x_train, y: y_train})))
