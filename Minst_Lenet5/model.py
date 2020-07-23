# 由于minst的图像输入是28*28，而LeNet要求的输入大小为32*32
# 所以经常用一个类似LeNet-5模型的卷积神经网络来解决MNIST数字识别问题

# 导入模块
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 读入数据
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# one-hot ?即用包含0和1的tensor来表示数字标签，数字1所在的索引值（从0开始）即为我们的数字标签，
# 例如我们有0-9的数字标签，则标签5所对应的one-hot形式为[0 , 0 , 0 , 0 , 0 ,1 , 0 , 0 , 0 , 0]，
# 因为1所在位置的索引值为5


# 创建模型，输入节点和输出结点大小, inputNode = 784, outputNode = 10
# 初始化输入x
x = tf.placeholder(tf.float32, [None, 784])  # 读入图片 28*28
# 构建一个描述您要执行的计算的计算图。这个阶段实际上不执行任何计算;它只是建立了计算的符号表示。
# 该阶段通常将定义一个或多个表示计算图输入的“占位符”（placeholder）对象。
# 多次运行计算图。 每次运行图形时（例如，对于一个梯度下降步骤），您将指定要计算的图形的哪些部分，
# 并传递一个“feed_dict”字典，该字典将给出具体值为图中的任何“占位符”。
# tf.placeholder(    此函数可以理解为形参，用于定义过程，在执行的时候再赋具体的值
# 	dtype, 			 数据类型。常用的是tf.float32,tf.float64等数值类型
# 	shape=None, 	 数据形状。默认是None，就是一维值，也可以是多维（比如[2,3], [None, 3]表示列是3，行不定），
# 	                 因为MNIST数据为28*28*1图像，所以为784
# 	name=None        该函数操作的名字
# )
y_ = tf.placeholder(tf.float32, [None, 10])  # 读入标签
# 初始化输出y, 因为MNIST为[0,9]共十个分类

W = tf.Variable(tf.zeros([784, 10]))  # 权重初始化
# tf.Variable(initializer， name) 功能：保存和更新神经网络中的参数
# 参数：(1)initializer:初始化参数(2)name:变量名
# tf.zeros(shape,dtype=tf.float32,name=None)
b = tf.Variable(tf.zeros([10]))  # 偏置初始化
y = tf.nn.softmax(tf.matmul(x, W) + b)  # 预测值
# Softmax简单的说就是把一个N*1的向量归一化为（0，1）之间的值，
# 由于其中采用指数运算，使得向量中数值较大的量特征更加明显。
# tf.matmul（）矩阵相乘



# 第一层：卷积层
# conv1 filter 5*5*6 input=28*28*1 output= 24*24*6
# 创建卷积核W_conv1,表示卷积核大小为5*5
W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 6], stddev=0.1))# 过滤器大小为5*5, 当前层深度为1， 过滤器的深度为6
#   tf.truncated_normal( 截断正态分布
# 	shape,               shape为常量(向量空间)的形状，如shape=[2,3,4]为X=2,Y=3,Z=4的矩阵
# 	mean=None,           正态分布均值（默认为0）
# 	stddev=None,         正态分布标准差（默认为1） 在tf.truncated_normal中如果x的取值在区间（μ-2σ，μ+2σ）之外则重新进行选择。
#                      	这样保证了生成的值都在均值附近。
# 	dtype=None,          dtype为输出的数据类型，如tf.float32等
# 	seed=None,           随机数种子，为整型，设置后每次生成的随机数都一样。
# 	name=None            该函数操作的名字
# )
#   tf.Variable(initializer， name) 保存和更新神经网络中的参数
b_conv1 = tf.Variable(tf.constant(0.1, shape=[6])) #将偏置参数初始化为小的正数，以避免死神经元
#    tf.constant(             创建常量
#     value,                 value可为数字和数组，
#     dtype=None,            dtype为常量的数据类型，如tf.float32等，
#     shape=None,            shape为常量(向量空间)的形状，如shape=[2,3,4]为X=2,Y=3,Z=4的矩阵
#     name='Const',          name为该常量的名字，string类型
#     verify_shape=False     verify_shape默认为False，如果修改为True的话表示检查value的形状与shape是否相符，如果不符会报错。
# )
#把输入x(二维张量,shape为[None, 784])变成4维的x_image，x_image的shape是[None,28,28,1]
#-1表示自动推测这个维度的size,即传给None
x_image = tf.reshape(x, [-1, 28, 28, 1])
# tf.reshape(          改变张量形式,将张量 tensor 的形状改为 shape
#     tensor,          tensor类型，即placeholder所处理得到的
#     shape,           输出的张量形式
#     name=None        该函数操作的名字
# )
# Mnist中数据，输入n个样本，每个样本是784个列构成的向量。
# 所以输入的是n*784的矩阵。但是输入到CNN中需要卷积，需要每个样本都是矩阵
# 将n个784个向量，变成n个28*28的
# 数组新的shape属性应该要与原来的配套，如果等于-1的话，那么Numpy会根据剩下的维度计算出数组的另外一个shape属性值
h_conv1 = tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='VALID')# 移动步长为1,  不使用全0填充
#   tf.nn.conv2d (		卷积层
# 	input, 				输入，即卷积的图片，要求为一个张量，shape为 [ batch, in_height, in_weight, in_channel ]，
#            	其中batch为图片的数量，in_height 为图片高度，in_weight 为图片宽度，in_channel 为图片的通道数，灰度图该值为1，彩色图为3。
# 	filter, 			卷积核参数，要求为一个张量，shape为 [ filter_height, filter_weight, in_channel, out_channels ]，
#                    	其中 filter_height 为卷积核高度，filter_weight 为卷积核宽度，in_channel 是图像通道数 ，
#                    	和 input 的 in_channel 要保持一致，out_channel 是卷积核数量。
# 	strides,            卷积步长，是一个长度为4的一维向量，[ 1, strides(height), strides(width), 1]，规定前后必须为1
# 	padding, 			string类型，值为“SAME” 和 “VALID”，表示是否考虑边界填充。
# 	                    "SAME"是考虑边界，不足的时候用0去填充周围，"VALID"则不考虑
# 	use_cudnn_on_gpu, 	bool类型，是否使用cudnn加速，默认为true（gpu真的贵，穷学生买不起）
# 	name=None            该函数操作的名字
# )
# 结果返回一个Tensor，该输出就是我们常说的feature map，shape仍然是[batch, height, width, channels]这种形式。
h_conv1 = tf.nn.relu(h_conv1 + b_conv1) # 激活函数Relu去线性化
# relu激化和池化, tf.nn.relu，使用relu激活函数
# tf.nn.relu()函数的目的是，将输入小于0的值幅值为0，输入大于0的值不变


# 第二层：最大池化层
# #  pooling input=24*24*6 output=12*12*6
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
# 池化层过滤器的大小为2*2, 移动步长为2，使用全0填充
# tf.nn.max_pool(   	池化层
#     input,   			输入，一般池化层接在卷积层后面，所以输入通常是feature map，依然是[batch, in_height, in_width, in_channels]
#                      这样的shape 其中，in_height为卷积map的out_height, in_width卷积后map的out_width, in_channels=filter的out_channel
#     ksize,  			池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]，
#                       因为我们不想在batch和channels上做池化，所以这两个维度设为1
#     strides,  		池化滑动步长，是一个长度为4的一维向量，[ 1, strides, strides, 1]，第一位和最后一位一般是1
#     padding,			string类型，值为“SAME” 和 “VALID”，表示是否考虑边界填充。
#                       "SAME"是考虑边界，不足的时候用0去填充周围，"VALID"则不考虑
#     name=None        	该函数操作的名字
# )
# 返回一个Tensor，类型不变，shape仍然是[batch, height, width, channels]这种形式
# h_pool1的输出即为第一层网络输出，shape为[batch,14,14,32]


# 第三层：卷积层
# conv2 filter 5*5*16 input=12*12*6 output=8*8*16
W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 6, 16], stddev=0.1))# 过滤器大小为5*5, 当前层深度为6，过滤器的深度为16
b_conv2 = tf.Variable(tf.constant(0.1, shape=[16]))
h_conv2 = tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='VALID')# 移动步长为1, 不使用全0填充
h_conv2 = tf.nn.relu(h_conv2 + b_conv2)


# 第四层：最大池化层
# pooling input=8*8*16 output=4*4*16
h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
# 池化层过滤器的大小为2*2, 移动步长为2，使用全0填充


# 第五层：全连接层
# FC1 input=4*4*16=256  output=120
# W的第1维size为4*4*16，4*4是h_pool2输出的size
W_fc1 = tf.Variable(tf.truncated_normal([256, 120], stddev=0.1))# 4*4*16=256把前一层的输出变成特征向量
b_fc1 = tf.Variable(tf.constant(0.1, shape=[120]))
# 将第2层的输出reshape成[batch, 4*4*16]的张量
h_pool2_flat = tf.reshape(h_pool2, [-1, 256])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
# tf.matmul(  矩阵相乘x*y
# 	x,    	  矩阵x
# 	y 		  矩阵y
# )

# 为了减少过拟合，加入Dropout层
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)  # 防止过拟合规定前后必须为1
# tf.nn.dropout(
#     x,					输入,为tensor类型
#     keep_prob,			float类型，每个元素被保留下来的概率，设置神经元被选中的概率,在初始化时keep_prob只是一个占位符
#     noise_shape=None,     一个1维的int32张量，代表了随机产生“保留/丢弃”标志的shape。
#     seed=None,			随机数种子，为整型，设置后每次生成的随机数都一样。
#     name=None     		该函数操作的名字
# )


# 第六层：全连接层.
# FC2: input = 120, output = 84.
W_fc2 = tf.Variable(tf.truncated_normal(shape=(120, 84), stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[84]))
fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


# 第七层：输出层
# input=84, output =10
W_fc3 = tf.Variable(tf.truncated_normal([84, 10], stddev=0.1)) # 分类节点10
b_fc3 = tf.Variable(tf.constant(0.1, shape=[10]))
y_conv = tf.nn.softmax(tf.matmul(fc2, W_fc3) + b_fc3)
# tf.matmul（）将矩阵a乘以矩阵b
# Softmax简单的说就是把一个N*1的向量归一化为（0，1）之间的值，
# 由于其中采用指数运算，使得向量中数值较大的量特征更加明显。


# 定义损失和优化
# 采用交叉熵做目标函数
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_,logits = y_conv))
# tf.nn.softmax_cross_entropy_with_logits(
# 	logits, 		神经网络最后一层的输出，如果有batch的话，它的大小就是[batchsize，num_classes]，
#                	单样本的话，大小就是num_classes，需和标签大小一致
# 	labels, 		实际的标签
# 	name=None   	该函数操作的名字
# )
# tf.reduce_mean(      		计算张量tensor沿着指定的数轴（tensor的某一维度）上的的平均值，
#                           主要用作降维或者计算tensor（图像）的平均值。
# 	input_tensor,			输入张量
#     axis=None,			指定的轴，如果不指定，则计算所有元素的均值;
#     keep_dims=False, 		是否降维度，设置为True，输出的结果保持输入tensor的形状，设置为False，输出结果会降低维度;
#     name=None,			设置函数名称
#     )

# 选择优化器，并让优化器最小化损失函数/收敛, 反向传播
# 引入tf.train.AdamOptimizer().minimize进行梯度下降，学习率为0.0001
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)  # 实现Adam算法的优化器

# 评估函数
# tf.argmax()返回的是某一维度上其数据最大所在的索引值，在这里即代表预测值和真实值
# 判断预测值y_conv和真实值y_中最大数的索引是否一致，y的值为1-10概率
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
# tf.argmax(						返回最大的那个数值所在的下标
#     input,						输入矩阵
#     axis=None,				    axis可被设置为0或1，分别表示0：按列计算，1：行计算
#     name=None,					设置函数名称
#     dimension=None,
#     output_type=tf.int64			返回数据类型
#     )
# tf.equal(				判断输入x,y是否相等
# 	x, 				    输入x
# 	y,    				输入y
# 	name=None  			设置函数名称
# 	)

# 精确度计算,用平均值来统计测试准确率
# 计算正确预测项的比例，因为tf.equal返回的是布尔值，
# 使用tf.cast把布尔值转换成浮点数，然后用tf.reduce_mean求平均值
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# tf.cast(         执行 tensorflow 中张量数据类型转换
#   input,         待转换的数据（张量）
# 	dtype, 		   目标数据类型
# 	name=None      设置函数名称
# 	)
# 释义：数据类型转换x，输入张量dtype，转换数据类型name，名称

# tf.reduce_mean(      	计算张量tensor沿着指定的数轴（tensor的某一维度）上的的平均值，主要用作降维或者计算tensor（图像）的平均值。
# 	input_tensor,		输入张量
#     axis=None,		指定的轴，如果不指定，则计算所有元素的均值;
#     keep_dims=False, 	是否降维度，设置为True，输出的结果保持输入tensor的形状，设置为False，输出结果会降低维度;
#     name=None,		设置函数名称
#     )
# tf.reduce_mean 函数用于计算张量tensor沿着指定的数轴（tensor的某一维度）上的的平均值，
# 主要用作降维或者计算tensor（图像）的平均值。


#将训练好的模型参数保存起来，以便以后进行验证或测试
saver = tf.train.Saver()


#运行
# tf.Session()创建一个会话，当上下文管理器退出时会话关闭和资源释放自动完成。
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) # 添加节点用于初始化全局变量，再进行训练
    # 返回一个初始化所有全局变量的操作。在你构建完整个模型并在会话中加载模型后，运行这个节点。
    for i in range(20000):
        # train.next_batch(50)   每次从训练集中抓取50幅图像
      batch = mnist.train.next_batch(50)
      if i%1000 == 0:
          # 每迭代1000步输出在测试集上的精确度
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0}) # 对偶变量keep-prob代替dropout,评估阶段不使用
          # accuracy.eval()函数的作用:
          # f.Tensor.eval(feed_dict=None, session=None)：
          # 作用：
          # 在一个Seesion里面“评估”tensor的值（其实就是计算），首先执行之前的所有必要的操作来产生这个计算这个tensor需要的输入，
          # 然后通过这些输入产生这个tensor。在激发tensor.eval()这个函数之前，
          # tensor的图必须已经投入到session里面，或者一个默认的session是有效的，或者显式指定session.
          # 参数：
          # feed_dict:一个字典，用来表示tensor被feed的值（联系placeholder一起看）
          # session:（可选） 用来计算（evaluate）这个tensor的session.要是没有指定的话，那么就会使用默认的session。
          # 返回：
          # 表示“计算”结果值的numpy ndarray
        print("step %d, training accuracy %g" %(i, train_accuracy))
      train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5}) # 训练阶段使用50%的Dropout
        #函数feed_dict是用来给之前占位符赋值，即把batch[0]给x等


    save_path = saver.save(sess, "tmp_mnist/model")
    print ("Model saved in file: ", save_path)

    # 在测试数据上测试准确率
    print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

