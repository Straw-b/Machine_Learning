# 导入模块
import tensorflow as tf
from PIL import Image, ImageFilter
import numpy as np

# 定义预测函数，返回预测的整数
def predictint(imvalue):
    
    # 定义模型(与创建模型文件时相同)
    x = tf.placeholder(tf.float32, [None, 784])  # 读入图片 28*28
    W = tf.Variable(tf.zeros([784, 10]))  # 权重初始化
    b = tf.Variable(tf.zeros([10]))  # 偏置初始化
    # conv1 filter 5*5*6 input=28*28*1 output= 24*24*6
    W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 6], stddev=0.1))
    b_conv1 = tf.Variable(tf.constant(0.1, shape=[6]))
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    h_conv1 = tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='VALID')
    h_conv1 = tf.nn.relu(h_conv1 + b_conv1)
    #  pooling input=24*24*6 output=12*12*6
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    # conv2 filter 5*5*16 input=12*12*6 output=8*8*16
    W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 6, 16], stddev=0.1))
    b_conv2 = tf.Variable(tf.constant(0.1, shape=[16]))
    h_conv2 = tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='VALID')
    h_conv2 = tf.nn.relu(h_conv2 + b_conv2)
    # pooling input=8*8*16 output=4*4*16
    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    # FC1 input=4*4*16=256  output=120
    W_fc1 = tf.Variable(tf.truncated_normal([256, 120], stddev=0.1))
    b_fc1 = tf.Variable(tf.constant(0.1, shape=[120]))
    h_pool2_flat = tf.reshape(h_pool2, [-1, 256])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    # FC2: input = 120, output = 84.
    W_fc2 = tf.Variable(tf.truncated_normal(shape=(120, 84), stddev=0.1))
    b_fc2 = tf.Variable(tf.constant(0.1, shape=[84]))
    fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    # input=84, output =10
    W_fc3 = tf.Variable(tf.truncated_normal([84, 10], stddev=0.1))  # 分类节点10
    b_fc3 = tf.Variable(tf.constant(0.1, shape=[10]))
    y_conv = tf.nn.softmax(tf.matmul(fc2, W_fc3) + b_fc3)


    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, "tmp_mnist/model")
        print ("Model restored.") # 模型恢复
        prediction=tf.argmax(y_conv,1) #返回每一行的最大值的索引
        return prediction.eval(feed_dict={x: imvalue ,keep_prob: 1.0}, session=sess)



# 定义图像准备，返回一个numpy值
def imageprepare(argv):
    im = Image.open(argv).convert('L') # 模式“L”为灰色图像
    img = im.resize((28, 28), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
    # 这个函数img.resize((width, height),Image.ANTIALIAS)改变图片大小和质量，Image.ANTIALIAS：高质量
    # filter()来调用滤波函数对图像进行滤波,ImageFilter.SHARPEN：锐化滤波
    data = img.getdata()
    # Image.getdata(band=None)
    # 函数的作用是把图片的像素信息flatten到一维，搞成一个特征向量的形式，
    # 当然这边并没有进行特征提取，只是直接把所有像素做成了向量。
    data = np.matrix(data,dtype="float")#矩阵处理
    data = (255.0 - data) / 255.0
    new_data = np.reshape(data, (1, 28 * 28))
    return new_data

# 主函数
def main(argv=None):
    path = "test/05.png" # 测试图片
    imvalue = imageprepare(path)
    imvalue = np.array(imvalue) #将数据转化为矩阵
    predint = predictint(imvalue)
    print ("result:",predint[0]) 
    
if __name__ == "__main__":
    main()
