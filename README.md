# AutoGrad
automatic derivation, in python

- grad.py

实现自动求导所需的基本模块，支持矩阵相关运算，神经网络常见函数的自动求导

- grad_test.py

grad.py的测试文件，使用梯度下降计算一个多元高斯分布的均值和方差

- lenet.py

基于grad.py实现一个Lenet神经网络，训练数据为 *mnist* 数据集，从输入层到输出层单元数为 784 -> 300 -> 100 -> 10

100个iteration之后，测试集上的分类准确率 > 99%
