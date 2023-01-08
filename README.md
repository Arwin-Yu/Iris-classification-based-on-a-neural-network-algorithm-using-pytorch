# Iris classification based on a neural network algorithm using pytorch

# 一、Introduction
 
This project uses the pytorch deep learning framework to implement a classification task for the Iris dataset based on a neural network algorithm and is perfect for beginners to neural network algorithms.

This readme is introduced in Chinese (including most of the comments in the code). Please translate it into English if necessary.
 

 
# 二、Dataset
Iris数据集是常用的分类实验数据集，由Fisher, 1936收集整理。Iris也称鸢尾花卉数据集，是一类多重变量分析的数据集。数据集包含150个数据样本，分为3类，每类50个数据，每个数据包含4个属性。可通过花萼长度，花萼宽度，花瓣长度，花瓣宽度4个属性预测鸢尾花卉属于（Setosa，Versicolour，Virginica）三个种类中的哪一类。

具体数据详见文件 Iris_data.txt

鸢尾花卉数据集的读取方法是自定义的，继承了pytorch的Dataset类，详见文件 data_loader.py

# 三、Fully connected network
模型的搭建，训练，验证和测试详见文件 fully_connected_network.py 

开启模型的训练，验证和测试只需要在IDE中执行fully_connected_network.py脚本即可；或者在控制台终端执行命令行fully_connected_network.py 脚本的log打印示例如下：
 
 <img src="https://user-images.githubusercontent.com/102544244/211217925-3b96de9c-48a1-4463-b328-3f73b820a85d.png" width="600px">
 
 # 三、Perception
 文件 perception.py 是纯python脚本实现的感知机模型，不依赖于pytorch。实际上，与本项目：“使用pytorch基于神经网络实现鸢尾花分类”无关。实现在这里只是为了一些对感知机模型底层代码感兴趣的同学们。在脚本中，实现了一个简单的接受两个输入的感知机模型，并使用这个感知机模型，训练推理了真值表中的“and” 功能。
