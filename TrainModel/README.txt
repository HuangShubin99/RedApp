文件夹：
checkpoints	保存下来的模型权重
FinalModel	意外得到的准确率更高（94%）的模型
test_data		测试用的图片集
train_data		训练用的图片集
模型文件：
model.tflite	tflite模型，用于Android
my_model.h5	tensorflow模型
程序文件：
classify.py	主函数，用于调用训练程序、测试程序......
convert.py	将tensorflow模型转化为tflitemoxing
data.py		图片数据的处理	
layer.py		模型创建
Pred.py		测试模型，预测结果
rename.py	重命名图片集文件名
Train.py		训练模型

环境：
Python 3.6.8
tensorflow-gpu 2.0.1
numpy 1.16.2
GTX1050 3GB  cuda10（显卡不同，环境可能也要不同）