import os 
import argparse 
import sys
from torch.utils.data import DataLoader
from tqdm import tqdm 
import torch
import torch.optim as optim  
import torch.nn as nn 
from data_loader import iris_dataload

parser = argparse.ArgumentParser()
parser.add_argument('--num_classes', type=int, default=100, help='the number of classes')
parser.add_argument('--epochs', type=int, default=20, help='the number of training epoch')
parser.add_argument('--batch_size', type=int, default=16, help='batch_size for training')
parser.add_argument('--lr', type=float, default=0.005, help='star learning rate')   
parser.add_argument('--data_path', type=str, default="/mnt/d/Codes/GNN/NN/Iris_data.txt") 
parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
opt = parser.parse_args()  

# 初始化神经网络
class Neuralnetwork(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Neuralnetwork, self).__init__()
        self.layer1 = nn.Linear(in_dim, n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.layer3 = nn.Linear(n_hidden_2, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

# 定义当前模型的训练环境
device = torch.device(opt.device if torch.cuda.is_available() else "cpu") 

# 划分数据集并加载
custom_dataset = iris_dataload( "./Iris_data.txt") 
train_size = int(len(custom_dataset) * 0.7)
validate_size = int(len(custom_dataset) * 0.2)
test_size = len(custom_dataset) - validate_size - train_size
train_dataset, validate_dataset, test_dataset = torch.utils.data.random_split(custom_dataset, [train_size, validate_size, test_size])
 
train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=False )
validate_loader = DataLoader(validate_dataset, batch_size=1, shuffle=False )
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False ) 
print("Training set data size:", len(train_loader)*opt.batch_size, ",Validating set data size:", len(validate_loader), ",Testing set data size:", len(test_loader)) 
 
# 定义推理过程，返回准确率。用于验证阶段和测试阶段
def infer(model, dataset, device):
    model.eval()
    acc_num = 0.0  
    with torch.no_grad(): 
        for  data in dataset:
            datas, labels = data
            outputs = model(datas.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            acc_num += torch.eq(predict_y, labels.to(device)).sum().item()  
    accuratcy = acc_num / len(dataset)
    return accuratcy 

# 定义训练，验证和测试过程
def main(args): 
    print(args)
 
    model = Neuralnetwork(4, 12, 6, 3).to(device) # 实例化模型
    loss_function = nn.CrossEntropyLoss() # 定义损失函数
    pg = [p for p in model.parameters() if p.requires_grad] # 定义模型参数
    optimizer = optim.Adam(pg, lr=args.lr) # 定义优化器
 
    # 定义模型权重存储地址
    save_path = os.path.join(os.getcwd(), 'results/weights')
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    # 开始训练过程
    for epoch in range(opt.epochs):
        ############################################################## train ######################################################
        model.train() 
        acc_num = torch.zeros(1).to(device)    # 初始化，用于计算训练过程中预测正确的数量
        sample_num = 0                         # 初始化，用于记录当前迭代中，已经计算了多少个样本
        # tqdm是一个进度条显示器，可以在终端打印出现在的训练进度
        train_bar = tqdm(train_loader, file=sys.stdout, ncols=100)
        for datas in train_bar :
            data, label = datas
            label = label.squeeze(-1)
            sample_num += data.shape[0]

            optimizer.zero_grad()
            outputs = model(data.to(device)) # output_shape: [batch_size, num_classes]
            pred_class = torch.max(outputs, dim=1)[1] # torch.max 返回值是一个tuple，第一个元素是max值，第二个元素是max值的索引。
            acc_num += torch.eq(pred_class, label.to(device)).sum()
 
            loss = loss_function(outputs, label.to(device)) # 求损失
            loss.backward() # 自动求导
            optimizer.step() # 梯度下降

            # print statistics 
            train_acc = acc_num.item() / sample_num  
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,  opt.epochs, loss)

        ############################################################## validate ###################################################### 
        val_accurate = infer(model = model, dataset=validate_loader, device=device)
        print('[epoch %d] train_loss: %.3f  train_acc: %.3f  val_accuracy: %.3f' %  (epoch + 1, loss, train_acc, val_accurate))    
        torch.save(model.state_dict(), os.path.join(save_path, "AlexNet.pth") )

        # 每次迭代后清空这些指标，重新计算 
        train_acc = 0.0
        val_accurate = 0.0 
    print('Finished Training')

    ################################################################# test ############################################################  
    test_accurate = infer(model = model, dataset = test_loader, device= device)
    print(' test_accuracy: %.3f' %  ( test_accurate))  

if __name__ == '__main__': 
    main(opt)



 