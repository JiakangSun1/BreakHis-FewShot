from todata import *
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
import torch.nn.functional as F
from our_model import VGGInceptionNet
from bilstm import BidirectionalLSTM
import argparse
import pandas as pd
parser = argparse.ArgumentParser()
parser.add_argument("--mode", help="train or val or test", type=str,default='test',choices=['train and val','test'])
parser.add_argument("--dim", help="特征维度", type=int,default=60)
parser.add_argument("--epoch", help="epoch", type=int,default=10)
parser.add_argument("--K_way", help="k", type=int,default=3)
parser.add_argument("--shot", help="shot of train or val", type=int,default=5,choices=[1,5])
parser.add_argument("--train_query", help="query of train", type=int,default=1)
parser.add_argument("--val_query", help="query of val", type=int,default=3,choices=[2,3])
parser.add_argument("--train_list_size", help="train_list_size", type=int,default=140)
parser.add_argument("--val_list_size", help="val_list_size", type=int,default=10,choices=[15,10])
parser.add_argument("--test_list_size", help="test_list_size", type=int,default=10)
parser.add_argument("--dataset", help="dir of dataset", type=str,default='E:/100X/100X')
args = parser.parse_args()


acc2csv = {
    "stepacc": [],
    "acc": []
}
loss2csv={
    "steploss": [],
    "loss":[]
}
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cnn_net = VGGInceptionNet().to(self.device)
        self.lstm = BidirectionalLSTM(layer_sizes=[int(args.dim/2)], batch_size=1, vector_dim=args.dim).to(self.device)

    def forward(self,x):
        x=self.cnn_net(x)
        x=x.unsqueeze(0)
        x=self.lstm(x)
        x=x.squeeze()
        return x
class Prototypicl_Net():
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net=Net().to(self.device)
        self.z_dim = args.dim
        # 优化器
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.net.parameters()), lr=0.0005)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 1, gamma=0.2, last_epoch=-1)
        print("使用：   ", self.device)

    def cal_euc_distance(self, query_z, center, K_way, N_query):
        center = center.unsqueeze(0).expand(K_way * N_query, K_way, self.z_dim)  # (K_way*N_query,K_way,z_dim)
        query_z = query_z.unsqueeze(1).expand(K_way * N_query, K_way, self.z_dim)  # (K_way*N_query,K_way,z_dim)
        cos_similarity = torch.nn.functional.cosine_similarity(query_z, center, dim=2)
        return torch.pow(query_z - center, 2).sum(2),cos_similarity  # (K_way*N_query,K_way)

    def loss_acc(self, query_z, center, K_way, N_query):
        target_inds = torch.arange(0, K_way).view(K_way, 1).expand(K_way, N_query).long().to(self.device)  # shape=(K_way, N_query)
        distance,distance1 = self.cal_euc_distance(query_z, center, K_way, N_query)  # (K_way*N_query,K_way)
        predict_label = torch.argmin(distance, dim=1)  # (K_way*N_query)
        predict_label1=torch.argmax(distance1,dim=1)
        acc = torch.eq(target_inds.contiguous().view(-1),predict_label).float().mean()  # 准确率
        acc1 = torch.eq(target_inds.contiguous().view(-1), predict_label1).float().mean()  # 准确率
        loss = F.log_softmax(-distance, dim=1).view(K_way,N_query, K_way)  # (K_way,N_query,K_way)
        loss = - \
            loss.gather(dim=2, index=target_inds.unsqueeze(2)).view(-1).mean()
        loss1=F.log_softmax(distance1,dim=1).view(K_way,N_query, K_way)
        loss1 = - \
            loss1.gather(dim=2, index=target_inds.unsqueeze(2)).view(-1).mean()
        loss=0.6*loss+0.4*loss1

        acc=(acc1+acc)/2

        return loss, acc

    def set_forward_loss(self, K_way, N_shot, N_query, sample_datas):

        z= self.net(sample_datas)
        z = z.view(K_way, N_shot + N_query, -1)
        support_z = z[:, :N_shot]
        query_z = z[:, N_shot:].contiguous().view(K_way * N_query, -1)
        center = torch.mean(support_z, dim=1)
        return self.loss_acc(query_z, center, K_way, N_query)

    def train(self, epochs,K_way,N_shot,N_query,train_list_size,name):
        """
            进行一个episode的训练，随机采样N个类，每个类使用K个数据集。
        """
        self.net.train()
        train_list=CustomDataset(K_way,N_shot,N_query,train_list_size,name)
        train_loader = DataLoader(train_list, batch_size=1, shuffle=True)

        step=1
        for epoch in range(epochs):
            for i, batch_data in enumerate(train_loader):
                data = batch_data[0].to(self.device)
                loss, acc = self.set_forward_loss(K_way, N_shot, N_query, data)
                if step % 5 == 0:
                    loss2csv["steploss"].append(step)
                    loss2csv["loss"].append(loss)
                self.optimizer.zero_grad()
                loss.requires_grad_(True)
                loss.backward()
                self.optimizer.step()

                if step%20==0:
                    print("=====================step is {}======================".format(step))
                    val_acc1 = self.eval_model(args.K_way, N_shot, args.val_query, args.val_list_size, name)
                    print("验证集：1-shot：{:.4}".format(val_acc1))
                    acc2csv["stepacc"].append(step)
                    acc2csv["acc"].append(val_acc1)
                step+=1
            self.scheduler.step()
            print("=======================epoch is {}==============================".format(epoch))
            val_acc2 = self.eval_model(args.K_way, N_shot, args.val_query, args.val_list_size,name)
            print("验证集：1-shot：{:.4}".format(val_acc2))
        df = pd.DataFrame(loss2csv)
        df1 = pd.DataFrame(acc2csv)
        torch.save(self.net.state_dict(), '5shotourmodel400.pth')
        df.to_csv('C:/Users/Casit/Desktop/5shot400loss.csv', index=False)
        df1.to_csv('C:/Users/Casit/Desktop/5shot400acc.csv', index=False)
    def eval_model(self,K_way, N_shot, N_query, val_list_size,name):
        self.net.eval()
        val_list = CustomDataset_test(K_way, N_shot, N_query, val_list_size, name)
        accs = []
        for i, batch_data in enumerate(val_list):
            data= batch_data.to(self.device)
            loss, acc = self.set_forward_loss(K_way, N_shot, N_query,data)
            accs.append(acc.item())
        self.net.train()
        return sum(accs) / val_list_size
    def test_model(self,K_way, N_shot, N_query, test_list_size,name):
        self.net.load_state_dict(torch.load('ourmodel100.pth'))
        self.net.eval()
        test_list = CustomDataset_test(K_way, N_shot, N_query, test_list_size, name)
        accs = []
        for i, batch_data in enumerate(test_list):
            data = batch_data.to(self.device)
            loss, acc = self.set_forward_loss(K_way, N_shot, N_query, data)
            accs.append(acc.item())
        return sum(accs) / test_list_size
if args.mode=='train and val':
    net = Prototypicl_Net()
    net.train(args.epoch, args.K_way,args.shot,args.train_query,args.train_list_size,name=args.dataset)
else:
    avg = 0
    net = Prototypicl_Net()
    for i in range(10):
        acc = net.test_model(args.K_way, args.shot, args.val_query, args.test_list_size, name=args.dataset)
        avg += acc
        print(acc)
    avg /= 10.0
    print(avg)


