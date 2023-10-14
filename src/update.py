



import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset


class DatasetSplit(Dataset):
    """

    """
    def __init__(self,dataset,idxs):
        self.dateset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dateset[self.idxs[item]]
        return torch.as_tensor(image),torch.as_tensor(label)


class LocalUpdate(object):
    def __init__(self,args,dataset,idxs,logger):
        self.args = args
        self.logger = logger
        self.trainloader,self.validloade,self.teseloader = self.train_val_test(dataset,list(idxs))

        self.device = 'cuda' if args.gpu else 'cpu'
        # 默认使用NLL 损失函数
        self.criterion = nn.NLLLoss().to(self.device)

    def train_val_test(self,dataset,idxs):
        """
        :param dateset:
        :param idxs:
        :return:
        """
        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset,idxs_train),
                                 batch_size=self.args.local_bs,shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset,idxs_val),
                                 batch_size=int(len(idxs_val)/10),shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset,idxs_test),
                                batch_size=int(len(idxs_test)/10),shuffle=False)
        return trainloader,validloader,testloader


    def update_weights(self,model,global_round):
        # 设置训练模型
        model.train()
        epoch_loss = []

        # 设置优化方案为局部更新
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(),lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(),lr=self.args.lr,
                                         weight_decay=1e-4)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx,(images,labels) in enumerate(self.trainloader):
                images , labels = images.to(self.device),labels.to(self.device)

                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs,labels)
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| 全局轮次 : {} | 本地轮次 : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round,iter,batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss',loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(),sum(epoch_loss) / len(epoch_loss)

    def inference(self,model):
        """ 返回推断精确值和损失值
        """

        model.eval()
        loss,total,correct = 0.0 ,0.0,0.0

        for batch_idx,(images,labels) in enumerate(self.teseloader):
            images,labels = images.to(self.device),labels.to(self.device)

            # 推断
            outputs = model(images)
            batct_loss = self.criterion(outputs,labels)
            loss += batct_loss.item()

            # 预测
            _, pred_labels = torch.max(outputs,1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels,labels)).item()
            total += len(labels)

        accuracy = correct / total
        return accuracy,loss


def test_inference(args,model,test_dataset):
    """ return 测试精确度和损失
    """

    model.eval()
    loss,total,correct = 0.0 ,0.0,0.0

    device = 'cuda' if args.gpu else 'cpu'
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset,batch_size=128,
                            shuffle=False)

    for batch_idx,(images,labels) in enumerate(testloader):
        images,labels = images.to(device),labels.to(device)

        # 推断
        outputs = model(images)
        batct_loss = criterion(outputs,labels)
        loss += batct_loss.item()

        # 预测
        _, pred_labels = torch.max(outputs,1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels,labels)).item()
        total += len(labels)

    accuracy = correct / total
    return accuracy,loss

