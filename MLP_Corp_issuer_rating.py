import torch, time, copy
import numpy as np
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim import lr_scheduler, Adam, SGD
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from matplotlib import pyplot as plt

import math
import time
import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# parameters
training_epochs = 10000
batch_size = 16
learning_rate = 0.01

class TensorData(Dataset):
    #전처리를 하는 부분
    def __init__(self, x_data, y_data):
        self.x_data = torch.FloatTensor(x_data)
        self.y_data = torch.tensor(y_data).type(torch.int64)
        self.y_data = F.one_hot(self.y_data)
        self.y_data = self.y_data.type(torch.FloatTensor)
        self.len = self.y_data.shape[0]
    #특정 1개의 샘플을 가져오는 부분
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index] 
    
    def __len__(self):
        return self.len
    
class NeuralNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size): #input: sample의 size  hidden: output의 size
        super(NeuralNet, self).__init__()
        self.input_layer  = torch.nn.Linear(input_size, 8)
        self.hidden_layer1 = torch.nn.Linear(8, 16)
        self.hidden_layer2 = torch.nn.Linear(16, 4)
        self.output_layer = torch.nn.Linear(4, output_size)
        self.dropout = torch.nn.Dropout(0.2)
        self.relu = torch.nn.ReLU()
        self.soft = torch.nn.Softmax(dim=1)
    def forward(self, x):        
        output = self.relu(self.input_layer(x))
        output = self.relu(self.hidden_layer1(output))
        # output = self.dropout(output)
        output = self.relu(self.hidden_layer2(output))
        # output = self.dropout(output)
        output = self.soft(self.output_layer(output))
        return output
    
def show_mlp(epochs, loss, acc):
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)


    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(epochs, loss)
    plt.title('Training Loss')
    plt.subplot(1,2,2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(epochs, acc)
    plt.title('Training Acc')
    # plt.show()
        

hiddensize = 1
model = NeuralNet(7,hiddensize,2).to(device)
criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
# factor: lr 감소시키는 비율 / patience: 얼마 동안 변화가 없을 때 lr을 감소시킬지 / threshold: Metric의 변화가 threhold이하일 시 변화가 없다고 판단
# eps: lr의 감소 최소치 지정
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2,
                                           threshold=0.0001, threshold_mode='rel', min_lr=0, eps=1e-6, verbose=False)



##############추출할 test용 data뽑기### 
choose_num = 3
best_list = []

#K-fold cross validation
for choose_num in range(4):

    ## 데이터 4등분을 하고 train test data 만들기 ###########################################
    import itertools
    data=pd.read_csv('D:/OneDrive - 대전대학교/jupyter/financial_data.csv',encoding='cp949')
    data_X = data.iloc[:,1:9].values
    #데이터 스케일링
    scaler = StandardScaler()
    scaler.fit(data_X)
    X = scaler.transform(data_X)
    df = pd.DataFrame(X)
    a=df.iloc[:,[0,2,3,4,5,6,7]]
    X = a.values.tolist()

    Y_t = data['Y'].replace(['AAA','AA','A','BBB','BB','B','CCC','CC','C','D'],
                            [0,0,0,0,1,1,1,1,1,1])
    Y = Y_t.tolist()

    X_list = []
    X_list.append(X[:230])
    X_list.append(X[230:460])
    X_list.append(X[460:690])
    X_list.append(X[690:])
    test_x = X_list[choose_num]
    del X_list[choose_num]
    X_list=list(itertools.chain(*X_list))
    train_x = np.array(X_list)

    #y data 4등분으로 나누기
    Y_list = []
    Y_list.append(Y[:230])
    Y_list.append(Y[230:460])
    Y_list.append(Y[460:690])
    Y_list.append(Y[690:])

    test_y = Y_list[choose_num]
    del Y_list[choose_num]
    train_y = list(itertools.chain(*Y_list))
    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size = 0.2, shuffle=True)

    ####################################################################################################################
    # data=pd.read_csv('D:/OneDrive - 대전대학교/jupyter/financial_data.csv',encoding='cp949')
    # data_X = data.iloc[:,1:9].values
    # #데이터 스케일링
    # scaler = MinMaxScaler()
    # scaler.fit(data_X)
    # X = scaler.transform(data_X)

    # df = pd.DataFrame(X)
    # a=df.iloc[:,[0,1,2,5,7]]
    # X = a.values.tolist()
    # # X = data_X

    # # Y_t = data['Y'].replace(['AAA','AA','A','BBB','BB','B','CCC','CC','C','D'],
    # #                            [0,1,2,3,4,5,6,7,8,9])
    # # Y_t = data['Y'].replace(['AAA','AA','A','BBB','BB','B','CCC','CC','C','D'],
    # #                            [0,0,0,1,1,2,2,2,2,2])
    # Y_t = data['Y'].replace(['AAA','AA','A','BBB','BB','B','CCC','CC','C','D'],
    #                            [0,0,0,0,1,1,1,1,1,1])
    # Y = Y_t.tolist()

    # train_x, test_x, train_y, test_y = train_test_split(X,Y, test_size = 0.2, shuffle=True)
    # train_x, val_x, train_y, val_y = train_test_split(train_x,train_y, test_size = 0.2, shuffle=True)
    ############################################################################################################


    trainsets = TensorData(train_x, train_y)
    trainloader = torch.utils.data.DataLoader(trainsets, batch_size=batch_size, shuffle=False, drop_last=False)
    validationsets = TensorData(val_x, val_y)
    validationloader = torch.utils.data.DataLoader(validationsets, batch_size=batch_size, shuffle=False, drop_last=False)
    testsets = TensorData(test_x, test_y)
    testloader = torch.utils.data.DataLoader(testsets, batch_size=batch_size, shuffle=False, drop_last=False)



    ##Training
    start = time.time()
    math.factorial(1234567)
    loss_list = []
    acc_list = []
    best_loss = 0
    best_acc = 0

    for epoch in (range(training_epochs)):
        model.train()
        y_data_len = 0
        loss_sum = 0
        loss_cost = 0
        correct=0
        for i, (X, label) in enumerate(trainloader):    
            X = X.to(device)
            label = label.to(device)

            #경사도 초기화
            optimizer.zero_grad()
            train_output=model(X)
            Loss = criterion(train_output, label)
            #Backpropagation
            Loss.backward()
            optimizer.step()

            loss_sum+=Loss.item()
            correct += (train_output.argmax(1) == label.argmax(1)).sum().item()
        
        #모델 검증
        val_loss = 0
        val_correct = 0
        model.eval()
        with torch.no_grad():
            for j,(V, targets) in enumerate(validationloader):
                V = V.to(device)
                targets = targets.to(device)

                val_output = model(V)
                loss = criterion(val_output, targets)
                val_loss += loss.item()
                val_correct += (val_output.argmax(1) == targets.argmax(1)).sum().item()
                
        train_loss = (loss_sum / len(trainloader))
        train_acc = (correct/len(trainloader.dataset))*100
        loss_list.append(train_loss)
        acc_list.append(train_acc)
        # print('Epoch: {:4d}/{} \tTraining Loss: {:.6f} \tTraining Accuracy: {:.2f}%'.format(epoch+1, training_epochs, train_loss, train_acc))

        #learning rate 출력
        visual_lr = optimizer.param_groups[0]['lr']
        val_loss /= len(validationloader)
        val_acc = 100. * val_correct / len(validationloader.dataset)
        # loss_list.append(val_loss)
        # acc_list.append(val_acc)
        print('Ch: {}   Epoch: {:4d}/{} \tValidation Loss: {:.6f} \tValidation Accuracy: {:.2f}% \tLearning rate: {:.10f}'
            .format(choose_num, epoch+1, training_epochs, val_loss, val_acc, visual_lr))
        
        
        # 가장 loss가 적은 모델 저장
        if epoch == 0 or val_loss < best_loss:
            best_epoch = epoch
            best_loss = val_loss
            best_acc = val_acc
            best_lr = visual_lr
            low_loss_model = copy.deepcopy(model.state_dict())
            print('BEST \nEpoch: {} \tBest Loss: {:.6f} \tBest Accuracy: {:.2f}% \tLearning rate: {:.10f}'
            .format(best_epoch+1, best_loss, best_acc, best_lr))
        scheduler.step(val_loss)
        


    epoch_list = np.arange(1,training_epochs+1)

    end = time.time()
    sec = (end - start)
    result_list = str(datetime.timedelta(seconds=sec)).split(".")
    print(result_list[0])


    ##Testing

    model.load_state_dict(low_loss_model)
    model.eval()

    with torch.inference_mode():
        list_correct=[]
        list_target=[]
        cost = 0
        test_correct = 0
        test_loss = 0
        for i, (x_test, y_test) in enumerate(testloader):
            x_test = x_test.to(device)
            y_test = y_test.to(device)
            
            prediction = model(x_test)
            cost = criterion(prediction, y_test)
            
            test_loss += cost.item()
            test_correct += (prediction.argmax(1) == y_test.argmax(1)).sum().item()
            
            list_correct += prediction.argmax(1).tolist()
            list_target += y_test.argmax(1).tolist()

        print(list_correct)
        print(list_target)
        loss_avg = test_loss / len(testloader)
        acc = (test_correct/len(testloader.dataset))*100
        print(f'Test Accuracy: {(acc):>0.1f}%     Loss: {loss_avg:>8f} \n')

    show_mlp(epoch_list, loss_list, acc_list)
    plt.savefig('./corp_issure_rating_png/ch{},feature{},eps{},batch{},lr{},hidden{},Acc{:.1f}.png'
                .format(choose_num,a.shape[1],training_epochs,batch_size,learning_rate,hiddensize,acc),dpi=100)

    ##loss가 가장 적은 거 출력
    print('BEST \nEpoch: {} \tBest Loss: {:.6f} \tBest Accuracy: {:.2f}% \tLearning rate: {:.10f}'
            .format(best_epoch+1, best_loss, best_acc, best_lr))

    best_list.append(str("BEST Choose: {}   Epoch: {}   Best Loss: {:.6f}   Best Accuracy: {:.2f}%   Learning rate: {:.10f}"
            .format(choose_num, best_epoch+1, best_loss, best_acc, best_lr)))

    torch.save(low_loss_model,'./corp_issure_rating_save/choose{},eps{},Acc{:.1f}.pth'.format(choose_num,training_epochs, acc))

print(best_list[0],"\n",best_list[1],"\n",best_list[2],"\n",best_list[3],"\n")



# import winsound as sd
# def beepsound():
#     fr = 2000    # range : 37 ~ 32767
#     du = 1000     # 1000 ms ==1second
#     sd.Beep(fr, du) # winsound.Beep(frequency, duration)
# beepsound()

