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
from sklearn.model_selection import KFold

import math
import time
import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# parameters
training_epochs = 1000
batch_size = 64
learning_rate = 0.001

class TensorData(Dataset):
    #전처리를 하는 부분
    def __init__(self, x_data, y_data):
        self.x_data = torch.FloatTensor(x_data)
        self.y_data = torch.FloatTensor(y_data)
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
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(epochs, loss)
    plt.title('Validation Loss')
    
    plt.subplot(1,2,2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(epochs, acc)
    plt.title('Validation Acc')
    plt.show()
        

hiddensize = 1
kfold = KFold(n_splits=4)
model = NeuralNet(7,125,10).to(device)
criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
# factor: lr 감소시키는 비율 / patience: 얼마 동안 변화가 없을 때 lr을 감소시킬지 / threshold: Metric의 변화가 threhold이하일 시 변화가 없다고 판단
# eps: lr의 감소 최소치 지정
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100,
                                           threshold=0.0001, threshold_mode='rel', min_lr=0, eps=1e-7, verbose=False)


best_list = []

data=pd.read_csv('D:/OneDrive - 대전대학교/jupyter/financial_data.csv',encoding='cp949')
data2=pd.read_csv('D:/OneDrive - 대전대학교/jupyter/gan.csv',encoding='cp949')
data_X = data.iloc[:,1:9]
data_X2 = data2.iloc[:,0:8]
#데이터 스케일링
scaler = MinMaxScaler()
scaler.fit(data_X)
X = scaler.transform(data_X)
df = pd.DataFrame(X)
data_X = df.iloc[:,[0,2,3,4,5,6,7]].values
data_X2 = data_X2.iloc[:,[0,2,3,4,5,6,7]].values
X = np.concatenate((data_X, data_X2))

Y_t = data['Y'].replace(['AAA','AA','A','BBB','BB','B','CCC','CC','C','D'],
                            [0,1,2,3,4,5,6,7,8,9])
y = data2['Y'].replace([0,1,2,3,4,5,6,7,8,9],
                        [0,1,2,3,4,5,6,7,8,9])
Y_t=pd.concat([Y_t,y])
onehot_encoded = pd.get_dummies(Y_t)
Y = onehot_encoded.values
test_x = X[:585]
test_y = Y[:585]
X = X[585:]
Y = Y[585:]

trainsets = TensorData(X, Y)
testsets = TensorData(test_x, test_y)
testloader = DataLoader(testsets, batch_size=batch_size, shuffle=False, drop_last=False)


##k-fold Training
start = time.time()
math.factorial(1234567)
best_fold = 0
best_loss = 0
best_acc = 0
fold_val_loss = 0
fold_loss_list = []

for fold, (train_fold, val_fold) in enumerate(kfold.split(trainsets)):
    print("Fold: {}".format(fold+1))
    train_data = torch.utils.data.Subset(trainsets, train_fold)
    trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    val_data = torch.utils.data.Subset(trainsets, val_fold)
    validationloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    
    loss_list = []
    acc_list = []
    best_loss = 0
    best_acc = 0

    for epoch in (range(training_epochs)):
        loss_sum = 0
        loss_cost = 0
        correct=0
        #모델 학습
        model.train()
        for i, (X, label) in enumerate(trainloader):
            X = X.to(device)
            label = label.to(device)
            #경사도 초기화
            optimizer.zero_grad()
            train_output=model(X)
            Loss = criterion(train_output, label)
            #Backpropagation
            Loss.backward()
            #가중치 업데이트
            optimizer.step()

            #batch 시이즈 만큼 나온 loss 더하기
            loss_sum+= Loss.item()
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
    #     loss_list.append(train_loss)
    #     acc_list.append(train_acc)
    #     print('Epoch: {} \tTraining Loss: {:.6f} \tTraining Accuracy: {:.2f}%'.format(epoch+1, train_loss, train_acc))

        #learning rate 출력
        visual_lr = optimizer.param_groups[0]['lr']
        val_loss /= len(validationloader)
        val_acc = 100. * val_correct / len(validationloader.dataset)
        loss_list.append(val_loss)
        acc_list.append(val_acc)
        print('Epoch: {} \tValidation Loss: {:.6f} \tValidation Accuracy: {:.2f}% \tLearning rate: {:.10f}'
              .format(epoch+1, val_loss, val_acc, visual_lr))

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
    show_mlp(epoch_list, loss_list, acc_list)


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
    print(f'Train \t\t\t Loss: {fold_val_loss:>8f}')
    print(f'Test Accuracy: {(acc):>0.1f}%     Loss: {loss_avg:>8f} \n')
    
torch.save(low_loss_model,'./corp_issure_rating_save/fold{},eps{},Acc{:.1f}.pth'.format(fold+1,training_epochs, acc))




# import winsound as sd
# def beepsound():
#     fr = 2000    # range : 37 ~ 32767
#     du = 1000     # 1000 ms ==1second
#     sd.Beep(fr, du) # winsound.Beep(frequency, duration)
# beepsound()

