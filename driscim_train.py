# driscim_train
import sys 
sys.path.append(r"E:\ML\Dog-Cat-GANs")
import discrim as dis
import gans1dset as data
import torch
import numpy as np
import sklearn.utils as utl
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class discrim():
    def __init__(self, model_type = 0):
        if model_type == 1:
            self.model = dis.discriminator()
        elif model_type == 2:
            self.model = dis.discriminator2()
        elif model_type == 3:
            self.model = dis.discriminator3()
        elif model_type == 4:
            self.model = dis.discriminator4()
        elif model_type == 5:
            self.model = dis.discriminator5()
        elif model_type == 6:
            self.model = dis.discriminator_dense()
        elif model_type == 7:
            self.model = dis.discriminator_encoder()
        elif model_type == 8:
            self.model = dis.discriminator_resnet()
        else:
            print('error')
        self.x,self.y,self.xx,self.yy = data.train_test_data()
        #for i in range(8,10):
            #data.visualize(self.x[i])
    
    def train(self, epochs = 10, lr = 0.01, batch_size = 100):
        self.model.to(device)
        loss_list = []
        dev_loss_list = []
        optimizer = torch.optim.Adam(self.model.parameters(), lr = lr, weight_decay= lr/100, amsgrad=True)
        criterion = torch.nn.BCELoss()
        for epoch in range(epochs):
            x,y = utl.shuffle(self.x,self.y)
            if epoch%5 == 0:
                print('Epoch',epoch)
            for batch in range(batch_size,9900+1,batch_size):
                self.model.train()
                optimizer.zero_grad()
                # Forward pass
                y_pred = self.model(torch.from_numpy(x[batch-batch_size:batch]).float().to(device))
                loss = criterion(y_pred.squeeze(), torch.reshape(torch.from_numpy(y[batch-batch_size:batch]).to(device),[len(y_pred.squeeze())]).float())
                # Backward pass
                loss.backward()
                optimizer.step()
                #dev/test set loss 
                if epoch%5 == 0 and batch%300 == 0:
                    #save_model('E:\ML\Dog-Cat-GANs\train-discriminator-params.pth')
                    print('     Batch {}:\n           train loss: {}'.format(int(batch/batch_size), loss.item()))
                    loss_list.append(loss)
                    '''
                    self.model.eval()
                    y_pred = self.model(torch.from_numpy(self.xx).float())
                    loss = criterion(y_pred.squeeze(), torch.reshape(torch.from_numpy(self.yy),[len(y_pred.squeeze())]).float())
                    dev_loss_list.append(loss)
                    print('           dev loss: {}'.format(loss.item()))
                    '''
                
        loss_list = [i.cpu().detach().numpy() for i in loss_list]
        dev_loss_list = [i.detach().numpy() for i in dev_loss_list]
        data.plt.figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')
        #data.plt.plot(dev_loss_list, label = 'Dev Loss', color = 'red')
        data.plt.plot(loss_list, label = 'Train Loss', color = 'black')
        data.plt.ylabel('BCE Loss')
        data.plt.xlabel('{} Epochs'.format(epochs))
        data.plt.title('Discriminator Training Loss')
        data.plt.legend()
        data.plt.show()
        
        return loss_list
    
    def save_model(self,path):
        torch.save(self.model.state_dict(), path)
    
    def load_model(self,path):
        self.model.load_state_dict(torch.load(path))
    
    def evaluate(self,optm=0.5):
        evl = self.model.eval().cpu()
        print(self.xx.shape)
        y_pred = evl(torch.from_numpy(self.xx).float())
        a=0
        FP=0
        TP=0
        TN=0
        FN=0
        for i in range(len(self.yy)):
            if(y_pred.detach().numpy().reshape(self.yy.shape)[i] > optm and self.yy[i] < optm):
                FP+=1
            elif(y_pred.detach().numpy().reshape(self.yy.shape)[i] < optm and self.yy[i] < optm):
                TN+=1
            elif(y_pred.detach().numpy().reshape(self.yy.shape)[i] > optm and self.yy[i] > optm):
                TP+=1
            elif(y_pred.detach().numpy().reshape(self.yy.shape)[i] < optm and self.yy[i] > optm):
                FN+=1
        FPR = FP/(FP+TN)
        TPR = TN/(TN+FP)
        for i in range(len(self.yy)):
                if (self.yy[i] > optm and y_pred.detach().numpy().reshape(self.yy.shape)[i] > optm) or (self.yy[i] < optm and y_pred.detach().numpy().reshape(self.yy.shape)[i] < optm):
                    a += 1
        a = a / len(self.yy)
        data.plt.figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')
        [data.plt.plot(y_pred.detach().numpy().reshape(self.yy.shape)[i],i/len(self.yy),'.',color=('blue' if self.yy[i] > 0.5 else 'red')) for i in range(len(self.yy))]
        [data.plt.plot(optm,0+i/len(self.yy),'|',color='red') for i in range(len(self.yy))]
        data.plt.ylabel('normalized example #')
        data.plt.xlabel('prediction Value')
        data.plt.title('Prediction Clusters, accuracy: {0}, fpr: {1}, tpr: {2}'.format(round(a,2),round(FPR,2),round(TPR,2)))
        data.plt.show()
        #confusion matrix
        print('                ### Confusion Matrix ###')
        print('              Predicted Dog   Predicted Cat')
        print('Actual Dog      ',TN,'             ',FP,'             ',FP+TN)
        print('Actual Cat      ',FN,'             ',TP,'             ',TP+FN)
        print('\n                ',FN+TN,'             ',TP+FP)
        print('accuracy:', a)
        return y_pred.detach().numpy().reshape(self.yy.shape),self.yy
    
    def set_evaluate(self,x,optm=0.5):
        evl = self.model.eval().cpu()
        y_pred = evl(torch.from_numpy(x).float())
        return y_pred.detach().numpy()
    
#%%
model = discrim(8)
print(model.model)
#%%
loss = model.train(epochs=200,lr=0.00000001, batch_size=11)
#%%
y, yy = model.evaluate(optm=0.5)
#_, _ = model.evaluate(optm=0.7)#optm=0.9,0.79 for d3
#_, _ = model.evaluate(optm=0.71)
#%%
for i in range(len(y)):
    data.visualize(model.xx[i],show_exp=True,ye=yy[i]>0.5,yp=y[i]>0.5)
#%%
model.load_model('E:\ML\Dog-Cat-GANs\D_resnet_gpu.pth')
#%%
#model.save_model('E:\ML\Dog-Cat-GANs\D_resnet_gpu.pth')
#%%
artingi = [data.img_load('E:\ML\Dog-Cat-GANs\Dataset\cats\catarty (%d).jfif'%(i+1),show=False) for i in range(7)]
artingi = np.array(artingi)
arty = np.zeros((7,3,128,128))
for i in range(7):
    for j in range(3):
        arty[i,j,:,:] = artingi[i,:,:,j]
yp = model.set_evaluate(arty,optm=0.5)
[data.visualize(artingi[i].T,show_exp=True,ye=1,yp=yp[i]>0.5)for i in range(7)]
print(yp)
#%%
'''
_,_,x,y=data.train_test_data()
x = x[:5]
y = y[:5]
[data.visualize(x[i],show_exp=True,ye=y[i],yp='before')for i in range(5)]
#np.random.shuffle(x)
xx,yy = utl.shuffle(x,y)#shuffle(x,y)
[data.visualize(xx[i],show_exp=True,ye=yy[i],yp='after')for i in range(5)]
'''
#%%
'''
model = dis.discriminator_resnet()
print(model)
artingi = [data.img_load('E:\ML\Dog-Cat-GANs\Dataset\cats\catarty (%d).jfif'%(i+1),show=False) for i in range(7)]
artingi = np.array(artingi)
arty = np.zeros((7,3,128,128))
for i in range(7):
    for j in range(3):
        arty[i,j,:,:] = artingi[i,:,:,j]
model.eval()
print(model(torch.from_numpy(arty).float())[0])
'''