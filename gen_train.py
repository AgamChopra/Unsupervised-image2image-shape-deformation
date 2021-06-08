# gen_train
import sys 
sys.path.append(r"E:\ML\Dog-Cat-GANs")
import discrim as dis
import gans1dset as data
import gen
import torch
import numpy as np
import sklearn.utils as utl
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class generator():
    def __init__(self, model_type = 0):
        self.G = gen.generator()
        self.G_inv = gen.generator()
        self.D = dis.discriminator_resnet()
        self.D.load_state_dict(torch.load('E:\ML\Dog-Cat-GANs\D_resnet_gpu.pth'))
        #dog data: train->4950 test->50
        temp_x,_ = data.doge_data()
        self.xd = temp_x[:1000, :,:,:]
        del temp_x
        #cat data: train->5000
        self.xc = data.cat_data()[:1000, :,:,:]
        '''
        ct = 0
        i = 1
        for child in self.D.children():
           if ct == 0:
                for param in child.parameters():
                    if i <= 310:
                        param.requires_grad = False
        '''
        
    def train(self, epochs = 10, lr = 0.01, batch_size = 100):
        #load models to device
        #self.G.to(device)
        #self.D.to(device)
        optimizerD = torch.optim.Adam(self.D.parameters(), lr = lr/10, weight_decay= lr/100, amsgrad=False)
        optimizerG = torch.optim.Adam(self.G.parameters(), lr = lr, weight_decay= lr/100, amsgrad=False)
        optimizerG_inv = torch.optim.Adam(self.G_inv.parameters(), lr = lr, weight_decay= lr/100, amsgrad=False)
        criterion = torch.nn.BCELoss()
        criterion_sim = torch.nn.SmoothL1Loss(beta = 1.0)
        criterion_inv = torch.nn.SmoothL1Loss(beta = 1.0) # introduce for cycle gans
        self.D.train()
        
        #self.G.cpu()
        #temp_x = self.G(torch.from_numpy(self.xd).float()).detach().numpy()
        
        for epoch in range(epochs):
            print('epoch:{} of {}'.format(epoch + 1,epochs))
            #create randomized dataset for D
            self.G.to(device)
            self.G_inv.to(device)
            self.D.to(device)
            temp_x = np.zeros(shape=self.xd.shape)
            for i in range(1000):
                temp_x[i] = self.G(torch.from_numpy(np.reshape(self.xd[i],(1,3,128,128))).float().to(device)).cpu().detach().numpy()
            temp_x = np.stack(temp_x, axis=0)
            
            x,y = utl.shuffle(np.concatenate((self.xc,temp_x)),np.concatenate((np.ones((1000,1)),np.zeros((1000,1)))))
            loss = 0
            del temp_x
            '''
            Discriminator, train over all batches, each batch is randomly sampled form xc->1, xd->0, and G(xd)->0  
            '''
            for batch in range(batch_size,1001,batch_size):# xc,xd,G(xd)
                optimizerD.zero_grad()
                y_pred = self.D(torch.from_numpy(x[batch-batch_size:batch]).float().to(device))
                loss = criterion(y_pred.squeeze(), torch.reshape(torch.from_numpy(y[batch-batch_size:batch]).float().to(device),[len(y_pred.squeeze())]).float())
                loss.backward()
                optimizerD.step()
            print('          Loss D: {}'.format(loss))   
            del loss,x,y
            
            '''
            Generator, train over all batches, each batch is randomly sampled from xd. 
            The goal is to use the loss of the updated D(that learnt to categorize G(xd) as 0),
            to update G's paramaters in order to minimize D(G(xd)) 
            ie- try to get the G(xd) distribution as close to xc distribution as possible.
            '''
            x = utl.shuffle(self.xd)
            self.G.train()
            self.G_inv.train()
            for batch in range(batch_size,1001,batch_size):
                optimizerG.zero_grad()
                x_transform = self.G(torch.from_numpy(x[batch-batch_size:batch]).float().to(device))
                y_pred = self.D(x_transform)
                similarity_loss = criterion_inv(torch.from_numpy(x[batch-batch_size:batch]).float().to(device),x_transform)
                #implement cycle gans
                x_transform = self.G_inv(x_transform)
                # for color and landscape correctness
                cycle_consistency_loss = criterion_sim(torch.from_numpy(x[batch-batch_size:batch]).float().to(device),x_transform)
                loss = criterion(y_pred.squeeze(), torch.reshape(torch.ones((batch_size,1)).to(device),[len(y_pred.squeeze())]).float()) + 0.5 * (cycle_consistency_loss) + 0.01 * (similarity_loss)
                loss.backward()
                optimizerG.step()
                optimizerG_inv.step()
           
            print('          Loss G: {}'.format(loss))
            del loss,cycle_consistency_loss,x
    
    def save_model(self,pathG,pathG_inv,pathD):
        torch.save(self.G.state_dict(), pathG)
        torch.save(self.G_inv.state_dict(), pathG_inv)
        torch.save(self.D.state_dict(), pathD)
    
    def load_model(self,pathG,pathG_inv,pathD):
        self.G.load_state_dict(torch.load(pathG))
        self.G_inv.load_state_dict(torch.load(pathG_inv))
        self.D.load_state_dict(torch.load(pathD))
   
#%%
model = generator()
#%%
model.train(epochs=10,lr=0.000001, batch_size=5)
#%%
model.load_model('E:\ML\Dog-Cat-GANs\G.pth','E:\ML\Dog-Cat-GANs\G_inv.pth','E:\ML\Dog-Cat-GANs\D.pth')
#%%
#model.save_model('E:\ML\Dog-Cat-GANs\G.pth','E:\ML\Dog-Cat-GANs\G_inv.pth','E:\ML\Dog-Cat-GANs\D.pth')
#%%
xx,x = data.doge_data()
x = x[:10]
xx = xx[:10]
#%%
model.G.to('cpu')
model.G.eval()
#x_transform = model.G(torch.from_numpy(x).float()).detach().numpy()
xx_transform = model.G(torch.from_numpy(xx).float()).detach().numpy()
#print(x.shape)
#print(x_transform.shape)
#%%
for i in range(5):
    data.visualize(xx[i])
    data.visualize((xx_transform[i]-np.min(xx_transform[i]))/(np.max(xx_transform[i])-np.min(xx_transform[i])),show_exp=True,ye='1',yp=model.D(xx_transform[i]))
print((xx_transform[0]-np.min(xx_transform[0]))/(np.max(xx_transform[0])-np.min(xx_transform[i])))
#%%
torch.cuda.empty_cache() #empty gpu cache instead of restarting kernel
#%%
ct = 0
i = 1
for child in model.D.children():
   if ct == 0:
        for param in child.parameters():
            if i <= 310:
                param.requires_grad = False
#%%
a = np.zeros((100, 3, 128, 128))
b = np.zeros((100, 3, 128, 128))
print(np.concatenate((a,b)).shape)
#%%
a = np.ones((100, 3, 128, 128))
temp_x = np.zeros((100, 3, 128, 128))  
for i in range(100):
    print(a[i].shape)
    temp_x[i] = np.reshape(a[i],(1,3,128,128))
    print(np.reshape(a[i],(1,3,128,128)).shape)
print(temp_x.shape)