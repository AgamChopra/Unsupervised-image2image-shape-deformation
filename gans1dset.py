# gans1dset

import cv2
import numpy as np
from matplotlib import pyplot as plt 
import random
import sys 
sys.path.append(r"E:\ML\Dog-Cat-GANs\Dataset\cats")

def load_cats():
    cat_list = []
    for i in range(5000):
        img = cv2.imread('E:\ML\Dog-Cat-GANs\Dataset\cats\cat (%d).jpg'%(i+1))
        cat_list.append(cv2.resize(img, dsize=(128, 128), interpolation=cv2.INTER_CUBIC))
        #if i%200 == 0:
        #    plt.imshow(cv2.cvtColor(cat_list[i], cv2.COLOR_BGR2RGB))
        #    plt.show()  
    print('.cat data loaded')
    return cat_list
#load_cats()

def load_not_cats():
    not_cat_list = []
    for i in range(5000):
        img = cv2.imread('E:\ML\Dog-Cat-GANs\Dataset\cats\catnt (%d).jpg'%(i+1))
        not_cat_list.append(cv2.resize(img, dsize=(128, 128), interpolation=cv2.INTER_CUBIC))
        #if i%200 == 0:
        #    plt.imshow(cv2.cvtColor(not_cat_list[i], cv2.COLOR_BGR2RGB))
        #    plt.show() 
    print('..not cat data loaded')
    return not_cat_list
#load_not_cats()

def load_dataset():
    random.seed(15)
    cat = load_cats()
    catnt = load_not_cats()
    rand_seed = random.sample(range(10000), 10000)
    x = []
    y = []
    for i in range(10000):
        if rand_seed[i] < 5000:
            x.append(cat[rand_seed[i]].T)
            y.append(1)
        else:
            x.append(catnt[rand_seed[i]-5000].T)
            y.append(0)
    print('...data stitching and randomization finished')
    return x,y

def train_test_data():
    x,y = load_dataset()
    print('....train test data loaded')
    return np.stack(x[:9900]),np.stack(y[:9900]),np.stack(x[9900:]),np.stack(y[9900:])
    # returns train x, train y, test x, and test y sets in (1900, 128, 128, 3) (1900,) (100, 128, 128, 3) (100,) numpy.ndarray format respectively.
    
def doge_data():
    ds = load_not_cats()
    x = []
    [x.append(i.T) for i in ds]
    return np.stack(x[:4950]),np.stack(x[4950:])

def cat_data():
    ds = load_cats()
    x = []
    [x.append(i.T) for i in ds]
    return np.stack(x)

def visualize(x,show_exp=False,ye=0,yp=0):
    plt.imshow(cv2.cvtColor(x.T, cv2.COLOR_BGR2RGB))
    if show_exp:
        plt.title('label: {0}, prediction: {1}'.format(ye,yp))
    plt.show()
    
def img_load(path, show = True):
    img = cv2.imread(path)
    x = cv2.resize(img, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
    if show:
        plt.imshow(cv2.cvtColor(x, cv2.COLOR_BGR2RGB))
        plt.show()  
    print('image loaded!')
    return x

'''
x,y,xx,yy = train_test_data()
for i in range(20):
    visualize(x[i],show_exp= True, ye = y[i], yp = 'NA')
print(x.shape,y.shape,xx.shape,yy.shape)
print(type(x))
'''