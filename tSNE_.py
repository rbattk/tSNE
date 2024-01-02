# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 19:57:13 2021

@author: user1
"""
import numpy as np
import pandas as pd 
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
# ----------------------------------------------------------------------------#
#--Step 1: Upload data--------------------------------------------------------#
path = "C:\\Users\\rabia\\Desktop\\Published\\seeds_dataset.txt"
seed_data = np.loadtxt (path)
 
sd = pd.DataFrame(seed_data, columns=(['area','perimete','compactnes', 
                                       'length of kerne','width of kernel',
                                       'asymmetry coefficient' , 
                                       'length of kernel groov', 'Class']))                       
#-----------------------------------------------------------------------------#
#--Step 2: Data selection process for data normalization----------------------#
#--Separating features and classes--------------------------------------------#
features = ['area','perimete','compactnes', 'length of kerne',
            'width of kernel','asymmetry coefficient' , 
            'length of kernel groov'] 
x = sd.loc[:, features].values #loc=for specific row/column access
y =  sd.loc[:,['Class']].values  
#-----------------------------------------------------------------------------#
#--Step 4: tSNE---------------------------------------------------------------#
tsne = TSNE(n_components=2, random_state=0)
st = tsne.fit_transform(x)
pDf = pd.DataFrame(data = st
             , columns = ['principal component 1', 'principal component 2'])
Df = pd.concat([pDf, sd[['Class']]], axis = 1)
#-----------------------------------------------------------------------------#
# Step 5: Visualize the data in 2-dimensional space---------------------------#
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component t-SNE', fontsize = 20)
Classfy = [1,2,3]
colors = ['r', 'g', 'b']
for Class, color in zip(Classfy,colors):
    ind = Df['Class'] == Class
    ax.scatter(Df.loc[ind, 'principal component 1']
               , Df.loc[ind, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(Classfy)
ax.grid()
#-----------------------------------------------------------------------------#