
import pandas as pd
import numpy as np
df=pd.read_csv("covid19.csv")
df.describe()
print(df.head())
del df['bloodoxy']
del df['Heartrate'] 
print(df.head())
train_data=df.sample(frac=0.8)

test_data=df.drop(train_data.index)

print(train_data.shape)
print(test_data.shape)
train_label=train_data.pop('temp')

test_label=test_data.pop('temp')
print(train_label.shape)
print(test_label.shape)

mf = [[['gaussmf',{'mean':np.mean(np.arange(72,80)),'sigma':np.std(np.arange(80,87))}],['gaussmf',{'mean':np.mean(np.arange(87,90)),'sigma':np.std(np.arange(90,92))}],['gaussmf',{'mean':np.mean(np.arange(95,99)),'sigma':np.std(np.arange(99,102))}]],
      [['gaussmf',{'mean':np.mean(np.arange(102,104)),'sigma':np.std(np.arange(104,106))}],['gaussmf',{'mean':np.mean(np.arange(30,40)),'sigma':np.std(np.arange(40,55))}],['gaussmf',{'mean':np.mean(np.arange(55,60)),'sigma':np.std(np.arange(60,100))}]]]
        
from membership import membershipfunction
mfc = membershipfunction.MemFuncs(mf)

import anfis

anf = anfis.ANFIS(train_data,train_label, mfc)

pred_train=anf.trainHybridJangOffLine(epochs=20)

train_label=np.reshape(train_label,[1,len(train_label)])
test_label=np.reshape(test_label,[1,len(test_label)])
print(train_label.shape)
print(test_label.shape)

error=np.mean((pred_train-train_label)**2)

print(error)
anf.plotErrors()
anf.plotResults()





