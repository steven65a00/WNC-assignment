from sklearn.neural_network import MLPClassifier
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import KernelPCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
import numpy as np
import pandas as pd


#--------------First prediction by kNN-----------------------
print('----------First prediction by Random Forest-----------')
# First time preprocessing
# load data and replace "" "NA" "#DIV/0!" with nah object
df = pd.read_csv('pml-training.csv',na_values=["","NA","#DIV/0!"])
df_t = pd.read_csv('pml-testing.csv',na_values=["","NA","#DIV/0!"])
# extract header and label and fill nah with 0
header = list(df.columns)
header_t = list(df_t.columns)
df = df.fillna(0)
df = df.drop(labels=[header[0],'user_name','cvtd_timestamp'],axis=1)
df_t = df_t.fillna(0)
df_t = df_t .drop(labels=[header_t[0],'user_name','cvtd_timestamp'],axis=1)
# Replace "no" in new_window with 0 and "yes" with 1
cov2float = {"new_window":{"yes":1,"no":0},"classe":{"A":1,"B":2,"C":3,"D":4,"E":5}}
df = df.replace(cov2float)
df_t = df_t.replace(cov2float)

# drop the last column of testing data
df_t = df_t.drop(labels = ['problem_id'],axis = 1)
datas_t = df_t.values


# shuffle the data
datas = df.values
np.random.seed(100)
np.random.shuffle(datas)
np.random.shuffle(datas)

# split data and label
label = datas[:,-1]
datas = np.delete(datas,-1,axis=1)
clf_RF = RandomForestClassifier(n_estimators=5).fit(datas, label)
predict_class=[]
get_class = {1:"A",2:"B",3:"C",4:"D",5:"E"}
for x in clf_RF.predict(datas_t):
    predict_class.append(get_class[x])
print(predict_class)

#--------------Second prediction by kNN-----------------------
print('---------------Second prediction by kNN----------------')
# reduce the feature dimensions by observation on raw dataset
df2 = pd.read_csv('pml-training.csv',na_values=["","NA","#DIV/0!"])
df2_t = pd.read_csv('pml-testing.csv',na_values=["","NA","#DIV/0!"])
header = list(df2.columns)
header_t = list(df2_t.columns)
df2 = df2.drop(labels=header[0:6],axis=1)
df2_t = df2_t.drop(labels=header_t[0:6],axis=1)
label2 = df2['classe']
df2 = df2.drop(labels=header[-1],axis=1)
df2_t = df2_t.drop(labels=header_t[-1],axis=1)
datas = df2.values
datas_t = df2_t.values

# delete the features with Nah inside the column
del_col = []
for i in range(0,len(datas[0])):
    if np.isnan(datas[0][i]):
        del_col.append(i)
datas_red = np.delete(datas,del_col,axis=1)
datas_t = np.delete(datas_t,del_col,axis=1)

# normalize the data
ma = np.max(datas_red,axis=0)
mi = np.min(datas_red,axis=0)
ma_t = np.max(datas_t,axis=0)
mi_t = np.min(datas_t,axis=0)
# find out the col where ma-mi=0, otherwise it will cause the ambiguous point when normalization
del_col=[]
del_col_t=[]
for i in range(0,len(ma)):
    if ((ma[i]-mi[i])==0):
        del_col.append(i)
for i in range(0,len(ma_t)):
    if ((ma_t[i]-mi_t[i])==0):
        del_col_t.append(i)

ma = np.delete(ma,del_col)
mi = np.delete(mi,del_col)
ma_t = np.delete(ma_t,del_col_t)
mi_t = np.delete(mi_t,del_col_t)

datas_red= np.delete(datas_red,del_col,axis=1)
datas_t= np.delete(datas_t,del_col_t,axis=1)
datas_red=(datas_red-mi)/(ma-mi)
datas_t=(datas_t-mi_t)/(ma_t-mi_t)
label2 = np.reshape(np.array(label2),(len(np.array(label2)),1))
datas_red = np.append(datas_red,label2,axis=1)

# shuffle the data
np.random.seed(100)
np.random.shuffle(datas_red)
np.random.shuffle(datas_red)

# split data and label
datas_red = np.array(datas_red)
label2 = datas_red[:,-1:]
datas_red = datas_red[:,:-1]
label2 = np.reshape(label2,(len(label2),))

# dataset size is too large, causing memory crash in kPCA, split the dataset into 2
l = int(len(datas_red)/2)
datas_r1 =datas_red[0:l]
datas_r2 =datas_red[l:]

# find out # of components = 14 and transform the data into new feature space
kPCA = KernelPCA(n_components=14,kernel='poly',degree = 3).fit(datas_r1)
Ldim_datas = list(kPCA.transform(datas_r1))+list(kPCA.transform(datas_r2))
Ldim_datas_t = kPCA.transform(datas_t)

clf= KNeighborsClassifier(n_neighbors=3).fit(Ldim_datas,label2)
print(clf.predict(Ldim_datas_t))
