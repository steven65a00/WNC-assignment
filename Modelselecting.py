from sklearn.neural_network import MLPClassifier
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import KernelPCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
import numpy as np
import pandas as pd
import statistics
#---------------First time preprocessing-------------------------------------------------
print("-----------First time preprccessing-------------")
# First time preprocessing
# load data and replace "" "NA" "#DIV/0!" with nah object
df = pd.read_csv('pml-training.csv',na_values=["","NA","#DIV/0!"])
df_t = pd.read_csv('pml-testing.csv',na_values=["","NA","#DIV/0!"])

# extract header and label and fill nah with 0
header = list(df.columns)
header_t = list(df_t.columns)
df = df.fillna(0)
df = df.drop(labels=header[0],axis=1)
df_t = df_t.fillna(0)
df_t = df_t .drop(labels=header_t[0],axis=1)

# Replace "no" in new_window with 0 and "yes" with 1
cov2float = {"new_window":{"yes":1,"no":0},"classe":{"A":1,"B":2,"C":3,"D":4,"E":5}}
df = df.replace(cov2float)
df_t = df_t.replace(cov2float)

# Get label before dummy
label = df['classe']
df = df.drop(labels = ['classe'],axis = 1)

# make dummy variables for username and cvtd_timestamp
df = pd.get_dummies(df,columns=['user_name','cvtd_timestamp'])
df_t = pd.get_dummies(df_t,columns=['user_name','cvtd_timestamp'])

# put label back
df['classe'] = label


# shuffle the data
datas = df.values
datas_t = df_t.values
np.random.seed(100)
np.random.shuffle(datas)
np.random.shuffle(datas)

# split data and label
label = datas[:,-1]
datas = np.delete(datas,-1,axis=1)

# Modeling and check the CV error (kNN, Neural Network, SVM, Random Forest)

CV_kNN=[]
CV_kNN_Var=[]
CV_NN=[]
CV_NN_Var=[]
CV_SVM=[]
CV_SVM_Var=[]
CV_RF=[]
CV_RF_Var=[]
CV_num = 10


#Num_neibor =[1,3,21,49,101]
#Num_nodes_layer = [3,5,10]
#Num_degree = [3,5,10]
#Num_est=[1,2,5]

Num_neibor =[1,3]
Num_nodes_layer = [3,5,20]
Num_degree = [5,10]
Num_est=[1,2,5]


for i in Num_neibor:
    clf_kNN = KNeighborsClassifier(n_neighbors=i)
    CV_kNN.append(statistics.mean(cross_val_score(clf_kNN, datas, label, cv=10)))
    CV_kNN_Var.append(statistics.variance(cross_val_score(clf_kNN, datas, label, cv=10)))

for i in Num_nodes_layer:
    clf_NN =  MLPClassifier(hidden_layer_sizes=(i,)*2,learning_rate_init=0.001,max_iter=100000)
    CV_NN.append(statistics.mean(cross_val_score(clf_NN,datas, label, cv=10)))
    CV_NN_Var.append(statistics.variance(cross_val_score(clf_NN, datas, label, cv=10)))


for i in Num_degree:
    clf_SVM = SVC(kernel='poly',degree=i)
    CV_SVM.append(statistics.mean(cross_val_score(clf_SVM,datas, label, cv=10)))
    CV_SVM_Var.append(statistics.variance(cross_val_score(clf_SVM, datas, label, cv=10)))

for i in Num_est:
    clf_RF = RandomForestClassifier(n_estimators=i)
    CV_RF.append(statistics.mean(cross_val_score(clf_RF,datas, label, cv=10)))
    CV_RF_Var.append(statistics.variance(cross_val_score(clf_RF, datas, label, cv=10)))


print('kNN CV error:',CV_kNN)
print('kNN CV Variance:',CV_kNN_Var)
print('NN CV error:',CV_NN)
print('NN CV Variance:',CV_NN_Var)
print('SVM CV error:',CV_SVM)
print('SVM CV Variance:',CV_SVM_Var)
print('RF CV error:',CV_RF)
print('RF CV Variance:',CV_RF_Var)

#----------Sencod time preprccessing------------------------------------------------------
print("----------Sencod time preprccessing------------")
# since it does not perform well on the model other than Random Forest, try to reduce to do 2nd preprocessing
# reduce the feature dimensions by observation on raw dataset
df2 = pd.read_csv('pml-training.csv',na_values=["","NA","#DIV/0!"])
header = list(df2.columns)
df2 = df2.drop(labels=header[0:6],axis=1)
label2 = df2['classe']
df2 = df2.drop(labels=header[-1],axis=1)
datas = df2.values

# delete the features with Nah inside the column
del_col = []
for i in range(0,len(datas[0])):
    if np.isnan(datas[0][i]):
        del_col.append(i)
datas_red = np.delete(datas,del_col,axis=1) 

# normalize the data
ma = np.max(datas_red,axis=0)
mi = np.min(datas_red,axis=0)
# find out the col where ma-mi=0, otherwise it will cause the ambiguous point when normalization
del_col=[]
for i in range(0,len(ma)):
    if ((ma[i]-mi[i])==0):
        del_col.append(i)
ma = np.delete(ma,del_col)
mi = np.delete(mi,del_col)
datas_red= np.delete(datas_red,del_col,axis=1)
datas_red=(datas_red-mi)/(ma-mi)
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


# Feature extraction by kPCA
"""
# finding the best number of features to extract
Var_kPCA = []
i_index=[]

for i in range(10,15):
    kPCA = KernelPCA(n_components=i,kernel='poly',degree = 3).fit(datas_r1)
    kPCA_feature = kPCA.transform(datas_r1)+kPCA.transform(datas_r2)
    Var_kPCA.append(sum(np.var(kPCA_feature,axis=0)))
    i_index.append(i)
print([[i_index[i],Var_kPCA[i]] for i in range(0,len(i_index))])
"""
# find out # of components = 14 and transform the data into new feature space
kPCA = KernelPCA(n_components=14,kernel='poly',degree = 3).fit(datas_r1)
Ldim_datas = list(kPCA.transform(datas_r1))+list(kPCA.transform(datas_r2))


# Modeling and check the CV error (kNN, Neural Network, SVM, Random Forest)

CV_kNN = []
CV_kNN_Var = []
CV_NN = []
CV_NN_Var = []
CV_SVM = []
CV_SVM_Var = []
CV_RF = []
CV_RF_Var = []
CV_num = 10


Num_neibor =[1,3,5]
Num_nodes_layer = [3,5,40]
Num_degree = [5,10,15]
Num_est=[1,5,20]



for i in Num_neibor:
    clf_kNN = KNeighborsClassifier(n_neighbors=i)
    CV_kNN.append(statistics.mean(cross_val_score(clf_kNN, Ldim_datas, label2, cv=10)))
    CV_kNN_Var.append(statistics.variance(cross_val_score(clf_kNN, Ldim_datas, label2, cv=10)))

for i in Num_nodes_layer:
    clf_NN =  MLPClassifier(hidden_layer_sizes=(i,)*2,learning_rate_init=0.001,max_iter=100000)
    CV_NN.append(statistics.mean(cross_val_score(clf_NN,Ldim_datas, label2, cv=10)))
    CV_NN_Var.append(statistics.variance(cross_val_score(clf_NN, Ldim_datas, label2, cv=10)))

for i in Num_degree:
    clf_SVM = SVC(kernel='poly',degree=i)
    CV_SVM.append(statistics.mean(cross_val_score(clf_SVM, Ldim_datas, label2, cv=10)))
    CV_SVM_Var.append(statistics.variance(cross_val_score(clf_SVM, Ldim_datas, label2, cv=10)))

for i in Num_est:
    clf_RF = RandomForestClassifier(n_estimators=i)
    CV_RF.append(statistics.mean(cross_val_score(clf_RF,Ldim_datas, label2, cv=10)))
    CV_RF_Var.append(statistics.variance(cross_val_score(clf_RF, Ldim_datas, label2, cv=10)))

print('kNN CV error:',CV_kNN)
print('kNN CV Variance:',CV_kNN_Var)
print('NN CV error:',CV_NN)
print('NN CV Variance:',CV_NN_Var)
print('SVM CV error:',CV_SVM)
print('SVM CV Variance:',CV_SVM_Var)
print('RF CV error:', CV_RF)
print('RF CV Variance:', CV_RF_Var)
