import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle as pkl

df=pd.read_csv('train.csv')
df = df.iloc[:,1:]
df.info()

le = LabelEncoder()
df.iloc[:,-1] = le.fit_transform(df.iloc[:,-1])
X_train,X_test,Y_train,Y_test = train_test_split(df.iloc[:,:-1],df.iloc[:,-1],test_size=.3,random_state=44,stratify=df.iloc[:,-1])

svc = SVC(kernel='linear').fit(X_train,Y_train)
with open('model.pkl','wb') as f:
     pkl.dump(svc,f)