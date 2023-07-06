import pandas as pd
import pickle

from sklearn.model_selection import train_test_split

iris_data=pd.read_excel("iris .xls")
x=iris_data.drop('Classification',axis=1)
y=iris_data['Classification']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)

from sklearn.svm import SVC

svmclf=SVC(kernel='linear')
svmclf.fit(x_train,y_train)

pickle.dump(svmclf,open('model.pkl','wb'))

