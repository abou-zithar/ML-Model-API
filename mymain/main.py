import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
# seaborn is a library such as matplotlib but better because it has a function called count plot which lets you know the number
# of features so that the model does not overfit or be biased
import seaborn as sn
from sklearn.preprocessing import LabelEncoder
import statsmodels.regression.linear_model as sm
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score ,confusion_matrix
import joblib
import re
from joblib import dump
import uvicorn ##ASGI
from fastapi import FastAPI,Path
from pydantic import BaseModel
from itertools import chain

url = "https://raw.githubusercontent.com/mohamedaminsharaf/api-1/main/PCsDataSet%20(1).csv"
data=pd.read_csv(url, encoding = 'latin1',error_bad_lines=False)
data.isna().sum()
sn.countplot(data=data,x="PC_Type")
data.drop("Name",inplace=True,axis=1)
X=data.iloc[:,:-1]
y=data.iloc[:,-1]
encoded_data=pd.get_dummies(data=X,drop_first=True)
se=LabelEncoder()
target=se.fit_transform(data.iloc[:,-1].values)
y=target.reshape(-1,1)

columns=encoded_data.columns[:-1]

def backwardElimination(X,y,sl,num_var,columns):
    
    for i in range(num_var):
        regressor=sm.OLS(y,X).fit()
        max_p_value =max(regressor.pvalues)
        if max_p_value>sl:
            for j in range(0,num_var-i):
                if regressor.pvalues[j]==max_p_value:
                    X=np.delete(X,j,1)
                    columns=np.delete(columns,j)
    
        
            
    return X,columns

selecet_features,selected_columns=backwardElimination(encoded_data.iloc[:,:-1].values,encoded_data.iloc[:,-1].values,0.05,encoded_data.shape[1]-1,columns)

datafeatured=pd.DataFrame(data=selecet_features,columns=selected_columns)

datafeatured['target']=y
X=datafeatured.iloc[:,:-1].values
y=datafeatured.iloc[:,-1].values

trainx,testx,trainy,testy=train_test_split(X,y)
model=DecisionTreeClassifier()
model.fit(trainx,trainy)
ypridect=model.predict(testx)
print("The Decision Tree Classifier Accuracy Score is:")
print(accuracy_score(testy,ypridect))
print(confusion_matrix(testy,ypridect))
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred = gnb.fit(trainx,trainy).predict(testx)
print("The Gaussian Classifier Accuracy Score is:")
print(accuracy_score(testy,y_pred))
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
randFor = RandomForestClassifier()
y_predic = randFor.fit(trainx,trainy).predict(testx)
print("The Random Forest Classifier Accuracy Score is:")
print(accuracy_score(testy,y_predic))



dump(model, 'classifier.joblib')
dump(gnb,'gnb.joblib')
dump(randFor,'randFor.joblib')
def final_row(m,Cpu_Type,Cpu_core,Cpu_gen,Clockable,GPU,Gpu_gen,Gpu_type,price,PC_Type = "x"):
    prediction = []
    pc_type=""
    a = data[data['Price']<=price]
    if PC_Type=='gaming':
        b= a[a['PC_Type'] == 'gaming']
        b = b.iloc[1,:-1]
        return b
    elif PC_Type == 'business':
        b = a[a['PC_Type'] == 'business']
        b=b.iloc[1,:-1]
        return b
    elif PC_Type == 'General':
        b = a[a['PC_Type'] == 'General']
        b = b.iloc[1,:-1]
        return b
    if len(a)>4:
        if Cpu_Type in a.Cpu_Type.values:
            a = a[a['Cpu_Type']==Cpu_Type]
        else:
            a=a
    if len(a)>4:
        if Cpu_core in a.Cpu_core.values:
            a=a[a['Cpu_core']==Cpu_core]
        else:
            a=a
    if len(a)>4:
        if Cpu_gen in a.Cpu_gen.values:
            a= a[a['Cpu_gen'] == Cpu_gen]
        else:
            a=a
    if len(a)>4:
        if Clockable in a.Clockable.values:
            a = a[a['Clockable'] == Clockable]
        else:
            a=a
    if len(a)>4:
        if GPU in a.GPU.values:
            a=a[a['GPU']==GPU]
        else:
            a=a
    if len(a)>4:
        if Gpu_gen in a.Gpu_gen.values:
            a=a[a['Gpu_gen']==Gpu_gen]
        else:
            a=a
    if len(a)>4:
        if Gpu_type in a.Gpu_type.values:
            a=a[a['Gpu_type']==Gpu_type]
        else:
            a=a
    
    else:
        for i in range(3):
            r = a.index[i]
            a = a.iloc[:,:-1]
            prediction.append(m.predict(datafeatured.iloc[r,:-1].values.reshape(1,-1)))
            if prediction[i] == 2:
                pc_type = 'gaming'
                a = list(chain.from_iterable(a.values))
                a.insert(((i*23)+23),pc_type)
                a = ' '.join([str(s) for s in a])
                a = a.replace(u'\xa0', u'')
                return {a}
            elif prediction[i] == 1:
                pc_type = 'business'
                a = list(chain.from_iterable(a.values))
                a.insert(((i*23)+23),pc_type)
                a = ' '.join([str(s) for s in a])
                a = a.replace(u'\xa0', u'')
                return {a}
            else:
                pc_type = 'general'
                a = list(chain.from_iterable(a.values))
                a.insert(((i*23)+23),pc_type)
                a = ' '.join([str(s) for s in a])
                a = a.replace(u'\xa0', u'')
                return {a}
app = FastAPI()
m = joblib.load('classifier.joblib')
gnb = joblib.load('gnb.joblib')
ran = joblib.load('randFor.joblib')

# 4. Route with a single parameter, returns the parameter within a message
#    Located at: http://127.0.0.1:8000/AnyNameHere
#/{Cpu_Type}/{Cpu_core}/{Cpu_gen}/{Clockable}/{GPU}/{Gpu_gen}/{Gpu_type}
@app.get('/get_model/')
async def get_model(Cpu_Type : str , Cpu_core : int , Cpu_gen : str, Clockable : str, GPU : str,Gpu_gen : str,Gpu_type:str,price : int):
    response = final_row(ran,Cpu_Type,Cpu_core,Cpu_gen,Clockable,GPU,Gpu_gen,Gpu_type,price)
    return response

# def get_name(Cpu_Type : str , Cpu_core : int , Cpu_gen : str, Clockable : str, GPU : str,Gpu_gen : str,Gpu_type:str):



# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)