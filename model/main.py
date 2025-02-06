import pandas as pd
from sklearn.metrics import accuracy_score,classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# import pickle5 as 
import pickle

### --> About pickle :
# It allows you to convert complex Python objects 
# (like lists, dictionaries, classes, etc.) into a byte stream,
#  which can be stored in a file or sent over a network. 
# This is useful for saving the state of an object and reloading it later.

# Machine Learning Models: It's commonly used to save and load
#  machine learning models, making it easier to reuse trained models
#  without having to retrain them each time.

def create_model(data):
    
    # Define X and y
    X = data.drop(['diagnosis'],axis=1)
    y = data['diagnosis']

    # Use StandardScaler to Normalize our data
    sc = StandardScaler()
    X = sc.fit_transform(X)
    

    # Split data
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    # Train
    model = LogisticRegression()
    model.fit(X_train,y_train)

    # Test our model 
    y_pred = model.predict(X_test)
    print(f"Accuracy of our model is {accuracy_score(y_test,y_pred)*100:.2f}%")
    print("Classification report : \n", classification_report(y_test,y_pred))
    print(f"The required columns : {data.columns}")
    return model,sc,y_pred


def clean_data():
    data = pd.read_csv("Data/data.csv")
    data = data.drop(["Unnamed: 32","id"],axis=1)
    data['diagnosis'] = data['diagnosis'].map({"M" : 1,"B" : 0})
    # print(data.head())
    return data


def main():
    data =  clean_data()
    
    model,sc,y_pred= create_model(data)

    with open('model/model.pkl','wb') as f:
      pickle.dump(model,f)
    
    with open('model/scaler.pkl','wb') as f:
        pickle.dump(sc,f)

  

if __name__ == '__main__':
    main()