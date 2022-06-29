from fileinput import filename
from flask import Flask
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

from flask import render_template,request,redirect,url_for,jsonify
app = Flask(__name__)  

@app.route('/')  
def upload():  
    return render_template('login.html') 
@app.route('/file_upload')  
def load():
    return render_template('file_upload.html')
@app.route('/about')  
def about():
    return render_template('about.html')
@app.route('/success', methods = ['POST'])  
def success():
    if request.method == 'POST':
        f = request.files['file']  
        f.filename ="dataset.csv"
        f.save(f.filename)  
        return render_template('success.html', name = f.filename)  
@app.route('/test')
def test():
    df= pd.read_csv("C:/Users/Pradeep_NG/Desktop/datasetAccuracyModel/dataset.csv")
    X=df.iloc[:,:-1]
    y=df.iloc[:,-1]
    df = df.fillna(df.mean())
    
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.2, random_state=0)
    X_train = X_train.fillna(X_train.mean())
    X_test = X_test.fillna(X_test.mean())

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    

    import xgboost as xgb
    from sklearn.metrics import accuracy_score
    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    xg=accuracy_score(y_test, preds)
    xg=xg*100
    kam=str(xg)
    
    from sklearn.tree import DecisionTreeClassifier

    cl = DecisionTreeClassifier(criterion = 'entropy', random_state = 0, max_depth = 5)
    model2 =cl.fit(X_train, y_train)
    y_pred = model2.predict(X_test)
    dt=accuracy_score(y_test, y_pred)
    print("Acurracy of DCT: ", dt)
    dt=dt*100
    kav=str(dt)

    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors = 8)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    result = accuracy_score(y_test,y_pred)
    print("Accuracy of KNN classifier:",result)
    result=result*100
    pra=str(result)


    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)
    from sklearn import metrics
    nb = metrics.accuracy_score(y_test, y_pred)
    print("Accuracy of Naive Bayes:", nb)
    nb=nb*100
    kab=str(nb)

    #random forest classifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    rfc = RandomForestClassifier(random_state=0)
    rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_test)
    rf = accuracy_score(y_test, y_pred)
    print("accuracy of Random forest classifier",rf)
    rf=rf*100
    deep=str(rf)

    #logistic regression 
    from sklearn.linear_model import LogisticRegression
    lc = LogisticRegression()
    lc.fit(X_train, y_train)
    y_pred = lc.predict(X_test)
    lcc = accuracy_score(y_test, y_pred)
    print("accuracy of Logistic regression : ", lcc)
    lcc = lcc*100
    ng=str(lcc)
    '''
    #Light Gradient Boosting Model
    import lightbgm as lgb
    clf = lgb.LGBMClassifier()
    clf.fit(X_test, y_test)
    y_pred = lgb.predict(X_test)
    lg = accuracy_score(y_test, y_pred)
    print("accuracy score of LGBM:", lg)
    lg = lg*100
    ach = str(lg)
    '''

    return render_template('test.html', test = kam, test1=kav, test2=pra, test3=kab, test4=deep, test5=ng)


if __name__ == "__main__":
     app.run(debug=True)

