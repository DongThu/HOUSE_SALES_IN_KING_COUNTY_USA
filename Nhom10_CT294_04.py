import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import array as arr
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor

X1 = np.array([3,3,2])
X2 = np.array([1,2.25,1])
X3 = np.array([1180,2570,770])
Y = np.array([221900,538000,180000])

def LR1(X1,X2,X3,Y,eta,lanlap, theta0, theta1, theta2, theta3):
    m = len(X1) #số phần tử
    for k in range(0,lanlap):
        print("Lan lap: ",k)
        for i in range(0,m):
            h_i = theta0 + theta1*X1[i] + theta2*X2[i] + theta3*X3[i]
            #theta0
            theta0 = theta0 + eta*(Y[i]-h_i)*1
            print ("Phan tu", i, "y=", Y[i], "h=",h_i,"gia tri theta0=", round(theta0,3))
            #theta1
            theta1 = theta1 + eta*(Y[i]-h_i)*X1[i]
            print ("Phan tu ",i, "gia tri theta1=", round(theta1,3))
            #theta2
            theta2 = theta2 + eta*(Y[i]-h_i)*X2[i]
            print ("Phan tu ",i, "gia tri theta2=", round(theta2,3))
            #theta3
            theta3 = theta3 + eta*(Y[i]-h_i)*X3[i]
            print ("Phan tu ",i, "gia tri theta3=", round(theta3,3))
    return [round(theta0,3), round(theta1,3), round(theta2,3), round(theta3,3)]

theta = LR1(X1,X2,X3,Y,0.0000001,1,0.0000003,0.0000004,0.0000001,0.0000002)
theta
theta2 =LR1(X1,X2,X3,Y,0.0000001,2,0.0000003,0.0000004,0.0000001,0.0000002)
theta2

#Dự báo cho phần tử mới tới
XX1 = [3,2,1]
XX2 = [2,1,1.5]
XX3 = [2500,1500,1000]
for i in range(3):
    YYY= theta[0] + theta[1]*XX1[i] + theta[2]*XX2[i] + theta[3]*XX3[i]
    print (round(YYY,3))


# Huấn luyện mô hình bằng giải thuật hồi quy tuyến tính
dt = pd.read_csv("kc_house_data.csv",index_col=0)
X=dt.iloc[:,3:20]
Y=dt.price
print(X)
#Vẽ đường hồi quy trên tập dữ liệu
#sns.regplot(x="sqft_living", y="price", data=dt, ci = None, color="blue")
#plt.show()
#Độ dài tập train và test
len(X_train)
len(X_test)

hoiquy = arr.array('d',[])
cay = arr.array('d',[])
rung = arr.array('d',[])

import sklearn
from sklearn.model_selection import train_test_split
from sklearn import linear_model
for i in range(0,10):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=1.0/3, random_state=50+i)
    # hoi quy tuyen tinh
    ln = linear_model.LinearRegression()
    ln.fit(X_train, y_train)
    y_pred_ln = ln.predict(X_test)
    err_ln = mean_squared_error(y_test, y_pred_ln)
    err_ln
    print("Sai so cua hoi quy tuyen tinh lan thu", i)
    round(np.sqrt(err_ln),3)
    hoiquy.append(round(np.sqrt(err_ln),3))
    #cay quyet dinh
    tree = DecisionTreeRegressor(random_state=0)
    bagging_regtree = BaggingRegressor(base_estimator=tree, n_estimators=10, random_state=50)
    bagging_regtree.fit(X_train, y_train)
    y_pred = bagging_regtree.predict(X_test)
    err_cay = mean_squared_error(y_test, y_pred)
    err_cay
    print("Sai so cua cay quyet dinh lan thu", i)
    round(np.sqrt(err_cay),3)
    cay.append(round(np.sqrt(err_cay),3))
    #Giải thuật rừng ngẫu nhiên để dự đoán giá nhà
    RForest = RandomForestRegressor(n_estimators=50, min_samples_leaf=25, max_depth=12,  random_state=50)
    RForest.fit(X_train, y_train)
    y_pred = RForest.predict(X_test)
    err_rung = mean_squared_error(y_test, y_pred)
    err_rung
    print("Sai so cua rung ngau nhien lan thu", i)
    round(np.sqrt(err_rung),3)
    rung.append(round(np.sqrt(err_rung),3))

Lanlap = np.array([1,2,3,4,5,6,7,8,9,10])
plt.bar(Lanlap - 0.2, hoiquy, color='red', width = 0.2, label = "Hoi quy tuyen tinh")
plt.bar(Lanlap, cay, color='blue', width = 0.2, label = "Cay quyet dinh")
plt.bar(Lanlap + 0.2, rung, color='green', width = 0.2, label = "Rung ngau nhien")
plt.legend(["Hoi quy tuyen tinh", "Cay quyet dinh", "Rung ngau nhien"], loc = "lower left")
plt.xlabel("Lần lặp")
plt.ylabel("Độ sai số tính theo RMSE")
plt.title("Biểu đồ thể hiện độ chênh lệch sai số giữa các giải thuật")
#plt.grid(True)
plt.show()
