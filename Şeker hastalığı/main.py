import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv('diabetes.csv')
data.head()

seker_hastasi = data[data.Outcome == 1]
saglikli_insan = data[data.Outcome == 0]

plt.scatter(saglikli_insan.Age, saglikli_insan.Glucose, label="Sağlıklı", color="Green", alpha=0.4)
plt.scatter(seker_hastasi.Age, seker_hastasi.Glucose, label="Şeker Hastası", color='red', alpha=0.4)
plt.title("Şeker hastalığı")
plt.xlabel("Yaş Değeri")
plt.ylabel("Glikoz Miktarı")
plt.legend()
plt.show()

y = data.Outcome.values
x_ham_veri = data.drop(["Outcome"], axis=1)

x = (x_ham_veri - np.min(x_ham_veri))/(np.max(x_ham_veri)-np.min(x_ham_veri))

print("Normalization Verisi")
print(x.head())

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.01,random_state=1)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)
prediction = knn.predict(x_test)
print("K = 3 için test verilerinin sonucu", knn.score(x_test,y_test))


sayac = 1

for k in range(1,11):
    knn_yeni = KNeighborsClassifier(n_neighbors=k)
    knn_yeni.fit(x_train, y_train)
    print(sayac," ", "Doğruluk oranı %",knn_yeni.score(x_test,y_test)*100)
    sayac+=1