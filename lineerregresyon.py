
import numpy as np
import matplotlib.pyplot as plt  # To visualize
import pandas as pd  # To read data
from sklearn.linear_model import LinearRegression

data = pd.read_csv('fiyatlar.csv')  # veriyi alıyoruz
X = data.iloc[:, 0].values.reshape(-1, 1)  # Verideki değerleri numpy dizisine döndürüyoruz
Y = data.iloc[:, 2].values.reshape(-1, 1)  # -1, satırların boyutunun hesaplandığı, ancak 1 sütun olduğu anlamına gelir


linear_regressor = LinearRegression()  # model objemizi olusturduk
linear_regressor.fit(X, Y)  # lineer regresyon modelimizi eğittik
Y_pred = linear_regressor.predict(X)  # tahminler yaptık
print("----------------------------")
print(data)
print("--------------------------")
print(X)#eğittiğimiz x değerleri
print("--------------------------")
print(Y_pred)#tahminler
print("--------------------------")
print("Skor: " + str(linear_regressor.score(X,Y)))

plt.xlabel("Km ler")
plt.ylabel("Tahmini fiyatlar")
plt.scatter(X, Y)#x ve y değerlerine göre noktalarımızı grafiğe yerleştirdik.
plt.plot(X, Y_pred, color='red')
plt.show()