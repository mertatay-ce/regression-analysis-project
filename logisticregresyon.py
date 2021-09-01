import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sn
import matplotlib.pyplot as plt

#veri setini excel dosyasından okuyoruz
dataset = pd.read_excel('dataset.xlsx',index_col=0)
#veri setini dataframe olacak şekilde pandas küt. yardımıyla tanımlıyoruz.
df = pd.DataFrame(dataset)
#Dataframe i gösteriyoruz
print (df)


#x elemanlarının yerine geçecek dataframede aşağıdaki belirtilen isimdeki sütunlardan verileri alıyoruz
X = df[['gmat', 'gpa','work_experience']]
#y elemanlarının yerine geçecek dataframede aşağıdaki belirtilen isimdeki sütunlardan verileri alıyoruz
y = df['admitted']
#eğitilecek x değerlerinden bir test veri seti elde ediyoruz. Belirtilen test_size = veri
#setinin çeyreğini kullanıyor.
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)

#lojistik regresyon modelini belirtiyoruz
logistic_regression= LogisticRegression()
#analize giricek modeli eğitiyoruz
logistic_regression.fit(X_train,y_train)
#yukarıdan alınan x elemanlarından oluşan test verisetinden eğitilen ve 
#tahmin edilen y değerlerini değişkene atıyoruz
y_pred=logistic_regression.predict(X_test)

print("------------------------")
print (X_test) #test veriseti
print("------------------------")
print (y_pred) #tahminsel değerler
print("Skor:" + str(logistic_regression.score(X_train,y_train)))# 0< h0(x) <1 durumunu sağlayan skor
print("------------------------")
print('Doğruluğu: ',metrics.accuracy_score(y_test, y_pred)) #Doğruluk skoru
print("------------------------")


#Burada sonuç matrisinde ne kadar değer neye eşit gösteriyoruz
karisiklik_matrisi= pd.crosstab(y_test, y_pred, rownames=['Gerçek'], colnames=['Tahmin'])
sn.heatmap(karisiklik_matrisi, annot=True)

plt.show()#Ekranda göster