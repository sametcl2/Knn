from django.shortcuts import render

# Matrix işlemleri için Numpy, CSV dosyasından datayı okumak için Pandas import edildi
import numpy as np  
import pandas as pd
# Gaussian Naive Bayes algoritması import edildi
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
# Karmaşıklık Matrisi ve Doğruluk oranının tespiti için gerekli kütüphaneler import edildi
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report 
from sklearn.model_selection import train_test_split      

# Data pandas ile okundu
train_df = pd.read_csv(r"C:\Users\Samet\Desktop\Projects\Odev\OdevDjango\odev\program\data\heart.csv")
# Kullanılmayacak kısımlar datadan çıkarıldı
train_df = train_df.drop(['oldpeak'], axis=1).drop(['slp'], axis=1).drop(['caa'], axis=1)

# X değişkenine datalar atandı, output kusmı çıkarıldı
x = train_df.drop(['output'], axis=1)
# Y değişkenine datanın sonuç kısmı atanadı
y = train_df[['output']]

# Data %20 test %80 train olarak bölündü
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 42)

# Standtard Scaler kullanılarak datalar 0 ve 1 arası bir değere dönüştürüldü
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# KNN algoritması ile model eğitirldi
knn = KNeighborsClassifier()
model = knn.fit(x_train, y_train)

# Eğitilmiş model ile tahmin yapıldı
test_y_pred = model.predict(x_test);
# Bu tahmine göre doğruluk yüzdesi, confusion matrix ve f-1 score, recall vb. hesaplandı
tn, fp, fn, tp = confusion_matrix(y_test, test_y_pred).ravel()
acc = "{:.2f}".format(accuracy_score(y_test, test_y_pred))
report = classification_report(y_test, test_y_pred, output_dict=True)

print(x_test)
def index(request):
     return render(request, 'program/index.html')

def hesapla(request):      
     sifir = report.get('0')
     sifir_f1 = sifir.get('f1-score')
     bir = report.get('1')
     bir_f1 = bir.get('f1-score')

     # Frontend kısmında girilen datalar çekildi
     cinsiyet = float(request.POST.get('cinsiyet'))
     angina = float(request.POST.get('angina'))
     yas = float(request.POST.get('yas'))  
     thallium_result = float(request.POST.get('thallium-result'))
     chest_pain_type = float(request.POST.get('chest-pain-type'))
     blood_pressure = float(request.POST.get('blood-pressure'))
     cholestoral = float(request.POST.get('cholestoral'))
     blood_sugar = float(request.POST.get('blood-sugar'))
     EKG = float(request.POST.get('EKG'))
     maxx = float(request.POST.get('max'))  

     # Bu dataların kullanılabilmesi için array'e atandı
     test_array = np.array([[yas, cinsiyet, chest_pain_type, blood_pressure, cholestoral, blood_sugar,
                             EKG, maxx, angina, thallium_result]])
     # Giriş değerleriyle aynı olması için Standart Scaler ile 0 ile 1 arası değerlere sıkıştırdıldı
     test_array_n = sc.transform(test_array)

     # Yukarıda eğitilmiş modele kullanıcının giridiği datalar verildi ve yeni bir tahmin alındı. 
     y_pred = model.predict(test_array_n)     

     # Bu sonuc frontend kısmına gönderildi
     if (y_pred == [0]):
          sonuc = "DÜŞÜK"
          context = {
               'sonuc': sonuc, 'acc': acc, 'sifir': sifir, 'sifir_f1': sifir_f1, 'bir': bir, 'bir_f1': bir_f1, 'tn': tn, 'tp': tp, 'fn': fn, 'fp': fp
          }
     elif (y_pred == [1]):
          sonuc = "YÜKSEK"
          context = {
               'sonuc': sonuc, 'acc': acc, 'sifir': sifir, 'sifir_f1': sifir_f1, 'bir': bir, 'bir_f1': bir_f1, 'tn': tn, 'tp': tp, 'fn': fn, 'fp': fp
          }   

     return render(request, 'program/index.html', context)