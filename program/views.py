from django.shortcuts import render
from sklearn.metrics.classification import accuracy_score

import numpy as np  
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier  # KNN algoritması import edildi
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report # Karmaşıklı Matrisi ve Doğruluk oranının
from sklearn.model_selection import train_test_split                                # tespiti için gerekli kütüphaneler import edildi

def index(request):
     return render(request, 'program/index.html')

def hesapla(request): 
     # Kullanılacak CSV dosyası okundu ve içindeki bazı datalar temizlendi. 
     data = pd.read_csv(r"C:\Users\Samet\Desktop\Projects\Odev\OdevDjango\odev\program\data\heart.csv")
     data.drop_duplicates(inplace=True)
     data.reset_index(drop=True, inplace=True)
     data = data.drop(['oldpeak'], axis=1).drop(['slp'], axis=1).drop(['caa'], axis=1)

     X = data.drop(['output'], axis=1)
     y = data[['output']]

     # Data %20 test ve %80 train olarak ayrıldı
     X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 42)

     # Knn algoritmasında komşu sayısı 8 olacak şekilde kullanıldı ve eğitildi
     knn = KNeighborsClassifier(n_neighbors=8)
     model = knn.fit(X_train, y_train)

     # Eğitilmiş model ile tahmin yapıldı.
     test_y_pred = model.predict(X_test);

     # Bu tahmine göre doğruluk yüzdesi ve confusion matrix hesaplandı
     tn, fp, fn, tp = confusion_matrix(y_test, test_y_pred).ravel()
     acc = "{:.2f}".format(accuracy_score(y_test, test_y_pred))
     
     report = classification_report(y_test, test_y_pred, output_dict=True)

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

     # Bu datalar kullanılabilmesi için bir array'e atandı
     test_array = np.array([[yas, cinsiyet, chest_pain_type, blood_pressure, cholestoral, blood_sugar,
                             EKG, maxx, angina, thallium_result]])

     # Yukarıda eğitilmiş modele kullanıcının giridiği datalar verildi ve yeni bir tahmin alındı. 
     y_pred = model.predict(test_array)     

     # Bu sonuc Frontend kısmına gönderildi
     if (y_pred == [0]):
          sonuc = "DÜŞÜK"
          context = {'sonuc': sonuc, 'acc': acc, 'sifir': sifir, 'sifir_f1': sifir_f1, 'bir': bir, 'bir_f1': bir_f1, 'tn': tn, 'tp': tp, 'fn': fn, 'fp': fp}
     elif (y_pred == [1]):
          sonuc = "YÜKSEK"
          context = {'sonuc': sonuc, 'acc': acc, 'sifir': sifir, 'sifir_f1': sifir_f1, 'bir': bir, 'bir_f1': bir_f1, 'tn': tn, 'tp': tp, 'fn': fn, 'fp': fp}   

     return render(request, 'program/index.html', context)