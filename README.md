# Kalp Krizi Veri Seti Analizi ve Makine Öğrenimi

Bu projenin temel amacı, kalp krizi riskini tahmin edebilen bir
makine öğrenimi modeli geliştirmektir. 

Bu kapsamda, Kaggle’da bulunan "Heart Attack Analysis & Prediction Dataset" veri seti kullanılarak, 
bireylerin çeşitli sağlık parametreleri analiz edilecek ve bu parametrelerin kalp krizi riski üzerindeki etkisi değerlendirilecektir.

# Proje Aşamaları

1 - Veri keşfi ve Ön işleme - 3 adımdan oluşur bunlar: Veri Setinin anlaşılması, Veri Temizleme, Veri dönüştürmedir.

2 - Veri Analizi ve Görselleştirme - 2 Aşamadan oluşur Keşifsel Veri Analizi (EDA) ve Korelasyon Analizi

3 - Aykırı Değerleri Bulma ve Veri setini temizleme

4 - Model Geliştirme - 3 Aşamadan oluşur bunlar: Model Seçimi , Model Eğitimi ve Hiperparametre Optimizasyonu

5 - Model Değerlendirme ve Test - 3 Aşamadan oluşur : Model Performans Ölçüleri, Çapraz Doğrulama ve Son model seçimi

6 - Sonuçlar - Modelin uygulanabilirliği kontrol edilir. Model eğer sağlık alanında uygulanabilirse seçilir.

<div align="center">
<img width="133" alt="path" src="https://github.com/sensoyyasin/heartdisease_prediction/assets/73845925/e7a9874e-de57-44ee-aeaf-8db913297e9b">
</div>

# Veri Seti Hakkında Bilgi

Veri Setimizde toplam 13 farklı değişken var. Bunlar: 

1 - age - yaş (yıl cinsinden)

2 - sex - cinsiyet (1 = erkek; 0 = kadın)

3- cp - göğüs ağrısı tipi (1 = tipik anjina; 2 = atipik anjina; 3 = anjin
dışı ağrı; 0 = semptomsuz)

4 - trestbps - dinlenme kan basıncı (hastaneye kabulde mm Hg
cinsinden)

5 - chol - serum kolesterolü (mg/dl cinsinden)

6 - fbs - açlıkkan şekeri > 120 mg/dl (1 = doğru; 0 = yanlış)

7 - restecg - dinlenme elektrokardiyografik sonuçlar (1 = normal; 2
= ST-T dalgası anormalliği; 0 = hipertrofi)

8 - thalach - ulaşılan maksimum kalp atış hızı

9 - exang - egzersize bağlı anjina (1 = evet; 0 = hayır)

10 - oldpeak - dinlenmeye göre egzersizle indüklenen ST
depresyonu

11 - slope - zirve egzersiz ST segmentinin eğimi (2 = yukarı doğru
eğimli; 1 = düz; 0 = aşağı doğru eğimli)

12 - ca - florosopi ile renklendirilen ana damar sayısı (0-3)

13 - thal - 2 = normal; 1 = sabit kusur; 3 = düzeltilebilir kusur

14 - num - tahmin edilen özellik - kalp hastalığı teşhisi (anjyografik
hastalık durumu) (Değer 0 = < çap daralması; Değer 1 = > %50 çap
daralması)

# Model Karşılaştırılması

<img width="796" alt="Ekran Resmi 2024-06-09 16 33 04" src="https://github.com/sensoyyasin/heartdisease_prediction/assets/73845925/9d6d5f1a-bc3a-48c9-8b16-240932e8f080">

# Proje Sonucu

Logistic Regresyon, Karar Ağaçları, Destek Vektör Makineleri ve Rastgele Orman algoritmaları kullanılarak modeller oluşturulur ve doğruluk skorları hesaplanır. Çapraz doğrulama (cross-validation) kullanılarak modellerin performansını değerlendirdik.
Proje Kapsamında Gerçekleştirdiğimiz faaliyetler listesi:

1. Proje kapsamında öncelikle veri setini Keşifsel Veri Analizi
(EDA) için hazır hale getirdik.
2. Keşifsel Veri Analizi (EDA) gerçekleştirdik. Exploratory Data Analysis.
3. Tek değişkenli analiz kapsamında sayısal ve kategorik
değişkenleri Distplot ve yuvarlak Grafiklerle analiz ettik.
4. İki değişkenli analiz kapsamında değişkenleri birbiri arasında
FacetGrid, Sayım Grafiği, Çift Grafiği, Sürücü Grafiği, Kutu
Grafiği ve Isı Haritası grafikleri kullanarak analiz ettik.
5. Veri setini model için hazır hale getirdik. Bu bağlamda, eksik ve aykırı değerlerle mücadele ettik.
6. Model aşamasında dört farklı algoritma kullandık.
7. Lojistik Regresyon modeli ile %87 doğruluk ve %88 AUC elde ettik.
8. Karar Ağacı Modeli ile %83 doğruluk ve %85 AUC elde ettik.
9. Destek Vektör Sınıflandırıcı Modeli ile %83 doğruluk ve %89 AUC elde ettik.
10. Ve Rastgele Orman Sınıflandırıcı Modeli ile %90.3 doğruluk ve %93 AUC elde ettik..
11. Tüm bu model çıktıları değerlendirildiğinde, en iyi sonuçları veren Rastgele Orman Algoritması ile oluşturduğumuz modeli tercih ettik.

