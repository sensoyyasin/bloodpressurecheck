import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

metinler = [
    "Programlamayı seviyorum",
    "Hatalardan nefret ediyorum",
    "Kod yazmak eğlenceli",
    "Hata ayıklama sinir bozucu",
    "Bu proje harika",
    "Bu kod çalışmıyor",
    "Yazılım geliştirmek güzel",
    "Bugün hava güzel",
    "Bu yemek lezzetli",
    "Bu film çok kötü",
    "Müzik dinlemek keyifli",
    "Bu hatayı sevmiyorum",
    "Bu ürün çok iyi",
    "Bu uygulama berbat",
    "Programlamak fena değil",
    "Programlamak fena güzel",
    "Programlamak fena kötü",
    "Programlama kötü",
    "Programlama iyi"
]

etiketler = [
    "pozitif",
    "negatif",
    "pozitif",
    "negatif",
    "pozitif",
    "negatif",
    "pozitif",
    "pozitif",
    "pozitif",
    "negatif",
    "pozitif",
    "negatif",
    "pozitif",
    "negatif",
    "pozitif",
    "pozitif",
    "negatif",
    "negatif",
    "pozitif"
]

# Model boru hattı oluştur
model = make_pipeline(CountVectorizer(), MultinomialNB()) #metinleri kelime frekanslarına bağlı olarak vektörlere dönüştürür.
# MultinomialNB() verilen kelime frekans vektörlerini ve etiketleri kullanarak öğrenir.
# Model, belirli kelimelerin ve kelime kombinasyonlarının belirli duygu durumlarıyla ne sıklıkta ilişkilendirildiğini öğrenir.

# Modeli eğit
model.fit(metinler, etiketler)

def predict_func(metin):
    tahmin = model.predict([metin])
    return tahmin[0]

# Fonksiyonu test et
test_metin = "Programlama seviyorum"
tahmin_edilen_duygu = predict_func(test_metin)
print(f"'{test_metin}' metninin duygu durumu {tahmin_edilen_duygu}")

