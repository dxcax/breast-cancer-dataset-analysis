import joblib
import numpy as np

# Modeli yükle
model = joblib.load("kanser_model.pkl")  # Model dosyanın adı ve yolu doğru olmalı

# Kullanıcının girmesi gereken özellikler

feature_names_tr_en = [
    'Yarıçap Ortalama (radius_mean)',
    'Çevre Ortalama (perimeter_mean)',
    'Alan Ortalama (area_mean)',
    'Kompaktlık Ortalama (compactness_mean)',
    'Çukurlaşma Ortalama (concavity_mean)',
    'Çukur Noktaları Ortalama (concave points_mean)',
    'Yarıçap Standart Hata (radius_se)',
    'Alan Standart Hata (area_se)',
    'Yarıçap En Kötü (radius_worst)',
    'Çevre En Kötü (perimeter_worst)',
    'Kompaktlık En Kötü (compactness_worst)'
]

print("Lütfen aşağıdaki özellikleri sırayla giriniz:")

inputs = []
for feature in feature_names_tr_en:
    val = float(input(f"{feature}: "))
    inputs.append(val)

inputs_np = np.array(inputs).reshape(1, -1)

prediction = model.predict(inputs_np)[0]

if prediction == 1:
    print("\nTahmin: Kötü huylu (Malignant) tümör")
else:
    print("\nTahmin: İyi huylu (Benign) tümör")


