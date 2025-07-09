# damageDetection
Derin öğrenme tabanlı bir VGG-16 transfer öğrenme modeli kullanarak araçlardaki çökme, çizik, far ve cam hasarlarını sekiz kategoride yüksek doğrulukla tespit eden ve web arayüzü üzerinden gerçek zamanlı analiz imkânı sunan bir sistem geliştirdim.
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


✅ İş Paketi 1: Proje Düzeni ve Veri Hazırlığı
◉ Açıklama: Ana klasör yapısı (dataset/images, dataset/labels.csv, model/, notebooks/, predict/, webapp/) oluşturuldu; labels.csv içindeki dosya adları .jpeg uzantısına göre güncellendi ve eksik resimler kontrol edilip tamamlandı.

✴️ Katkısı: Projenin her aşamasında tutarlı bir dizin yapısı sağlandı, veri bütünlüğü garanti altına alındı.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
✅ İş Paketi 2: Keşifsel Veri Analizi (EDA) ve Görselleştirme
◉ Açıklama: Pandas ve Matplotlib kullanılarak sınıf dağılımı, eğitim/doğrulama split oranları ve hasar kategorileri grafiklerle analiz edildi.

✴️ Katkısı: Veri dengesizlikleri ve örnek sayıları net olarak ortaya kondu; eğitim stratejisi bu bilgilere göre şekillendirildi.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
✅ İş Paketi 3: Model Mimarisi Tasarımı ve Transfer Öğrenme
◉ Açıklama: ImageNet ön-eğitimli VGG-16’nin konvolüsyon blokları include_top=False ile kullanıldı; GlobalAveragePooling2D, Dense(512, ReLU) ve Dropout(0.5) katmanları eklendi, Softmax çıkışla 8 sınıflı model derlendi.

✴️ Katkısı: Transfer öğrenme sayesinde sınırlı veriyle yüksek özellik çıkarımı, aşırı uyum riskinin minimize edilmesi.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
✅ İş Paketi 4: İnce Ayar (Fine-Tuning) ve Çoklu Etiketleme
◉ Açıklama: Son iki VGG bloğu serbest bırakılarak yeniden eğitime açıldı; çıkış katmanında sigmoid aktivasyonla çoklu hasar tespiti desteği eklendi; frame-frame video analizi ve tahmin hafızası hazırlandı.

✴️ Katkısı: Birden fazla hasar tipinin aynı anda tespiti, video analizi ile kararlılığın artması.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

✅ İş Paketi 5: Web Arayüzü Geliştirme (Flask + Bootstrap)
◉ Açıklama: app.py ve index.html ile dinamik resim/video yükleme, gerçek-zamanlı progress bar, sonuç kartı tasarlanıp renkli, mobil uyumlu Bootstrap arayüz oluşturuldu.

✴️ Katkısı: Kullanıcı dostu, etkileşimli ve platformdan bağımsız hasar tespit deneyimi sunuldu; hızlı prototipleme ve gelecek mobil entegrasyon için zemin hazırlandı.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
