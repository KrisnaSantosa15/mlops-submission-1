# Submission 1: Obesity Prediction Machine Learning
Nama: Krisna Santosa

Username dicoding: `krisna_santosa`

| | Deskripsi |
| ----------- | ----------- |
| Dataset | [Obesity Prediction Dataset](kaggle.com/datasets/ruchikakumbhar/obesity-prediction/data) |
| Masalah | Obesitas adalah masalah kesehatan global yang serius. Deteksi dini dan prediksi risiko obesitas dapat membantu dalam pencegahan dan intervensi medis yang tepat waktu. |
| Solusi machine learning | Mengembangkan model klasifikasi yang dapat memprediksi tingkat obesitas seseorang berdasarkan berbagai faktor seperti pola makan, aktivitas fisik, dan karakteristik personal. Target akurasi minimal 85%. |
| Metode pengolahan | Data cleaning untuk menangani missing values dan outliers, Feature encoding untuk variabel kategorikal, Feature scaling menggunakan standardization, Feature selection untuk memilih fitur yang paling berpengaruh.  |
| Arsitektur model | Deep Neural Network dengan: Input layer sesuai jumlah fitur, 3 hidden layers (128, 64, 32 nodes) dengan aktivasi ReLU, Dropout layers (0.3) untuk mencegah overfitting, Output layer dengan aktivasi softmax untuk klasifikasi multi-class |
| Metrik evaluasi | Accuracy: mengukur ketepatan prediksi secara keseluruhan, Precision: mengukur ketepatan prediksi positif, Recall: mengukur kemampuan model mendeteksi kasus obesitas, F1-Score: rata-rata harmonik precision dan recall |
| Performa model | Target performa: Accuracy: >85%, Precision: >80%, Recall: >80%, F1-Score: >80% |
