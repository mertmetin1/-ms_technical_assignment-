import pandas as pd 

# import data
data = pd.read_csv('dataset.csv')

# check load data is ok
print(data.head())
print()
""" 
   feature_1  feature_2  feature_3  feature_4  isVirus
0  -0.233467   0.308799   2.484015   1.732721    False
1   1.519003   1.238482   3.344450   0.783744    False
2   0.400640   1.916748   3.291096  -0.918519    False
3  -1.616474   0.209703   1.461544  -0.291837    False
4   1.480515   5.299829   2.640670   1.867559     True
"""

#dataset info 
print(data.info())
print()
"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1999 entries, 0 to 1998
Data columns (total 5 columns):
 #   Column     Non-Null Count  Dtype  
---  ------     --------------  -----  
 0   feature_1  1897 non-null   float64
 1   feature_2  1899 non-null   float64
 2   feature_3  1893 non-null   float64
 3   feature_4  1897 non-null   float64
 4   isVirus    1999 non-null   bool   
dtypes: bool(1), float64(4)
memory usage: 64.5 KB
"""


#descritoion
print(data.describe())
print()

"""
         feature_1    feature_2    feature_3    feature_4
count  1897.000000  1899.000000  1893.000000  1897.000000
mean      0.814404     1.795843     2.621096     0.807499
std       1.729538     1.605611     1.474973     1.768597
min      -2.285499    -7.363119    -5.363119    -3.006499
25%      -0.556433     0.975148     1.671905    -0.563357
50%       0.084789     1.881904     2.499623     0.021857
75%       2.270955     2.840511     3.470200     2.319822
max       5.929096     7.549658     9.549658     5.759355
"""
# check missed data 
print(data.isnull().sum())
print()
"""
feature_1    102
feature_2    100
feature_3    106
feature_4    102
isVirus        0
dtype: int64
"""

import matplotlib.pyplot as plt #virtulize lib
import seaborn as sns           #virutalize lib 

# virtulize to check raw data 
# feature pair plot.1
sns.pairplot(data, hue='isVirus')
plt.show()



# replace missed data with mean
data = data.fillna(data.mean())

# Check handled Missed data 
print(data.isnull().sum())
print()

"""
feature_1    0
feature_2    0
feature_3    0
feature_4    0
isVirus      0
dtype: int64
"""

# Corelation Matrix
# Coleration matrix plot.2
correlation_matrix = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Corelation Matrix")
plt.show()

#check after handle missed data 
#feature pair (handled missed value) plot.3 
sns.pairplot(data, hue='isVirus')
plt.show()

# Get Cols
X = data.iloc[:, :-1]
y = data['isVirus']

from sklearn.model_selection import train_test_split

# Split  Train,Test data Set 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier

#create ML model
model_rf = RandomForestClassifier(random_state=42)

# model traingn
model_rf.fit(X_train, y_train)

# Prediction
y_pred_rf = model_rf.predict(X_test)


from sklearn.metrics import classification_report, confusion_matrix

# prediction Report
print("Random Forest Classifier:")
print(classification_report(y_test, y_pred_rf))
print()
"""
Random Forest Classifier:
                precision    recall  f1-score   support

       False       0.89      0.93      0.91       263
        True       0.85      0.79      0.82       137

    accuracy                           0.88       400
   macro avg       0.87      0.86      0.86       400
weighted avg       0.88      0.88      0.88       400

"""

# Confusion Matrix plot.4
cm = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', cbar=False, annot_kws={"size": 15})
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

from sklearn.metrics import accuracy_score

# Random Forest Classifier Accuracty
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Random Forest Classifier Accuracy:", accuracy_rf)
print()

"""
Random Forest Classifier Accuracy: 0.88
"""

""" --Turkish tutorial for calculate metrics of analysis- 
        | Tahmin Edilen Pozitif  | Tahmin Edilen Negatif
---------------------------------------------------------------------------
| Gerçek Pozitif   |              TP                 |                 FN   |
| Gerçek Negatif   |              FP                 |                 TN   |
---------------------------------------------------------------------------

# Doğruluk (Accuracy)
accuracy = (TP + TN) / (TP + TN + FP + FN)

# Hassasiyet (Precision)
precision = TP / (TP + FP)

# Gerçek Pozitif Oranı (Recall)
recall = TP / (TP + FN)

# F1 Skoru
f1_score = 2 * precision * recall / (precision + recall)

"""

from sklearn.metrics import roc_curve, roc_auc_score

# ROC curl and AUC score calculate 
y_prob_rf = model_rf.predict_proba(X_test)[:, 1]  
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, y_prob_rf)  #calculate ThreshH Val of FPR, TPR for ROC 
auc_rf = roc_auc_score(y_test, y_prob_rf)  # area of under ROC curl  (AUC)

# ROC curl plot.5
plt.figure(figsize=(8, 6))
plt.plot(fpr_rf, tpr_rf, label='Random Forest (AUC = {:.2f})'.format(auc_rf))
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

print("AUC Score for Random Forest Classifier:", auc_rf)
"""
AUC Score for Random Forest Classifier: 0.9429796564069829
"""




"""
Conclusion 

When looking at the correlation matrix, I observed a strong positive correlation (0.75) between f1 and f4, and a negative correlation (-0.43) between f1 and the isVirus feature.
Missing values were identified in the dataset and filled using column averages.
I observed an increase in polarization between f1 and other features in the graphs.
A random forest model was used for the classification model.
The train and test data sets were split with a ratio of 0.35.
I examined the positive true/false and negative true/false rates using the confusion matrix.
I observed an AUC (Area Under Curve) of 0.94 on the ROC curve, indicating that the model performs better than random guessing.
The model's accuracy was observed to be 0.85 according to the test results.

Sonuç
Korelasyon matrisine baktığınızda, f1 ile f4 arasında (0.75) güçlü pozitif korelasyon  olduğunu 
ve f1 ile isVirus özellikleri arasında (-0.43)  negatif korelasyon olduğunu gözlemlendim
Veri setinizde eksik değerler olduğu belirlendi ve bunlar sütun ortalaması ile dolduruldu 
f1 ve diğer özellikler arasındaki grafiklerde kutuplaşmanın arrtığını gözlemledim 
Sınıflandırma modeli için random forest modeli kullanıldı 
Train ve Test veri setleri oran 0.35 olarak belirlendi
konfüzyon matrisi ile pozitif true,false negative true,false oranlarını inceledim 
ROC eğrisi altında kalan alanın (AUC) 0.94 olduğunu gözlemledim 
Eğri köşegenin üzerinde  model rastgele tahmin yapmaktan daha iyi performans gösterdiğini gözlemledim.
Test sonuçlarına Göre modelin kesinliğini 0.85 olarak gözlemledim 


"""
