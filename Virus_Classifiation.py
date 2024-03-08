import pandas as pd 

# import data
data = pd.read_csv('dataset.csv')

# check load data is ok
print(data.head())
print()

#dataset info 
print(data.info())
print()
#check data types
print(data.dtypes())
print()
#descritoion
print(data.describe())
print()
# check missed data 
print(data.isnull().sum())
print()

import matplotlib.pyplot as plt #virtulize lib
import seaborn as sns           #virutalize lib 

# virtulize to check raw data 
sns.pairplot(data, hue='isVirus')
plt.show()



# replace missed data with mean
data = data.fillna(data.mean())

# Check Missed data 
print(data.isnull().sum())
print()

# Corelation Matrix
correlation_matrix = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Corelation Matrix")
plt.show()

#check after handle missed data 
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

# Confusion Matrix
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

""" --Turkish tutorial for calculate conclude of analysis- 
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

# ROC curl 
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
When looking at the correlation matrix, I observed a strong positive correlation (0.75) between f1 and f4, and a negative correlation (-0.43) between f1 and the isVirus feature.
Missing values were identified in the dataset and filled using column averages.
I observed an increase in polarization between f1 and other features in the graphs.
A random forest model was used for the classification model.
The train and test data sets were split with a ratio of 0.35.
I examined the positive true/false and negative true/false rates using the confusion matrix.
I observed an AUC (Area Under Curve) of 0.94 on the ROC curve, indicating that the model performs better than random guessing.
The model's accuracy was observed to be 0.85 according to the test results.


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
