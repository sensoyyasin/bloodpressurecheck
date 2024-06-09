import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import tkinter as tk
from sklearn.preprocessing import RobustScaler
from scipy import stats
from scipy.stats import zscore
from scipy.stats.mstats import winsorize
import os

current_directory = os.getcwd()

file_name = "heart.csv"
file_path = os.path.join(current_directory, file_name)

df = pd.read_csv(file_path)
print(df.head())

new_columns = ["age","sex","cp","trtbps","chol","fbs","rest_ecg","thalach","exang","oldpeak","slope","ca","thal","target"]

df.columns = new_columns
print(df.head())

print('Shape of Dataset:', df.shape)
df.info()

isnull_number = []
for i in df.columns:
    x = df[i].isnull().sum()
    isnull_number.append(x)

pd.DataFrame(isnull_number, index = df.columns, columns = ["Total Missing Values"])

print(df["cp"].value_counts().sum())

print(df["cp"].value_counts().count()) #total number of differences, categorical structure

unique_number = []
for i in df.columns:
    x = df[i].value_counts().count()
    unique_number.append(x)

print(pd.DataFrame(unique_number, index = df.columns, columns = ["Total Missing Values"]))

print(df.head())

# Numeric variables are basically quantitative data obtained from a variable and the value has a sense of size.
# person's age, weight, distance, temperature the price ...

# Categorical variables are qualitative. It usually is a word or a symbol.
numeric_var = ["age", "trtbps", "chol", "thalach", "oldpeak"]
categoric_var = ["sex", "cp", "fbs", "rest_ecg", "exang", "slope", "ca", "thal", "target"]

print(df[numeric_var].describe())

print(numeric_var)

numeric_axis_name = ["Age of the Patient", "Resting Blood Pressure", "Cholesterol", "Maximum Heart Rate Achieved", "ST Depression"]

list(zip(numeric_var, numeric_axis_name))

title_font = {"family" : "arial", "color" : "darkred", "weight" : "bold", "size" : 15}
axis_font = {"family" : "arial", "color" : "darkblue", "weight" : "bold", "size" : 13}


# Seaborn ile grafikleri Numerik verileri çizdirme yoğunluklara göre.

# for i, z in list(zip(numeric_var, numeric_axis_name)):
#     plt.figure(figsize = (8, 6), dpi = 80)
#     sns.distplot(df[i], hist_kws = dict(linewidth = 1, edgecolor = "k"), bins = 20)
#     plt.title(i, fontdict = title_font)
#     plt.xlabel(z, fontdict = axis_font)
#     plt.ylabel("Density", fontdict = axis_font)
#     plt.tight_layout()
#     plt.show()

print(categoric_var)

categoric_axis_name = ["Gender", "Chest Pain Type", "Fasting Blood Pressure", "Resting Electrocardiographic Results",
                        "Exercise Induced Angina", "The Slope of ST Segment", "Number of Major Vessels", "Thal", "Target"]

list(zip(categoric_var, categoric_axis_name))

print(df["cp"].value_counts())
list(df["cp"].value_counts())
list(df["cp"].value_counts().index)

title_font = {"family" : "arial", "color" : "darkred", "weight" : "bold", "size" : 15}
axis_font = {"family" : "arial", "color" : "darkblue", "weight" : "bold", "size" : 13}

# Kategorik Verilerin Yuvarlak Grafiklendirilmesi

# for i, z in list(zip(categoric_var, categoric_axis_name)):
#     fig, ax = plt.subplots(figsize = (8, 6))

#     observation_values = list(df[i].value_counts().index)
#     total_observation_values = list(df[i].value_counts())

#     ax.pie(total_observation_values, labels=observation_values, autopct = "%1.1f%%", startangle=110, labeldistance=1.1)
#     ax.axis("equal")

#     plt.title((i + "(" + z + ")"), fontdict = title_font)
#     plt.legend()
#     plt.show()


# Examining the Missing Data According to the Analysis Result

print(df[df["thal"] == 0])
df["thal"] = df["thal"].replace(0, np.nan)
print(df.loc[[48, 281], :])

isnull_number = []
for i in df.columns:
    x = df[i].isnull().sum()
    isnull_number.append(x)

print(pd.DataFrame(isnull_number, index = df.columns, columns = ["Total Missing Values"]))

df["thal"].fillna(2, inplace = True)
print(df.loc[[48, 281], :])

print(df)

df["thal"] = pd.to_numeric(df["thal"], downcast = "integer")
print(df.loc[[48, 281], :])

isnull_number = []
for i in df.columns:
    x = df[i].isnull().sum()
    isnull_number.append(x)

print(pd.DataFrame(isnull_number, index = df.columns, columns = ["Total Missing Values"]))

print(df["thal"].value_counts())

print(numeric_var)
numeric_var.append("target")
print(numeric_var)

title_font = {"family" : "arial", "color" : "darkred", "weight" : "bold", "size" : 15}
axis_font = {"family" : "arial", "color" : "darkblue", "weight" : "bold", "size" : 13}


# Target'a göre grafik çizdirimi

# for i, z in list(zip(numeric_var, numeric_axis_name)):
#     graph = sns.FacetGrid(df[numeric_var], hue = "target", height = 5, xlim = ((df[i].min() - 10), (df[i].max() + 10)))
#     graph.map(sns.kdeplot, i, shade = True)
#     graph.add_legend()

#     plt.title(i, fontdict = title_font)
#     plt.xlabel(z, fontdict = axis_font)
#     plt.ylabel("Density", fontdict = axis_font)

#     plt.tight_layout()
#     plt.show()

# korelasyon
print(df[numeric_var].corr())
print(df[numeric_var].corr().iloc[:, [-1]])


# Categorical Variables - Target Variable(Analysis with Count Plot)

print(categoric_var)

print(df[categoric_var].corr())

print(df[categoric_var].corr().iloc[:, [-1]])


# Numerical Varibables - Target Variable(Analysiz with Count Plot)

print(numeric_var)

print(numeric_var.remove("target"))

print(df[numeric_var].head())

# Feature Scaling with the RobustScaler Method

robust_scaler = RobustScaler()

scaled_data = robust_scaler.fit_transform(df[numeric_var])

print(scaled_data)
print(type(scaled_data))

df_scaled = pd.DataFrame(scaled_data ,columns = numeric_var)
print(df_scaled.head())


# Creating a New DataFrame with the Melt() Function

df_new = pd.concat([df_scaled, df.loc[:, "target"]],axis = 1)
print(df_new.head())

melted_data = pd.melt(df_new, id_vars = "target", var_name = "variables", value_name = "value")
print(melted_data)


# Melted data figure
# plt.figure(figsize = (8, 5))
# sns.swarmplot(x = "variables", y = "value", hue = "target", data = melted_data)
# plt.show()


#  Numerical Variables - Categorical Variables (Analysis with Swarm Plot)

# axis_font = {"family" : "arial", "color" : "black", "weight" : "bold", "size" : 14}
# for i in df[categoric_var]:
#     df_new = pd.concat([df_scaled, df.loc[:, i]], axis = 1)
#     melted_data = pd.melt(df_new, id_vars = i, var_name = "variables", value_name = "value")

#     plt.figure(figsize = (8, 5))
#     sns.swarmplot(x = "variables", y = "value", hue = i, data = melted_data)

#     plt.xlabel("variables", fontdict = axis_font)
#     plt.ylabel("value", fontdict = axis_font)

#     plt.tight_layout()
#     plt.show()

# Numerical Variables - Categorical Variables (Analysis with Box Plot)

# axis_font = {"family" : "arial", "color" : "black", "weight" : "bold", "size" : 14}
# for i in df[categoric_var]:
#     df_new = pd.concat([df_scaled, df.loc[:, i]], axis = 1)
#     melted_data = pd.melt(df_new, id_vars = i, var_name = "variables", value_name = "value")

#     plt.figure(figsize = (8, 5))
#     sns.boxplot(x = "variables", y = "value", hue = i, data = melted_data)

#     plt.xlabel("variables", fontdict = axis_font)
#     plt.ylabel("value", fontdict = axis_font)

#     plt.tight_layout()
#     plt.show()


# Relationships between variables(Analysis with Heatmap)

print(df_scaled)

df_new2 = pd.concat([df_scaled, df[categoric_var]], axis = 1)

print(df_new2)
print(df_new2.corr())

# Heatmap
# plt.figure(figsize = (15, 10))
# sns.heatmap(data = df_new2.corr(), cmap = "Spectral", annot = True, linewidths = 0.5)
# plt.show()


# Preparation for Modeling

# Dropping Columns with Low Correlation

print(df.head())

df.drop(["chol", "fbs", "rest_ecg"], axis = 1, inplace = True)

print(df.head())

# Struggling Outliers

# Visualizing Outliers

# fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize = (20, 6))

# ax1.boxplot(df["age"])
# ax1.set_title("age")

# ax2.boxplot(df["trtbps"])
# ax2.set_title("trtbps")

# ax3.boxplot(df["thalach"])
# ax3.set_title("thalach")

# ax4.boxplot(df["oldpeak"])
# ax4.set_title("oldpeak")

# plt.show()


# Trtbps'nin ne kadar saptığını gördük. Z-score bakıyoruz  Z-skoru kullanarak, belirli bir eşik değerinden büyük olan aykırı değerlerin sayısını hesaplıyor
z_scores_trtbps = zscore(df["trtbps"]) # trtbps" sütunundaki değerlerin Z-skorlarını hesaplar.
for threshold in range(1, 4):
    print("Threshold Value : {}".format(threshold))
    print("Number of Outliers: {}".format(len(np.where(z_scores_trtbps > threshold)[0]))) # Z-skorları eşik değerinden büyük olan aykırı değerlerin sayısını hesaplar ve yazdırır.
    print("------------------")

print(df[z_scores_trtbps > 2][["trtbps"]])
print('trtbps_min_deger: ', df[z_scores_trtbps > 2].trtbps.min())

print('sınırlar icinde(170den kucuk) en buyuk sayı', df[df["trtbps"] < 170].trtbps.max())

winsorize_percentile_trtbps = (stats.percentileofscore(df["trtbps"], 165)) / 100
print(winsorize_percentile_trtbps)
print(1 - winsorize_percentile_trtbps)

trtbps_winsorize = winsorize(df.trtbps, (0, (1 - winsorize_percentile_trtbps)))

# After the examining trtbps value or column
# plt.boxplot(trtbps_winsorize)
# plt.xlabel("trtbps_winsorize", color = "b")
# plt.show()

df["trtbps_winsorize"] = trtbps_winsorize
print(df.head())

# Thalach Variable

def iqr(df, var):
    q1 = np.quantile(df[var], 0.25)
    q3 = np.quantile(df[var], 0.75)
    diff = q3 - q1
    lower_v = q1 - (1.5 * diff)
    upper_v = q3 + (1.5 * diff)
    return df[(df[var] < lower_v) | (df[var] > upper_v)]

thalach_out = iqr(df, "thalach")

# print(thalach_out)
df.drop([272], axis = 0, inplace = True)
# print(df["thalach"][270:275])

# plt.boxplot(df["thalach"])
# plt.show()

# Oldpeak Variable

def iqr(df, var):
    q1 = np.quantile(df[var], 0.25)
    q3 = np.quantile(df[var], 0.75)
    diff = q3 - q1
    lower_v = q1 - (1.5 * diff)
    upper_v = q3 + (1.5 * diff)
    return df[(df[var] < lower_v) | (df[var] > upper_v)]

print(iqr(df, "oldpeak"))
print(df[df["oldpeak"] < 4.2].oldpeak.max())

winsorize_percentile_oldpeak = (stats.percentileofscore(df["oldpeak"], 4)) / 100
print(winsorize_percentile_oldpeak)

oldpeak_winsorize = winsorize(df.oldpeak, (0, (1 - winsorize_percentile_oldpeak)))

# plt.boxplot(oldpeak_winsorize)
# plt.xlabel("oldpeak_winsorize", color = "b")
# plt.show()

df["oldpeak_winsorize"] = oldpeak_winsorize

print(df.head())

df.drop(["trtbps", "oldpeak"], axis = 1, inplace = True)

print(df.head())


# Determining Distributions of Numeric Variables

print(df.head())

# After the examining

# fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize = (20, 6))

# ax1.hist(df["age"])
# ax1.set_title("age")

# ax2.hist(df["trtbps_winsorize"])
# ax2.set_title("trtbps_winsorize")

# ax3.hist(df["thalach"])
# ax3.set_title("thalach")

# ax4.hist(df["oldpeak_winsorize"])
# ax4.set_title("oldpeak_winsorize")

# plt.show()

print(df[["age", "trtbps_winsorize", "thalach", "oldpeak_winsorize"]].agg(["skew"]).transpose())

# Transformation Operations on Unsymmetrical Data

df["oldpeak_winsorize_log"] = np.log(df["oldpeak_winsorize"])
df["oldpeak_winsorize_sqrt"] = np.sqrt(df["oldpeak_winsorize"])

print(df.head())

print(df[["oldpeak_winsorize", "oldpeak_winsorize_log", "oldpeak_winsorize_sqrt"]].agg(["skew"]).transpose())
df.drop(["oldpeak_winsorize", "oldpeak_winsorize_log"], axis = 1, inplace = True)
print(df.head())


# Applying One Hot Encoding Method to Categorical Variables

df_copy = df.copy()
df_copy.head()

print(categoric_var)

categoric_var.remove("fbs")
categoric_var.remove("rest_ecg")

print(categoric_var)

df_copy = pd.get_dummies(df_copy, columns = categoric_var[:-1], drop_first = True)

print(df_copy.head())

#  Feature Scaling with the RobustScaler Method for Machine Learning Algorithms

new_numeric_var = ["age", "thalach", "trtbps_winsorize", "oldpeak_winsorize_sqrt"]
robus_scaler = RobustScaler()
df_copy[new_numeric_var] = robust_scaler.fit_transform(df_copy[new_numeric_var])
print(df_copy.head())

# Separating Data into Test and Training Set

X = df_copy.drop(["target"], axis = 1)
y = df_copy[["target"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 3)
print(X_train.head())
print(y_train.head())

print(f"X_train: {X_train.shape[0]}")
print(f"X_test: {X_test.shape[0]}")
print(f"y_train: {y_train.shape[0]}")
print(f"y_test: {y_test.shape[0]}")


# Modelling

# First Algorithm - Logistic Regression

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# log_reg = LogisticRegression()
# print('Algorithm_name: ', log_reg)

# log_reg.fit(X_train, y_train)
# y_pred = log_reg.predict(X_test)

# accuracy = accuracy_score(y_test, y_pred)
# print("Test Accuracy : {}".format(accuracy)) # 0.87

# # Cross Validation

from sklearn.model_selection import cross_val_score

# scores = cross_val_score(log_reg, X_test, y_test, cv = 10)
# print("Cross-Validation Accuracy Scores", scores.mean()) # 0.8666666666666666

# Second Algorithm - Decision Tree Algorithm

from sklearn.tree import DecisionTreeClassifier

# dec_tree = DecisionTreeClassifier(random_state = 5)

# dec_tree.fit(X_train,y_train)

# y_pred = dec_tree.predict(X_test)

# print("The test accuracy score of Decision Tree is:", accuracy_score(y_test, y_pred)) #0.8387096774193549

# scores = cross_val_score(dec_tree, X_test, y_test, cv = 10)
# print("Cross-Validation Accuracy Scores", scores.mean()) #  0.8333333333333333

# Support Vector Machine Algorithm

from sklearn.svm import SVC

svc_model = SVC(random_state = 6)

# svc_model.fit(X_train, y_train)

# y_pred = svc_model.predict(X_test)
# print("The test accuracy score of SVC is:", accuracy_score(y_test, y_pred)) # 0.8709677419354839

# scores = cross_val_score(svc_model, X_test, y_test, cv = 10)
# print("Cross-Validation Accuracy Scores", scores.mean()) # 0.8333333333333334

# Random Forest Algorithm

from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(random_state = 6)
random_forest.fit(X_train, y_train)

y_pred = random_forest.predict(X_test)
print("The test accuracy score of Random forest is", accuracy_score(y_test, y_pred)) # 0.9032258064516129

# scores = cross_val_score(random_forest, X_test, y_test, cv = 5)
# print("Cross-Validation Accuracy Scores", scores.mean()) # 0.9


# ROC-CURVE For Random Forest -> 0.93
# from sklearn.metrics import roc_curve, auc
# import matplotlib.pyplot as plt

# # RandomForestClassifier modelinizin predict_proba metodu ile sınıflandırma olasılıklarını alın
# y_proba = random_forest.predict_proba(X_test)[:, 1]

# # ROC eğrisi için fpr ve tpr değerlerini hesaplayın
# fpr, tpr, _ = roc_curve(y_test, y_proba)

# # ROC eğrisinin altında kalan alanı (AUC) hesaplayın
# roc_auc = auc(fpr, tpr)

# # ROC eğrisini çizin
# plt.figure()
# plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.legend(loc="lower right")
# plt.show()

# Doğruluk oranlarını bir grafikte gösterme
accuracy_scores = {
    "Logistic Regression": 0.87,
    "Decision Tree": 0.8387,
    "Support Vector Machine": 0.8709,
    "Random Forest": 0.9032
}

# Doğruluk oranlarını ve algoritma isimlerini al
algorithms = list(accuracy_scores.keys())
scores = list(accuracy_scores.values())

# Yatay bar grafiği oluştur
plt.figure(figsize=(10, 6))
sns.barplot(x=scores, y=algorithms, palette="viridis")
plt.title("Accuracy Scores of Different Algorithms", fontsize=12, color='darkred', weight='bold')
plt.xlabel("Accuracy Scores", fontsize=13, color='darkblue', weight='bold')
plt.ylabel("Algorithms", fontsize=13, color='darkblue', weight='bold')
plt.xlim(0, 1)  # Doğruluk oranı 0-1 arasında olduğu için x ekseni sınırlarını belirleme
plt.grid(axis='x', linestyle='--', alpha=0.7)

# En yüksek doğruluk oranına sahip modeli belirtme
max_score = max(scores)
plt.axvline(max_score, color='green', linestyle='--', linewidth=2, label=f'Best Model ({max_score:.4f})')

plt.legend() # Aciklama kutusu icin
plt.show()

