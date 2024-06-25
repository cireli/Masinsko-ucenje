# Importovanje potrebnih biblioteka
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from ucimlrepo import fetch_ucirepo
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from ucimlrepo import fetch_ucirepo 
# fetch dataset 
statlog_vehicle_silhouettes = fetch_ucirepo(id=149) 
  
# data (as pandas dataframes) 
X = statlog_vehicle_silhouettes.data.features 
y = statlog_vehicle_silhouettes.data.targets 
  
# metadata 
print(statlog_vehicle_silhouettes.metadata) 
  
# variable information 
print(statlog_vehicle_silhouettes.variables) 

vehicle_df = statlog_vehicle_silhouettes.data.features
vehicle_df['class'] = statlog_vehicle_silhouettes.data.targets

# Inicijalizujemo LabelEncoder
label_encoder = LabelEncoder()

# Konvertujemo klasne vrednosti u numeričke vrednosti
vehicle_df['class'] = label_encoder.fit_transform(vehicle_df['class'])

# Prikaz korelacije između feature-a
cor = vehicle_df.corr()

# matrica korelacije
sns.set(font_scale=1.15)
fig, ax = plt.subplots(figsize=(18, 15))
sns.heatmap(cor, vmin=0.8, annot=True, linewidths=0.01, center=0, linecolor="white", cbar=False, square=True)
plt.title('Korelacija izmedju atributa', fontsize=18)
ax.tick_params(labelsize=18)
plt.savefig('matricakorelacije.png', format='png')
plt.show()

# Učitavanje podataka
data = pd.read_csv('vehicle.csv')
vehicle_counts = data['class'].value_counts()
plt.figure(figsize=(10, 6))
plt.pie(vehicle_counts, labels=vehicle_counts.index, startangle=90, autopct='%1.1f%%')
plt.title('Verovatnoća učestanosti klasa')
plt.savefig('pieplot.png', format='png')
plt.show()

# Podela podataka na trening i test skupove
#X = vehicle_df.drop(columns='class')
y = vehicle_df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Popunjavanje NaN vrednosti u X_train i X_test
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Definisanje modela
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'KNN': KNeighborsClassifier(),
    'SVM': SVC(),
    'MLP': MLPClassifier(max_iter=1000)
}

# Treniranje i evaluacija modela
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Classification Report for {name}:\n")
    print(metrics.classification_report(y_test, y_pred))
    print("\n" + "="*60 + "\n")
    results[name] = metrics.accuracy_score(y_test, y_pred)

# Prikaz rezultata tačnosti

preciznosti=[]

# Koriscenje MLP Classifier-a za treniranje
mlp = MLPClassifier(hidden_layer_sizes=(500, 100), max_iter=300)
mlp.fit(X_train, y_train)
y_mlp_pred = mlp.predict(X_test)

print("Ocena za X-train sa Y-train je : ", mlp.score(X_train, y_train))
print("Ocena za X-test sa Y-test je : ", mlp.score(X_test, y_test))
print("Evaluacija MLP-a : Accuracy score ", accuracy_score(y_test, y_mlp_pred))
preciznosti.append(accuracy_score(y_test, y_mlp_pred))
# Koriscenje Logicke Regresije (svodjenje na klasifikaciju) za treniranje
Lo_model = LogisticRegression(solver='liblinear')
Lo_model.fit(X_train, y_train)

print("Ocena za X-train sa Y-train je : ", Lo_model.score(X_train, y_train))
print("Ocena za X-test sa Y-test je : ", Lo_model.score(X_test, y_test))

y_pred_Lo = Lo_model.predict(X_test)
print("Evaluacija Logicke Regresije : Accuracy score ", accuracy_score(y_test, y_pred_Lo))
preciznosti.append(accuracy_score(y_test, y_pred_Lo))

# Koriscenje Decision Tree Classifiera za treniranje
Tree_model = DecisionTreeClassifier(random_state=42)

# Definisanje grida hiperparametara radi pronalazenja optimalnih vrednosti hiperparametara
param_grid = {
    'max_depth': [20, 25, 30, 35, 40],
    'min_samples_split': [2, 3, 4, 5],
    'min_samples_leaf': [1, 2, 3, 4],
    'max_features': [None, 'sqrt', 'log2'],
    'class_weight': [None, 'balanced']
}

# Pretraga najboljih parametara
grid_search = GridSearchCV(Tree_model, param_grid, cv=5, n_jobs=-1, verbose=1, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
# Overrideujemo Tree model novim objektom sa optimalnim hiperparametrima
Tree_model = DecisionTreeClassifier(**grid_search.best_params_)
Tree_model.fit(X_train, y_train)

y_pred_tree = Tree_model.predict(X_test)

print("Ocena za X-train sa Y-train je : ", Tree_model.score(X_train, y_train))
print("Ocena za X-test sa Y-test je : ", Tree_model.score(X_test, y_test))
print("Evaluacija Decision Tree : Accuracy score ", accuracy_score(y_test, y_pred_tree))
preciznosti.append(accuracy_score(y_test, y_pred_tree))
# Koriscenje SVC za treniranje
svc_model = SVC(C=50, kernel="rbf")

svc_model.fit(X_train, y_train)

y_pred_svc = svc_model.predict(X_test)

print("Ocena za X-train sa Y-train je : ", svc_model.score(X_train, y_train))
print("Ocena za X-test sa Y-test je : ", svc_model.score(X_test, y_test))
print("Evaluacija SVC-a : Accuracy score ", accuracy_score(y_test, y_pred_svc))
preciznosti.append(accuracy_score(y_test, y_pred_svc))
#Koriscenje KNeighbors Classifier-a za treniranje
K_model = KNeighborsClassifier(n_neighbors=5)
K_model.fit(X_train, y_train)

y_pred_k = K_model.predict(X_test)

print("Ocena za X-train sa Y-train je : ", K_model.score(X_train, y_train))
print("Ocena za X-test sa Y-test je : ", K_model.score(X_test, y_test))
print("Evaluacija K Neighbors-a : Accuracy score ", accuracy_score(y_test, y_pred_k))
preciznosti.append(accuracy_score(y_test, y_pred_k))

results_df = pd.DataFrame({'Model': ['MLP', 'Logistic Regression', 'Decision Tree', 'SVC', 'K Neighbors'],
                            'Accuracy': preciznosti})
plt.figure(figsize=(10, 6))
sns.barplot(x=['MLP', 'Logistic Regression', 'Decision Tree', 'SVC', 'K Neighbors'], y=preciznosti, palette='viridis')
plt.title('Poredjenje efikasnosti modela')
plt.ylabel('Accuracy')
plt.xlabel('Model')
plt.xticks(rotation=45)
plt.savefig('model_accuracy.png', format='png')
plt.show()