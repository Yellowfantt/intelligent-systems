import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


df = pd.read_csv("bases/ObesityDataSet_raw_and_data_sinthetic.csv")

print(df.head())

selected_features = df[['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight', 'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE']].copy()
target_variable = df['NObeyesdad'].copy()

le = LabelEncoder()
selected_features['Gender'] = le.fit_transform(selected_features['Gender'])
selected_features['family_history_with_overweight'] = le.fit_transform(selected_features['family_history_with_overweight'])
selected_features['FAVC'] = le.fit_transform(selected_features['FAVC'])
selected_features['CAEC'] = le.fit_transform(selected_features['CAEC'])
selected_features['SMOKE'] = le.fit_transform(selected_features['SMOKE'])
selected_features['SCC'] = le.fit_transform(selected_features['SCC'])

class_mapping = {
    'Insufficient_Weight': 0,
    'Normal_Weight': 1,
    'Overweight_Level_I': 2,
    'Overweight_Level_II': 3,
    'Obesity_Type_I': 4,
    'Obesity_Type_II': 5,
    'Obesity_Type_III': 6
}

target_variable = target_variable.map(class_mapping)

X_train, X_test, y_train, y_test = train_test_split(selected_features, target_variable, test_size=0.2, random_state=42)

# Normalização dos atributos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Testando diferentes valores para o hiperparâmetro k
print("---------------------------------------//----------------------------------------------------------")
best_k = None
best_accuracy = 0
for k in range(1, 11):
    # Criando o modelo k-NN
    model = KNeighborsClassifier(n_neighbors=k)
    
    # Treinando o modelo
    model.fit(X_train, y_train)
    
    # Fazendo previsões no conjunto de teste
    y_pred = model.predict(X_test)
    
    # Avaliando a precisão do modelo
    accuracy = accuracy_score(y_test, y_pred)
   
    print(f'k = {k}, Acurácia = {accuracy}')

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_k = k

print(f'Melhor k: {best_k} com acurácia: {best_accuracy}')
