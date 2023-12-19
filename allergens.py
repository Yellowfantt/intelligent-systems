import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Carregando os dados
df = pd.read_csv("bases/food_ingredients_and_allergens.csv")

# Visualizando as primeiras linhas do DataFrame
print("Original DataFrame:")
print(df.head())

# 1. Seleção dos atributos
# Vamos manter as colunas relevantes como atributos e a coluna 'Allergens' como variável alvo
colunas_relevantes = ['Main Ingredient', 'Sweetener', 'Fat/Oil', 'Seasoning', 'Allergens']
df_relevant = df[colunas_relevantes]

X = df_relevant.drop(columns=['Allergens'])
y = df_relevant['Allergens']

# 2. Encoding dos atributos categóricos
# Vamos usar One-Hot Encoding para transformar os valores categóricos em colunas binárias
X_encoded = pd.get_dummies(X)

# 3. Normalização dos atributos
# Normalizamos os atributos para colocá-los em uma escala semelhante
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X_encoded)

# Label Encoding para a variável alvo 'Allergens'
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Dividindo o conjunto de dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_encoded, test_size=0.2, random_state=42)

# 4. Classificação com KNN (Testar melhor hiperparâmetro K)
# Vamos testar diferentes valores de k
best_k = None
best_accuracy = 0

for k in range(1, 11):
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train, y_train)
    y_pred = knn_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f'Acurácia para k={k}: {accuracy}')
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_k = k

print(f'Melhor k: {best_k} com acurácia: {best_accuracy}')
