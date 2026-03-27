import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# 1. Charger les données
df = pd.read_csv("data/survey.csv")

# 2. Garder uniquement les colonnes utiles
colonnes = ['Age', 'Gender', 'family_history', 'work_interfere',
            'remote_work', 'benefits', 'seek_help', 'anonymity', 'treatment']
df = df[colonnes]

# 3. Nettoyer les données
# Corriger les âges aberrants
df = df[(df['Age'] >= 18) & (df['Age'] <= 65)]

# Simplifier le genre
df['Gender'] = df['Gender'].str.lower()
df['Gender'] = df['Gender'].apply(lambda x: 'male' if 'male' in str(x) 
                                   else ('female' if 'female' in str(x) else 'other'))

# Supprimer les lignes vides
df = df.dropna()

# 4. Encoder les colonnes texte en chiffres
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

# 5. Séparer X (variables) et y (cible)
X = df.drop('treatment', axis=1)
y = df['treatment']

# 6. Diviser : 80% entraînement / 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Entraîner le modèle
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 8. Afficher la précision
score = model.score(X_test, y_test)
print(f"✅ Précision du modèle : {score * 100:.2f}%")

# 9. Sauvegarder le modèle
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Modèle sauvegardé dans model.pkl")