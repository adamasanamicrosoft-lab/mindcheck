import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import pickle

# 1. Charger les données
df = pd.read_csv("data/survey.csv")

# 2. Garder les colonnes utiles
colonnes = ['Age', 'Gender', 'family_history', 'work_interfere',
            'remote_work', 'benefits', 'seek_help', 'anonymity', 'treatment']
df = df[colonnes]

# 3. Nettoyer
df = df[(df['Age'] >= 18) & (df['Age'] <= 65)]
df['Gender'] = df['Gender'].str.lower()
df['Gender'] = df['Gender'].apply(lambda x: 'male' if 'male' in str(x)
                                   else ('female' if 'female' in str(x) else 'other'))
df = df.dropna()

# 4. Encoder
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

# 5. Séparer X et y
X = df.drop('treatment', axis=1)
y = df['treatment']

# 6. Split 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Entraîner XGBoost
model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss'
)
model.fit(X_train, y_train)

# 8. Précision
score = model.score(X_test, y_test)
print(f"✅ Précision XGBoost : {score * 100:.2f}%")

# 9. Sauvegarder
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Modèle XGBoost sauvegardé dans model.pkl")