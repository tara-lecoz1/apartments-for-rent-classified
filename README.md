
# README - Analyse Prédictive des Loyers aux États-Unis

## 📋 Description du Projet

Ce projet analyse un dataset de **100 000 annonces** d'appartements à louer aux États-Unis (UCI). L'objectif est de :
- Nettoyer et explorer les données immobilières
- Identifier les facteurs déterminants du prix de location
- Construire des modèles prédictifs robustes
- Créer un simulateur de loyer fonctionnel

**Focus géographique :** Californie, New York et Floride (marchés à forte volatilité)

---

## 🔧 Technologies Utilisées

| Domaine | Technologies |
|---------|-------------|
| **Langage** | Python 3.x |
| **Data Processing** | Pandas, NumPy |
| **Visualisation** | Matplotlib, Seaborn, Plotly |
| **Machine Learning** | scikit-learn (Régression, Random Forest, K-Means) |
| **Notebook** | Jupyter |

---

## 📊 Étapes du Projet

### 1️⃣ **Nettoyage des Données**
- Suppression des valeurs aberrantes (prix < 400$ ou > 30 000$)
- Normalisation textuelle des noms de villes (`.strip()`, `.title()`)
- Analyse du Kurtosis pour détecter les "ultra-outliers"
- Filtrage sur le marché standard (< 6 000$ pour 95% de la population)

### 2️⃣ **Exploration Descriptive (EDA)**
- **Visualisations bivariées :** Stripplot, Boxplot, Violin Plot
- **Analyse géographique :** Comparaison des États par prix médian
- **Étude du prix au m² :** KDE plots pour comprendre la densité du marché
- **Corrélation :** Matrice heatmap des variables numériques

**Insights clés :**
- Écart massif NY vs FL : +30-50% sur les mêmes configurations
- Surface (`square_feet`) et salles de bain (`bathrooms`) = facteurs dominants
- Localisation (`state`) = multiplicateur de prix, pas additionnel

### 3️⃣ **Modélisation Prédictive**

#### **Modèle 1 : Régression Linéaire**
```python
model = LinearRegression()
model.fit(X_train, y_train)
```
- **R² Score :** ~0.75
- **Avantage :** Explicabilité maximale (coefficients = impact en $)
- **Limite :** Sensible aux outliers

#### **Modèle 2 : Régression avec Transformation Log**
```python
y_train_log = np.log(y_train)
model_log.fit(X_train, y_train_log)
```
- **Impact :** MAE réduit de 15-20%
- **Logique :** Stabilise la variance sur un marché asymétrique

#### **Modèle 3 : Random Forest (Best)**
```python
rf_model = RandomForestRegressor(n_estimators=100, max_depth=15)
rf_model.fit(X_train_s, y_train_s)
```
- **R² Score :** ~0.82 (marché < 6000$)
- **MAE :** ~$180
- **Avantage :** Capture les interactions complexes (ex: effet de l'État dépend de la surface)

### 4️⃣ **Segmentation par Clustering**
```python
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)
```
- **Objectif :** Identifier 3 segments de marché automatiquement
    - Cluster 1 : Marché économique
    - Cluster 2 : Marché standard/familial
    - Cluster 3 : Segment premium

### 5️⃣ **Analyse Géographique Interactive**
```python
fig = px.scatter_map(df_state, lat="latitude", lon="longitude")
```
- Visualisation des densités d'annonces par État
- Détection des hubs majeurs (Silicon Valley, Manhattan, côtes floridienne)

---

## 📈 Résultats et Performances

### Comparaison des Modèles (Marché < 6 000$)

| Modèle | R² Score | MAE ($) | Avantage |
|--------|----------|---------|----------|
| Linear Regression | 0.758 | $285 | Transparent |
| Log-Linear | 0.762 | $248 | Stabilisé |
| **Random Forest** | **0.822** | **$180** | Précis + interactions |

### Impact des Variables (Random Forest Importance)
1. **square_feet** : ~45% (socle du prix)
2. **bathrooms** : ~20% (standing)
3. **bedrooms** : ~15% (capacité)
4. **state_NY** : ~12% (premium location)
5. **state_FL** : ~8% (ajustement)

---

## 🎯 Simulateur de Loyer

Fonction standalone pour estimer un loyer :

```python
def simulateur_loyer(sqft, beds, baths, state):
        intercept = 500  # Socle du modèle
        loyer = (intercept + 
                         (sqft * 0.95) +           # Prix/m²
                         (beds * -154) +           # Impact négatif (petit espace)
                         (baths * 291) +           # Impact positif (standing)
                         (is_ny * -36) +           # Ajustement NY
                         (is_fl * -952))           # Ajustement FL
        return loyer

# Exemple : 1000 sqft, 2 bed, 2 bath à New York
# Résultat : ~$1,650/mois
```

---

## 🔍 Insights Métier

1. **Disparité géographique massive :** À surface/type égal, une appartement coûte 30-50% plus cher à NY qu'en FL
2. **L'importance du confort :** Une salle de bain supplémentaire ajoute +$291 en moyenne (plus rentable qu'une chambre)
3. **Marché fragmenté :** NYC et CA affichent un Kurtosis > 50 (longue traîne de luxe), FL est plus stable (K ~ 10)
4. **Données incomplètes :** 5% du dataset reste imprévisible (ultra-luxe dépendant de facteurs émotionnels/intangibles)

---

## 📁 Structure du Projet

```
apartments-for-rent-classified/
├── main.ipynb                                    # Notebook principal
├── apartments_for_rent_classified_100K.csv      # Dataset
└── README.md                                    # Ce fichier
```

---

## 🚀 Comment Reproduire

1. **Charger les données :**
     ```python
     df = pd.read_csv('apartments_for_rent_classified_100K.csv', 
                                         sep=';', encoding='cp1252')
     ```

2. **Exécuter le nettoyage :** Sections 1-2 du notebook

3. **Entraîner le modèle :** Section 3 (Random Forest recommandé)

4. **Prédire :** Utiliser le simulateur ou `rf_model.predict(X_new)`

---

## ⚠️ Limitations

- **Données de 2019 :** Pas d'ajustement inflation post-COVID
- **Marché ultra-luxe imprévisible :** R² baisse à ~0.60 pour prix > 6000$
- **Variables manquantes :** Pas d'info sur l'âge du bâtiment, finitions, proximité transports
- **Biais géographique :** Focus sur 3 États, données limitées en zones rurales

---

## 📚 Références

- **Dataset :** [UCI Machine Learning - Apartment for Rent Classified](https://archive.ics.uci.edu/dataset/555/)
- **Méthodes :** Scikit-learn documentation, Feature Engineering best practices
