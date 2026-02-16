# Analyse de Données & Modélisation Immobilière (US Market)

## Résumé des Performances

| Modèle | État | R² Score | Erreur Moyenne (MAE) |
| --- | --- | --- | --- |
| **Régression Linéaire** | Objectif initial | **0.65** | **258.28 $** |
| **Random Forest (Base)** | Comparatif | **0.77** | **184.56 $** |
| **Random Forest (Final)** | Optimisé | **0.876** | **138.18 $** |

---

## Analyse Exploratoire (EDA) & Nettoyage

Le projet débute par un assainissement rigoureux d'un dataset de 100 000 annonces :

* **Traitement des données manquantes** : Suppression de `address` (90% de vide) et `pets_allowed` (non discriminante).
* **Identification des Outliers** : Utilisation de Boxplots pour détecter les prix aberrants.
* **Segmentation Stratégique** : Création des catégories **Budget**, **Standard**, et **Prestige** via `pd.qcut` pour valider la structure du marché (Accuracy de classification : 83%).
* **Filtrage par le prix au pied carré** : Élimination du 1% extrême (`price_per_sqfeet`) pour ne garder que les données économiquement cohérentes.

---

## Feature Engineering (Enrichissement)

Pour améliorer la prédiction, plusieurs variables "métier" ont été créées :

* **Contexte Local (`city_wealth`)** : Prix médian par ville pour compenser l'absence de géolocalisation précise.
* **Extraction de Mots-Clés** : Création de variables binaires (`has_pool`, `has_gym`, `has_parking`, etc.) par analyse du texte de la colonne `amenities`.
* **Indicateur de Quartier (`avg_city_price_sqft`)** : Moyenne locale du prix au m² pour affiner la valeur intrinsèque de l'emplacement.

---

## Confrontation des Modèles

### La Régression Linéaire (Objectif du Projet)

Le modèle de régression linéaire a servi de base de référence.

* **Conclusion** : Avec un **R² de 0.65**, il s'avère insuffisant.
* **Analyse de l'échec** : Le modèle linéaire suppose une proportionnalité constante. Or, l'immobilier présente des **effets de seuil** (le prix n'augmente pas de façon rectiligne avec la surface) et des **interactions complexes** entre variables.

### Le Random Forest Regressor (Optimisation)

Pour pallier ces limites, un algorithme d'ensemble a été déployé :

* **Configuration** : `n_estimators=200`, `max_depth=25`.
* **Avantage** : Capture les relations non-linéaires et les interactions entre la richesse de la ville et les équipements.

---

## Nettoyage par les Résidus

Une étape avancée a consisté à analyser les erreurs de l'IA (résidus).

* **Constat** : Les erreurs > 20% se concentraient sur les extrêmes (**50% Prestige**, **49% Budget**).
* **Action** : Suppression de 4 000 lignes suspectes (données bruitées ou atypiques).
* **Résultat** : Un modèle expert du **"Cœur de Marché"** (500 - 1 300 sqft) avec une précision finale de **87,6%** et une erreur de seulement **138 $**.

---

## Importance des Variables

L'analyse finale des `feature_importances_` révèle que la **localisation** (`avg_city_price_sqft`) et la **richesse de la ville** (`city_wealth`) sont les facteurs les plus déterminants.
