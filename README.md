# Segmentation marketing de clients par clustering (K-Means)

[![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/asaphmbere18-hub/segmentation-marketing-clients/blob/main/Segmentation_Marketing_Clients_Portfolio.ipynb)
![Python](https://img.shields.io/badge/Python-3-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-K--Means%20%7C%20ACP%20%7C%20t--SNE-orange)

**Regrouper 2 046 clients en segments à partir de leurs dépenses et de leurs habitudes d'achat, puis proposer une action marketing pour chaque segment.**

Résultat principal : **22 % des clients génèrent 56 % des dépenses.** La segmentation fait ressortir trois groupes de valeur très différente, avec une priorité d'action pour chacun.

![Poids de chaque segment dans la clientèle et dans les dépenses](images/poids_segments.png)

Projet réalisé dans le cadre de mon Bachelor IA, Data et Management (NEXA Digital School).

---

## Sommaire
1. [Objectif](#objectif)
2. [Données](#données)
3. [Démarche](#démarche)
4. [Résultats](#résultats)
5. [Recommandations](#recommandations)
6. [Limites](#limites)
7. [Lancer le notebook](#lancer-le-notebook)
8. [Structure du dépôt](#structure-du-dépôt)

## Objectif
Une entreprise de produits de consommation veut mieux connaître sa clientèle pour adapter ses offres. Il s'agit de :
- regrouper les clients en **segments homogènes** ;
- décrire chaque segment (revenu, produits achetés, canaux, sensibilité aux promotions) ;
- proposer pour chacun une **stratégie marketing** : produits, canal de communication, priorité.

## Données
- **Source :** jeu de données Kaggle [Customer Personality Analysis](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis), fichier `marketing_campaign.csv` (colonnes séparées par des tabulations).
- **Contenu :** 2 240 clients et 29 colonnes : profil (année de naissance, revenu, enfants), dépenses sur 6 catégories de produits, achats par canal (web, catalogue, magasin), visites web, réponses à 6 campagnes marketing.
- **Après nettoyage :** 2 046 clients.

Le fichier `marketing_campaign.csv` est inclus dans ce dépôt : le notebook le lit directement, sans téléchargement.

## Démarche

| Étape | Ce qui a été fait | Pourquoi |
|---|---|---|
| **1. Nettoyage** | 24 revenus manquants remplacés par la médiane ; 182 doublons retirés (profils identiques hors identifiant) ; valeurs aberrantes de l'âge et du revenu retirées par la règle de l'IQR (dont un revenu de 666 666 et les années de naissance 1893, 1899 et 1900). | La médiane résiste mieux aux valeurs extrêmes que la moyenne. Un doublon compterait deux fois le même client. |
| **2. Création de variables** | `Total_Spending`, `Total_Purchases` (web + catalogue + magasin), `Total_Children`, `Customer_Tenure` (ancienneté en jours), `Average_Spending` (panier moyen), `Age` (année de référence 2014). | Cinq variables lisibles résument 6 catégories de dépenses et 3 canaux. 2014 est l'année de la dernière inscription du fichier : l'âge correspond à la période de collecte. |
| **3. Choix des variables du clustering** | Les 5 variables ci-dessus, standardisées. Le revenu, l'âge, les visites web, les promotions et les réponses aux campagnes servent seulement à **décrire** les segments après coup. | Si les réponses aux campagnes servaient à construire les segments, on ne pourrait plus s'en servir pour vérifier qu'ils sont cohérents. |
| **4. ACP** | 4 composantes dépassent 90 % de la variance (97,5 %). PC1 (52 %) = valeur et intensité d'achat ; PC2 (20 %) = ancienneté. | L'ACP sert à comprendre la structure des données et à les visualiser en 2D. K-Means est entraîné sur les 5 variables standardisées. |
| **5. Choix de k** | Méthode du coude (pas de coude net) et score de silhouette, calculés **avant et après** le retrait d'un client aberrant (1 achat pour 1 679 de dépenses). | Avant retrait, K-Means isolait ce client seul dans un cluster à partir de k = 3, ce qui gonflait artificiellement la silhouette (0,39 au lieu de 0,30). |
| **6. Modèle final** | K-Means, `k = 3`, `n_init=10`, `random_state=42`, sur 2 046 clients. | Voir le paragraphe « Pourquoi k = 3 » ci-dessous. |
| **7. Visualisation** | Projection ACP avec zones de chevauchement, comparaison avec t-SNE. | Voir où les segments se touchent et vérifier que la séparation n'est pas un effet de la projection. |

![Choix de k : inertie et silhouette avant et après retrait du client aberrant](images/choix_k.png)

### Pourquoi k = 3 alors que la silhouette est meilleure à k = 2 ?
Après nettoyage, la silhouette vaut **0,40 à k = 2** et **0,30 à k = 3**. Le critère statistique seul choisirait donc k = 2. Mais k = 2 sépare seulement les petits et les gros acheteurs, ce qui ne permet qu'une seule action marketing. k = 3 fait apparaître un **segment intermédiaire** qui a sa propre stratégie (faire monter en valeur). Son score reste du même ordre que ceux de k = 4 à 10. Le choix de k = 3 repose donc sur l'**intérêt métier**, et je l'assume comme tel.

## Résultats

| Segment | Clients | Part des dépenses | Dépense moyenne | Revenu moyen | Réponse à la dernière campagne |
|---|---|---|---|---|---|
| **Gros acheteurs aisés** | 458 (22 %) | **56 %** | 1 503 | 75 485 | **31 %** |
| **Familles actives et sensibles aux promotions** | 613 (30 %) | 36 % | 732 | 58 925 | 13 % |
| **Petits budgets familiaux** | 975 (48 %) | 8 % | 104 | 35 930 | 9 % |

- **Gros acheteurs aisés :** presque pas d'enfants (0,2 par foyer), vin et viande en tête, achats par catalogue et en magasin, très peu d'achats en promotion (7 %), les plus réceptifs aux campagnes.
- **Familles actives :** un enfant en moyenne, clients les plus anciens, le plus grand nombre d'achats en ligne et en promotion, le vin pèse 55 % de leur budget.
- **Petits budgets familiaux :** 1,3 enfant par foyer, 6,4 visites web par mois mais peu d'achats en ligne, plus d'un tiers des achats en promotion.

Les réponses aux campagnes n'ont pas servi à construire les segments. Le fait qu'elles suivent le même ordre (31 %, 13 %, 9 %) confirme que la segmentation est cohérente.

Les noms des segments sont attribués selon le **rang de dépenses** et non selon le numéro de cluster, car les numéros (0, 1, 2) peuvent changer d'une exécution à l'autre. Sur le graphique ci-dessous : cluster 0 = gros acheteurs aisés, cluster 1 = petits budgets familiaux, cluster 2 = familles actives.

![Projection des clusters sur les deux premières composantes et zones de chevauchement](images/clusters_pca.png)

| Espace de calcul | Silhouette |
|---|---|
| 5 variables standardisées | 0,30 |
| Projection ACP (2D) | 0,36 |
| Projection t-SNE (2D) | 0,14 |

Une séparation plus nette sur un graphique ne prouve pas un meilleur clustering : la silhouette de référence est celle de l'espace d'origine (0,30).

## Recommandations

| Priorité | Segment | Objectif | Actions clés | Canal |
|---|---|---|---|---|
| 1 | Gros acheteurs aisés | **Fidéliser** | Programme VIP, produits premium (vins, viandes), éditions limitées, pas de remise systématique | Catalogue, courrier personnalisé, événements en magasin |
| 2 | Familles actives | **Faire monter en valeur** | Offres groupées, programme de fidélité, vente croisée vers la viande et le premium | Site web, e-mail |
| 3 | Petits budgets familiaux | **Maintenir à moindre coût** | Offres d'appel, codes promotionnels, conversion des visites web en achats | E-mail, affichage en magasin |

## Limites
- **Structure modérée** (silhouette de 0,30) : les segments se chevauchent, surtout le segment intermédiaire. Certains clients sont proches de la frontière entre deux groupes.
- **Hypothèses à valider :** les besoins et les stratégies sont déduits du comportement d'achat. Il faudrait les tester (test A/B, enquête) avant de les déployer.
- **Dépenses et non marge :** le jeu de données ne contient pas de coûts. Le potentiel commercial est mesuré en dépenses.
- **Données d'exercice :** jeu public, sans date de collecte précise ni information sur l'entreprise d'origine.

## Lancer le notebook

**Google Colab (le plus simple)**
1. Cliquez sur le badge « Open in Colab » en haut de cette page.
2. Exécutez les cellules dans l'ordre. Les données sont lues directement depuis ce dépôt GitHub, rien à téléverser.

**En local**
```bash
git clone https://github.com/asaphmbere18-hub/segmentation-marketing-clients.git
cd segmentation-marketing-clients
pip install -r requirements.txt
jupyter notebook Segmentation_Marketing_Clients_Portfolio.ipynb
```

## Structure du dépôt
```
├── Segmentation_Marketing_Clients_Portfolio.ipynb   # notebook complet, avec les résultats
├── marketing_campaign.csv                           # données (2 240 clients, séparateur tabulation)
├── images/                                          # graphiques repris dans ce README
├── requirements.txt
└── README.md
```

## Outils
Python · pandas · NumPy · scikit-learn (StandardScaler, PCA, KMeans, TSNE, silhouette_score) · SciPy · Matplotlib · Seaborn

## Auteur
**Asaphs Mbere Itoua Ibara**, étudiant en Bachelor IA, Data et Management, en recherche d'alternance Data Analyst.
[Portfolio](https://asaphmbere18-hub.github.io) · [LinkedIn](https://www.linkedin.com/in/asaphs-mbere/) · [GitHub](https://github.com/asaphmbere18-hub)
