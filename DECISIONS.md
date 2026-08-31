# Registre des Décisions Méthodologiques (Decision Log)

**Projet** : Tarification et détection d'anomalies sur les sinistres (ACT209)  
**Auteur** : Antony MAIRLOT  

Ce registre documente les choix architecturaux, statistiques et technologiques effectués tout au long du projet. Conformément aux standards de développement en Data Science (Architecture Decision Records), son objectif est de tracer les alternatives écartées, de justifier les choix retenus et d'assumer de manière transparente les limites techniques et conceptuelles du modèle, dans une perspective d'amélioration continue.

---

## 1. Cadrage général et Environnement

**Décision : Choix du jeu de données "Kaggle Actuarial Loss Estimation"**
*   **Alternatives écartées :** Jeux de données publics classiques (French Motor Claims), sujets purement NLP, sujets purement ERM.
*   **Justification :** Volumétrie robuste (54 000 lignes) et présence simultanée de variables numériques, catégorielles et de texte libre. Cela permet de concevoir une architecture hybride (Machine Learning, Deep Learning, IA Générative) répondant conjointement aux problématiques de tarification et de détection d'anomalies.
*   **Limite assumée :** Le jeu de données est ancien (1988-2007) et partiellement synthétisé. Il est traité ici comme un cas d'école méthodologique pour valider une architecture, et non comme un portefeuille réel prêt à être mis en production sans audit complémentaire.

**Décision : Stack Python 3.11 avec environnement isolé via `uv`**
*   **Alternatives écartées :** Environnement Python système, gestionnaires lourds type Conda.
*   **Justification :** `uv` garantit une résolution de dépendances extrêmement rapide et isole strictement le projet. Python 3.11 offre la meilleure stabilité actuelle pour l'écosystème ML/DL combiné (TensorFlow, XGBoost, HDBSCAN).
*   **Limite assumée :** Les dépendances sont strictement figées (`requirements.txt`), le pipeline n'est pas conteneurisé via Docker pour le moment.

---

## 2. Préparation et Qualité des données

**Décision : Imputation de la variable `MaritalStatus` par une catégorie explicite `"Missing"`**
*   **Alternative écartée :** Imputation par le mode (SimpleImputer) ou imputation itérative multivariée.
*   **Justification :** En assurance, une valeur manquante n'est presque jamais MCAR (*Missing Completely At Random*). Une absence de saisie peut signaler un sinistre grave (urgence vitale) ou une déclaration atypique. Forcer ces observations vers la modalité majoritaire détruit ce signal. XGBoost traite nativement cette catégorie isolée.
*   **Limite assumée :** La taille de cet échantillon (n=29) offre une faible puissance statistique, rendant le test de Mann-Whitney non concluant au seuil de 5 %.

**Décision : Traitement des valeurs aberrantes en deux étapes (Règles métier + Winsorisation)**
*   **Alternative écartée :** Winsorisation aveugle ou exclusion des algorithmes sensibles.
*   **Justification :** Le plafonnement de `HoursWorkedPerWeek` à 98h repose sur une contrainte physique (14h/jour). La winsorisation aux 1er et 99e centiles traite ensuite les extrêmes plausibles. L'ablation documentée dans le notebook prouve empiriquement que XGBoost absorbe ce bruit natif (impact marginal sur le RMSE de 0,04 %), mais ce nettoyage est maintenu pour stabiliser l'autoencodeur et fiabiliser l'analyse exploratoire.

---

## 3. Modélisation et Optimisation de la Cible (Flux 1)

**Décision : Choix de la métrique d'évaluation (RMSE)**
*   **Alternative écartée :** Optimisation d'une fonction de perte Tweedie ou Pseudo-Huber.
*   **Justification :** L'optimisation quadratique (MSE/RMSE) a été conservée uniquement pour garantir une comparabilité directe (en dollars nominaux) avec le GLM Gamma de référence et les benchmarks standards.
*   **Limite assumée (Sous-optimum actuariel) :** La cible présente un coefficient d'asymétrie (skewness) extrême de 37,6. Optimiser un RMSE sur une telle distribution force mécaniquement l'arbre de décision à "chasser" les valeurs extrêmes plutôt qu'à tarifer correctement la masse des sinistres. Une cible de production exigerait la fonction objective `reg:tweedie` dans XGBoost.

**Décision : Prédiction transversale (Cross-Sectional) sur l'Ultimate Cost**
*   **Alternative écartée :** Modèles de développement longitudinaux (Chain-Ladder ML, Réseaux récurrents LSTM).
*   **Justification :** Limitation de l'approche supervisée standard sur un jeu de données "aplati".
*   **Limite assumée (Biais de maturité) :** Prédire le coût ultime sur un historique s'étalant de 1988 à 2007 sans intégrer explicitement une variable de *development lag* (maturité du sinistre) sous-évalue structurellement les sinistres récents, qui n'ont pas terminé leur développement. 

**Décision : Utilisation de l'échantillonneur bayésien TPE (Optuna)**
*   **Alternative écartée :** GridSearchCV exhaustif.
*   **Justification :** L'optimisation bayésienne permet d'explorer un espace continu de 6 hyperparamètres en concentrant le budget de calcul sur les régions les plus prometteuses, avec un taux de convergence optimal sur 40 essais.

---

## 4. Apprentissage Profond et Détection d'Anomalies (Flux 2)

**Décision : Architecture Autoencodeur Dense 64-32-16-32-64 avec perte MSE**
*   **Alternative écartée :** Autoencodeur Variationnel (VAE) ou fonction de perte hybride.
*   **Justification :** Architecture symétrique classique suffisante pour construire un MVP (*Minimum Viable Product*) de détection non supervisée sur des données tabulaires.
*   **Limite assumée :** L'application d'une perte quadratique (MSE) sur des variables catégorielles encodées en *one-hot* constitue une limite mathématique. L'erreur de reconstruction perd son interprétabilité probabiliste sur les variables binaires. Une perte combinée (MSE pour les continus + Binary Cross-Entropy pour le one-hot) serait indispensable en production.

**Décision : Forçage du déterminisme TensorFlow**
*   **Justification :** Sans fixation des *seeds* au niveau des opérations mathématiques (`tf.config.experimental.enable_op_determinism`), les poids d'initialisation de l'autoencodeur causaient une forte variance des scores de reconstruction, compromettant la reproductibilité revendiquée du projet. Le bug a été identifié et corrigé pour garantir la constance du score de Jaccard par rapport à Isolation Forest.

---

## 5. NLP, Sécurité et IA Générative (Flux 3)

**Décision : Utilisation d'un modèle de langage pour l'Agent de Triage via l'API OpenAI**
*   **Alternative écartée :** Small Language Model (SLM) auto-hébergé de type Llama 3 8B.
*   **Justification :** Démonstration de faisabilité technique d'un cadre agentique réel (appel dynamique d'outils) avec un budget d'infrastructure nul.
*   **Limite assumée (Violation stricte du RGPD) :** La variable `ClaimDescription` contient des informations de santé au sens de l'Article 9 du RGPD. Exécuter un appel vers une API tierce hébergée hors UE sur ce type de donnée est juridiquement inenvisageable en production. L'architecture cible imposerait un déploiement *On-Premise* ou une anonymisation stricte par un modèle NER local en amont.

**Décision : PCA sur les embeddings générés par 'all-MiniLM-L6-v2'**
*   **Alternative écartée :** ClinicalBERT (pour l'embedding) et UMAP (pour la réduction).
*   **Justification :** Sobriété computationnelle sur CPU.
*   **Limite assumée :** MiniLM est un modèle généraliste qui ne capte que partiellement le vocabulaire biomécanique des sinistres AT. De plus, réduire des embeddings par PCA (linéaire) écrase une grande partie de la topologie sémantique non-linéaire du corpus.

---

## 6. Audit d'Explicabilité

**Décision : Requalification de la "Fuite de données" en "Biais Algorithmique"**
*   **Justification :** L'importance native (critère de *Gain*) de XGBoost plaçait la variable `ReportDelayDays` en tête. Ce n'est pas un *Data Leakage* au sens strict, puisque cette variable est calculable dès le jour 1 de la déclaration. L'audit croisé avec SHAP et l'importance par permutation a révélé qu'il s'agissait du biais algorithmique classique des arbres CART, qui favorisent mécaniquement les variables continues présentant un grand nombre de points de coupures possibles, indépendamment de leur valeur causale ou prédictive. Un retrait de ces variables en production est recommandé (effet chiffré par ablation : RMSE quasi stable, léger gain de -0,84 %) ; elles restent toutefois présentes dans le modèle rapporté (Table 4) - seul le retrait est recommandé, il n'a pas été appliqué au résultat final.
