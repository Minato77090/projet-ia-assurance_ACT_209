# Tarification et détection d'anomalies sur les sinistres

**Approche hybride XGBoost, Autoencodeurs (DL) et Analyse Agentique (LLM)**
Projet réalisé dans le cadre de l'UE ACT209 (Intelligence Artificielle appliquée à l'Assurance) – Master 2 Actuariat, CNAM.

Ce dépôt contient l'intégralité du code source, de la documentation méthodologique et des livrables associés à la modélisation du jeu de données [Actuarial Loss Estimation](https://www.kaggle.com/competitions/actuarial-loss-estimation/data) (54 000 polices d'assurance accidents du travail).

---

## Structure du dépôt

```text
projet_IA_assurance/
├── data/
│   └── raw/
│       ├── train.csv                    # Jeu d'entraînement (Kaggle Actuarial Loss Estimation)
│       ├── test.csv
│       └── sample_submission.csv
├── notebooks/
│   └── projet_tarification_anomalies_MAIRLOT_Antony.ipynb   # Notebook principal (livrable)
├── report/
│   ├── Rapport ACT 209 MAIRLOT Antony.pdf                     # Rapport officiel (livrable, 15 pages)
│   ├── Presentation_Projet_IA_Assurance MAIRLOT Antony.pptx   # Support de soutenance (livrable, 16 slides)
│   ├── extract_figures.py     # Extraction des figures du notebook (base64 vers PNG)
│   ├── make_report_charts.py  # Génération des graphiques de synthèse (RMSE, ablation)
│   └── figures/                # Figures utilisées par le rapport et la présentation
├── DECISIONS.md                # Journal des choix méthodologiques (Architecture Decision Record)
├── run_real_agent_evidence.py  # Exécution isolée et journalisation des agents LLM
├── setup_env.ps1               # Script d'installation isolée de l'environnement (uv / venv)
├── requirements.txt            # Dépendances figées du projet
└── README.md
```

## 1. Données

Le jeu de données [Actuarial Loss Estimation](https://www.kaggle.com/competitions/actuarial-loss-estimation/data) (54 000 polices d'assurance accidents du travail) est fourni directement dans `data/raw/`.

## 2. Configuration de l'environnement

L'environnement repose sur le gestionnaire `uv` pour un déploiement déterministe sous Python 3.11. Pour initialiser l'environnement virtuel (`.venv`) et le kernel Jupyter :

```powershell
powershell -ExecutionPolicy Bypass -File setup_env.ps1
```

## 3. Exécution du pipeline (notebook)

Lancer Jupyter depuis l'environnement virtuel :

```powershell
.venv\Scripts\jupyter notebook notebooks\projet_tarification_anomalies_MAIRLOT_Antony.ipynb
```

Vérifier que le kernel actif est bien **« Python (projet IA assurance) »**.

Résumé des étapes du pipeline :

- **Prétraitement** : gestion des valeurs manquantes (`MaritalStatus` en catégorie `Missing` validée par test de Mann-Whitney) et traitement des valeurs aberrantes (règles métier sur `HoursWorkedPerWeek` et winsorisation).
- **EDA** : corrélations non linéaires (Spearman, information mutuelle, PPS), test de Kruskal-Wallis sur les catégorielles, et correction actuarielle de l'inflation (CPI Medical Care).
- **Modélisation** :
  - Flux 1 (ML supervisé) : XGBoost optimisé par Optuna (TPE Sampler), comparé à un GLM Gamma de référence et à LightGBM.
  - Flux 2 (DL non supervisé) : autoencodeur Dense pour la détection d'anomalies (comparé à Isolation Forest).
  - Flux 3 (NLP / agentique) : embeddings SentenceTransformer, clustering HDBSCAN, score de gravité zero-shot et agent LLM à appel d'outils (function calling).
- **Explicabilité & audit** : valeurs SHAP, importance par permutation et audit du biais du critère de gain de l'arbre CART sur les variables de déclaration.

## 4. Rapport et présentation

Le rapport (`report/Rapport ACT 209 MAIRLOT Antony.pdf`) et le support de soutenance (`report/Presentation_Projet_IA_Assurance MAIRLOT Antony.pptx`) sont fournis directement en l'état. Les graphiques de synthèse qu'ils utilisent (comparaison RMSE, ablation, coût des anomalies) peuvent être régénérés après une nouvelle exécution du notebook :

```powershell
.venv\Scripts\python.exe report\extract_figures.py
.venv\Scripts\python.exe report\make_report_charts.py
```

## 5. Sécurité et démonstration agentique

Aucune clé API n'est stockée dans le code source (le flux agentique LLM est désactivé par défaut). Pour tester l'appel de l'agent LLM réel sur l'échantillon d'anomalies :

1. Configurer la variable d'environnement : `$env:OPENAI_API_KEY = "sk-..."`
2. Exécuter le script de traçabilité :

```powershell
.venv\Scripts\python.exe run_real_agent_evidence.py
```

Les journaux d'exécution sont écrits dans `docs/agent_evidence/`.

## 6. Traçabilité méthodologique

L'ensemble des choix techniques, des alternatives écartées, des limites conceptuelles (choix du RMSE vs Tweedie, absence de variable de maturité/development lag, conformité RGPD de l'API LLM) ainsi que la résolution des bugs de reproductibilité sont documentés de manière exhaustive dans `DECISIONS.md`.
