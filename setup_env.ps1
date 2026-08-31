# Reconstruit l'environnement Python du projet de zero (Windows / PowerShell).
# Utilise `uv` (https://astral.sh/uv) pour installer un Python 3.11 isolé,
# indépendant de tout Python déjà présent sur la machine, afin d'éviter
# les conflits de version.
#
# Usage: powershell -ExecutionPolicy Bypass -File setup_env.ps1

$ErrorActionPreference = "Stop"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Output "Installation de uv (gestionnaire d'environnements Python)..."
    irm https://astral.sh/uv/install.ps1 | iex
}

Write-Output "Installation de Python 3.11 (isolé, ne modifie pas votre Python système)..."
uv python install 3.11

Write-Output "Création de l'environnement virtuel .venv..."
uv venv --python 3.11 .venv

Write-Output "Installation des dépendances (requirements.txt)..."
uv pip install --python .venv -r requirements.txt

Write-Output "Enregistrement du kernel Jupyter 'projet_ia_assurance'..."
& ".venv\Scripts\python.exe" -m ipykernel install --user --name=projet_ia_assurance --display-name "Python (projet IA assurance)"

Write-Output ""
Write-Output "Terminé. Ouvrez le notebook et sélectionnez le kernel 'Python (projet IA assurance)':"
Write-Output "  .venv\Scripts\jupyter notebook notebooks\projet_tarification_anomalies_MAIRLOT_Antony.ipynb"
