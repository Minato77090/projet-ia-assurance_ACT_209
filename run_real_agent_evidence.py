"""
Capture d'une preuve d'execution reelle des agents LLM du notebook
(Section 8.3 - agent de scoring de gravite, Section 8.4 - agent de triage
avec tool-calling), sur des donnees reelles de data/raw/train.csv.

Ce script reproduit fidelement le code deja present dans le notebook
(cellules 64, 67, 70, 71, 72, 74) et ne fait qu'executer, pour de vrai,
la partie jusqu'ici gatee derriere OPENAI_API_KEY.

Usage (depuis la racine du projet, avec le venv active) :
    $env:OPENAI_API_KEY = "sk-..."
    .venv\\Scripts\\python.exe run_real_agent_evidence.py

Aucune sortie n'est fabriquee. Tout ce qui suit vient d'un vrai appel API.
La cle n'est jamais ecrite dans un fichier : uniquement lue depuis la
variable d'environnement OPENAI_API_KEY.
"""
import os
import json
from datetime import datetime, timezone

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import cohen_kappa_score
from openai import OpenAI

RANDOM_STATE = 42
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "docs", "agent_evidence")
os.makedirs(OUT_DIR, exist_ok=True)

assert os.environ.get("OPENAI_API_KEY"), "OPENAI_API_KEY doit etre defini dans l'environnement"
client = OpenAI()

# ---------------------------------------------------------------------------
# 1. Chargement des donnees reelles
# ---------------------------------------------------------------------------
claim_df = pd.read_csv(os.path.join(ROOT, "data", "raw", "train.csv"))
print(f"Donnees chargees : {len(claim_df)} sinistres")

# ---------------------------------------------------------------------------
# 2. Heuristique lexicale de gravite (copie exacte de la cellule 64)
# ---------------------------------------------------------------------------
SEVERITY_KEYWORDS = {
    3: ['fracture', 'amputation', 'crush', 'burn', 'head', 'spine', 'death', 'severe', 'multiple'],
    2: ['tear', 'sprain', 'strain', 'dislocation', 'laceration', 'back', 'shoulder', 'knee'],
    1: ['bruise', 'contusion', 'abrasion', 'minor', 'sore'],
}


def severity_heuristic(description: str) -> int:
    text = str(description).lower()
    for level in (3, 2, 1):
        if any(keyword in text for keyword in SEVERITY_KEYWORDS[level]):
            return level
    return 1


claim_df['severity_score_heuristic'] = claim_df['ClaimDescription'].apply(severity_heuristic)

# ---------------------------------------------------------------------------
# 3. Isolation Forest reel sur variables numeriques (detecteur d'anomalies)
# ---------------------------------------------------------------------------
numeric_cols = ['Age', 'DependentChildren', 'DependentsOther', 'WeeklyWages',
                 'HoursWorkedPerWeek', 'DaysWorkedPerWeek', 'InitialIncurredCalimsCost']
X_if = claim_df[numeric_cols].fillna(claim_df[numeric_cols].median())
iso = IsolationForest(n_estimators=200, contamination=0.02, random_state=RANDOM_STATE)
if_flag = (iso.fit_predict(X_if) == -1).astype(int)
claim_df['if_flag'] = if_flag
claim_df['if_score'] = -iso.score_samples(X_if)  # plus grand = plus anormal
print(f"Isolation Forest reel entraine : {if_flag.sum()} sinistres flagues anormaux sur {len(claim_df)}")

# ---------------------------------------------------------------------------
# 4. Agent de scoring de gravite (Section 8.3, cellule 67) - 20 sinistres reels
# ---------------------------------------------------------------------------
AGENT_SYSTEM_PROMPT = (
    "En tant qu'expert medical en assurance accidents du travail, analyse cette "
    "description de sinistre et classe la gravite potentielle sur une echelle de 1 a 3 "
    "(1 = leger, 2 = modere, 3 = severe). Reponds uniquement par le chiffre."
)


def llm_severity_agent(description: str) -> int:
    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[
            {'role': 'system', 'content': AGENT_SYSTEM_PROMPT},
            {'role': 'user', 'content': description},
        ],
        temperature=0,
        max_tokens=2,
    )
    return int(response.choices[0].message.content.strip()[0])


sample = claim_df.sample(20, random_state=RANDOM_STATE).copy()
severity_records = []
print("\n=== Agent de scoring de gravite (Section 8.3) - 20 sinistres reels ===")
for _, row in sample.iterrows():
    llm_score = llm_severity_agent(row['ClaimDescription'])
    severity_records.append({
        'claim_number': str(row['ClaimNumber']),
        'description': row['ClaimDescription'],
        'severity_heuristic': int(row['severity_score_heuristic']),
        'llm_severity': llm_score,
    })
    print(f"  {row['ClaimNumber']}: heuristique={row['severity_score_heuristic']}  LLM_reel={llm_score}  | {row['ClaimDescription'][:70]}")

severity_df = pd.DataFrame(severity_records)
kappa_llm_heuristic = cohen_kappa_score(severity_df['llm_severity'], severity_df['severity_heuristic'])
print(f"\nKappa de Cohen (LLM reel vs heuristique lexicale) sur ces 20 sinistres : {kappa_llm_heuristic:.3f}")

# ---------------------------------------------------------------------------
# 5. Agent de triage avec tool-calling (Section 8.4, cellules 70-74)
# ---------------------------------------------------------------------------
TOOLS_TRIAGE = [
    {
        "type": "function",
        "function": {
            "name": "get_claim_signals",
            "description": (
                "Recupere les signaux quantitatifs deja calcules pour un sinistre "
                "(flag Isolation Forest, score d'anomalie, cout initial provisionne, "
                "score de gravite heuristique). A utiliser avant de decider si une "
                "escalade est justifiee."
            ),
            "parameters": {
                "type": "object",
                "properties": {"claim_number": {"type": "string", "description": "Identifiant ClaimNumber du sinistre"}},
                "required": ["claim_number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "flag_for_manual_review",
            "description": (
                "Signale formellement un sinistre pour revue humaine. A appeler "
                "uniquement si l'analyse (description + signaux recuperes) justifie "
                "une escalade - pas systematiquement."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "claim_number": {"type": "string"},
                    "reason": {"type": "string", "description": "Justification courte de l'escalade"},
                    "priority": {"type": "string", "enum": ["low", "medium", "high"]},
                },
                "required": ["claim_number", "reason", "priority"],
            },
        },
    },
]

ESCALATIONS_LOG = []


def execute_triage_tool(function_name, arguments, signals_lookup):
    if function_name == "get_claim_signals":
        claim_number = arguments["claim_number"]
        signals = signals_lookup.get(claim_number)
        if signals is None:
            return f"Aucun signal disponible pour le sinistre {claim_number}."
        return (
            f"Flag Isolation Forest : {'anormal' if signals['if_flag'] else 'normal'}\n"
            f"Score d'anomalie Isolation Forest (plus eleve = plus anormal) : {signals['if_score']:.3f}\n"
            f"Cout initialement provisionne : {signals['initial_cost']:.0f} $\n"
            f"Score de gravite heuristique (1-3) : {signals['severity_heuristic']}"
        )
    elif function_name == "flag_for_manual_review":
        ESCALATIONS_LOG.append(arguments)
        return f"Sinistre {arguments['claim_number']} enregistre pour revue humaine (priorite {arguments['priority']})."
    return f"Fonction inconnue : {function_name}"


def run_triage_agent_loop(system_prompt, user_prompt, tools, signals_lookup, client,
                           model="gpt-4o-mini", max_iterations=3, verbose=True):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    tool_calls_log = []
    total_tokens = {"input": 0, "output": 0}

    for iteration in range(max_iterations):
        response = client.chat.completions.create(model=model, messages=messages, tools=tools)
        total_tokens["input"] += response.usage.prompt_tokens
        total_tokens["output"] += response.usage.completion_tokens
        assistant_msg = response.choices[0].message

        if not assistant_msg.tool_calls:
            return {"content": assistant_msg.content, "tool_calls_log": tool_calls_log,
                    "iterations": iteration + 1, "total_tokens": total_tokens}

        messages.append(assistant_msg)
        for tc in assistant_msg.tool_calls:
            args = json.loads(tc.function.arguments)
            if verbose:
                print(f"   -> {tc.function.name}({args})")
            result = execute_triage_tool(tc.function.name, args, signals_lookup)
            tool_calls_log.append({"iteration": iteration + 1, "function": tc.function.name,
                                    "arguments": args, "result": result})
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    return {"content": assistant_msg.content or "[Agent interrompu - max iterations]",
            "tool_calls_log": tool_calls_log, "iterations": max_iterations, "total_tokens": total_tokens}


TRIAGE_SYSTEM_PROMPT = (
    "Tu es un expert senior en triage de sinistres accidents du travail. On te donne "
    "uniquement la description libre d'un sinistre. Utilise l'outil get_claim_signals "
    "pour recuperer les signaux quantitatifs disponibles AVANT de te prononcer. "
    "Decide ensuite, en croisant la description et les signaux, si le dossier justifie "
    "une escalade vers un gestionnaire humain ; si oui, appelle flag_for_manual_review "
    "avec une justification courte et une priorite. N'escalade pas systematiquement : "
    "seuls les dossiers reellement atypiques doivent l'etre."
)

anomalous = claim_df[claim_df['if_flag'] == 1]
triage_sample = anomalous.sample(min(15, len(anomalous)), random_state=RANDOM_STATE)

signals_lookup = {}
for _, row in triage_sample.iterrows():
    signals_lookup[str(row['ClaimNumber'])] = {
        'if_flag': int(row['if_flag']),
        'if_score': float(row['if_score']),
        'initial_cost': float(row['InitialIncurredCalimsCost']),
        'severity_heuristic': int(row['severity_score_heuristic']),
    }

triage_results = []
print("\n=== Agent de triage avec tool-calling (Section 8.4) ===")
for _, row in triage_sample.iterrows():
    claim_number = str(row['ClaimNumber'])
    description = row['ClaimDescription']
    print(f"\n--- Sinistre {claim_number} ---")
    result = run_triage_agent_loop(
        system_prompt=TRIAGE_SYSTEM_PROMPT,
        user_prompt=f"ClaimNumber: {claim_number}\nDescription: {description}",
        tools=TOOLS_TRIAGE,
        signals_lookup=signals_lookup,
        client=client,
        max_iterations=3,
    )
    print(result['content'])
    triage_results.append({
        'claim_number': claim_number,
        'description': description,
        'iterations': result['iterations'],
        'n_tool_calls': len(result['tool_calls_log']),
        'tool_calls_log': result['tool_calls_log'],
        'final_content': result['content'],
        'tokens_in': result['total_tokens']['input'],
        'tokens_out': result['total_tokens']['output'],
    })

triage_results_df = pd.DataFrame([{k: v for k, v in r.items() if k not in ('tool_calls_log', 'description', 'final_content')} for r in triage_results])
print("\nResume (n={} sinistres) :".format(len(triage_results_df)))
print(triage_results_df.describe())
print(f"\nSinistres escalades par l'agent : {len(ESCALATIONS_LOG)} / {len(triage_sample)}")
for esc in ESCALATIONS_LOG:
    print(' -', esc)

# ---------------------------------------------------------------------------
# 6. Sauvegarde du log complet (aucune cle API dans les fichiers)
# ---------------------------------------------------------------------------
evidence = {
    'generated_at_utc': datetime.now(timezone.utc).isoformat(),
    'model': 'gpt-4o-mini',
    'scope_note': (
        "Capture d'evidence ciblee : Isolation Forest reel (numerique) au lieu de "
        "l'autoencodeur pour la selection des sinistres anormaux, heuristique lexicale "
        "au lieu du zero-shot embeddings pour le signal de gravite dans signals_lookup. "
        "Tous les appels au modele LLM (scoring et triage) sont reels."
    ),
    'severity_agent': {
        'n_samples': len(severity_df),
        'kappa_llm_vs_heuristic': round(float(kappa_llm_heuristic), 3),
        'records': severity_records,
    },
    'triage_agent': {
        'n_samples': len(triage_results),
        'n_escalations': len(ESCALATIONS_LOG),
        'escalations': ESCALATIONS_LOG,
        'total_tokens_in': int(triage_results_df['tokens_in'].sum()),
        'total_tokens_out': int(triage_results_df['tokens_out'].sum()),
        'avg_iterations': round(float(triage_results_df['iterations'].mean()), 2),
        'avg_tool_calls': round(float(triage_results_df['n_tool_calls'].mean()), 2),
        'results': triage_results,
    },
}

out_path = os.path.join(OUT_DIR, "agent_execution_evidence.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(evidence, f, ensure_ascii=False, indent=2)

print(f"\nPreuve d'execution sauvegardee : {out_path}")
