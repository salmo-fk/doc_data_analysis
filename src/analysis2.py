# Build:
# 1) Disease profile selector (HTML with dropdown, dynamic Plotly + text)
# 2) Single HTML report containing all figures inline (self-contained)
# 3) Symptom family taxonomy (heuristic), + family analysis plots, + standalone HTML
import json
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

# -------------------------
# Load analysis-ready table
# -------------------------
long_df = pd.read_csv("/mnt/data/analysis_ready_long.csv")

# -------------------------
# 1) Disease profile selector
# -------------------------
def disease_profile_payload(df: pd.DataFrame):
    payload = {}
    for disease, g in df.groupby("Disease"):
        # Description
        desc = g["Description"].dropna().unique()
        desc = desc[0] if len(desc) else "(Pas de description)"
        # Precautions (first non-null row across precaution columns)
        precautions_cols = [c for c in g.columns if str(c).lower().startswith("precaution")]
        row = g[precautions_cols].dropna(how="all").head(1)
        precautions = [str(v) for v in row.iloc[0].dropna().tolist()] if not row.empty else []
        # Symptom severities
        sym = (g.dropna(subset=["Symptom"])
               .groupby("Symptom")["weight"].max()
               .sort_values(ascending=False))
        payload[disease] = {
            "description": desc,
            "precautions": precautions,
            "symptoms": [{"name": s, "weight": float(w) if pd.notnull(w) else None} for s, w in sym.items()]
        }
    return payload

profile_data = disease_profile_payload(long_df)
profile_html_path = Path("/mnt/data/disease_profile_selector.html")

template = f"""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="utf-8"/>
  <title>Profil de maladie — Sélecteur interactif</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 24px; }}
    .layout {{ display: grid; grid-template-columns: 320px 1fr; gap: 24px; }}
    .card {{ border: 1px solid #eee; border-radius: 12px; padding: 16px; box-shadow: 0 2px 6px rgba(0,0,0,0.04); }}
    h1 {{ margin-top: 0; }}
    h2 {{ margin: 8px 0 12px; font-size: 18px; }}
    select {{ width: 100%; padding: 8px; border-radius: 8px; border: 1px solid #ddd; }}
    ul {{ margin: 8px 0 0 18px; }}
    .muted {{ color: #666; }}
  </style>
</head>
<body>
  <h1>Profil interactif d'une maladie</h1>
  <div class="layout">
    <div class="card">
      <h2>Choisir une maladie</h2>
      <select id="diseaseSelect"></select>
      <div style="margin-top:16px">
        <h2>Description</h2>
        <div id="desc" class="muted">(sélectionne une maladie)</div>
        <h2 style="margin-top:16px">Précautions</h2>
        <ul id="precautions"></ul>
      </div>
    </div>
    <div class="card">
      <h2>Symptômes triés par sévérité</h2>
      <div id="chart" style="height:520px;"></div>
    </div>
  </div>
  <script>
    const DATA = {json.dumps(profile_data)};
    const diseases = Object.keys(DATA).sort();
    const select = document.getElementById('diseaseSelect');
    diseases.forEach(d => {{
      const opt = document.createElement('option');
      opt.value = d; opt.textContent = d;
      select.appendChild(opt);
    }});
    function render(disease) {{
      const d = DATA[disease];
      document.getElementById('desc').textContent = d.description || '(Pas de description)';
      const ul = document.getElementById('precautions');
      ul.innerHTML = '';
      if (d.precautions && d.precautions.length) {{
        d.precautions.forEach(p => {{
          const li = document.createElement('li'); li.textContent = p; ul.appendChild(li);
        }});
      }} else {{
        const li = document.createElement('li'); li.textContent = '(Aucune précaution disponible)'; ul.appendChild(li);
      }}
      const x = d.symptoms.map(s => s.name);
      const y = d.symptoms.map(s => s.weight);
      const trace = {{
        type: 'bar', x: x, y: y,
        hovertemplate: '%{x}: %{y}<extra></extra>'
      }};
      const layout = {{
        margin: {{t: 10, r: 10, b: 120, l: 50}},
        xaxis: {{tickangle: -45, title: 'Symptôme'}},
        yaxis: {{title: 'Poids (sévérité)', rangemode: 'tozero'}},
      }};
      Plotly.newPlot('chart', [trace], layout, {{responsive: true}});
    }}
    select.addEventListener('change', (e) => render(e.target.value));
    if (diseases.length) {{ select.value = diseases[0]; render(diseases[0]); }}
  </script>
</body>
</html>
"""
profile_html_path.write_text(template, encoding="utf-8")

# -------------------------
# 2) Single HTML report with embedded figures
# -------------------------
def fig_json(fig):
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder) if 'plotly' in globals() else fig.to_json()

# Re-create key figures inline (fresh) to embed
# Plot 1 (sorted by weight 1->5)
sev_counts = long_df.groupby("weight", dropna=False)["Symptom"].nunique().reset_index()
sev_counts = sev_counts.rename(columns={"Symptom": "Nombre_de_symptomes"}).dropna(subset=["weight"]).sort_values("weight")
fig1 = px.bar(sev_counts, x="weight", y="Nombre_de_symptomes",
              title="Distribution des symptômes par sévérité — triée par poids (1→5)",
              labels={"weight": "Poids (sévérité)", "Nombre_de_symptomes": "Nombre de symptômes"})

# Plot 2 (multi-color per average severity)
per_disease = (long_df.groupby("Disease")
               .agg(Nombre_de_symptomes=("Symptom","nunique"),
                    Severite_moyenne=("weight","mean"))
               .reset_index()
               .sort_values("Nombre_de_symptomes", ascending=False))
fig2 = px.bar(per_disease.head(25), x="Disease", y="Nombre_de_symptomes",
              color="Severite_moyenne",
              title="Top 25 maladies par nombre de symptômes (couleur = sévérité moyenne)")
fig2.update_layout(xaxis_tickangle=-45)

# Plot 3 heatmap disease x symptom (weights)
top_dis = per_disease.head(20)["Disease"].tolist()
common_symptoms = (long_df.groupby("Symptom")["Disease"].nunique()
                   .sort_values(ascending=False).head(30).index.tolist())
hm = (long_df[long_df["Disease"].isin(top_dis) & long_df["Symptom"].isin(common_symptoms)]
      .groupby(["Disease", "Symptom"])["weight"].max().reset_index())
heatmap_pivot = hm.pivot(index="Disease", columns="Symptom", values="weight").fillna(0)
fig3 = go.Figure(data=go.Heatmap(
    z=heatmap_pivot.values, x=list(heatmap_pivot.columns), y=list(heatmap_pivot.index),
    colorbar=dict(title="Poids")))
fig3.update_layout(title="Carte thermique (Disease × Symptom) — poids")

# Plot A: scatter regression
x = per_disease["Nombre_de_symptomes"].values.astype(float)
y = per_disease["Severite_moyenne"].values.astype(float)
coeffs = np.polyfit(x, y, 1) if len(per_disease) >= 2 else [0, float(np.nan)]
reg_line = coeffs[0]*x + coeffs[1]
figA = go.Figure()
figA.add_trace(go.Scatter(x=x, y=y, mode="markers", text=per_disease["Disease"], name="Maladies"))
figA.add_trace(go.Scatter(x=x, y=reg_line, mode="lines",
                          name=f"Régression linéaire (y={coeffs[0]:.2f}x+{coeffs[1]:.2f})"))
figA.update_layout(title="Corrélation: Nb symptômes vs Sévérité moyenne",
                   xaxis_title="Nb de symptômes", yaxis_title="Sévérité moyenne")

# Plot B: histogram + box
figB = px.histogram(per_disease, x="Severite_moyenne", nbins=20, marginal="box",
                    title="Distribution de la sévérité moyenne (hist + box)")

# Plot C: top symptoms by diseases
symptom_breadth = (long_df.groupby("Symptom")["Disease"]
                   .nunique().reset_index()
                   .rename(columns={"Disease":"Nombre_de_maladies"})
                   .sort_values("Nombre_de_maladies", ascending=False))
figC = px.bar(symptom_breadth.head(30), x="Symptom", y="Nombre_de_maladies",
              title="Top symptômes par nb de maladies")
figC.update_layout(xaxis_tickangle=-45)

# Plot D: co-occurrence heatmap (top 25)
top_symptoms = symptom_breadth.head(25)["Symptom"].tolist()
cooc = pd.DataFrame(0, index=top_symptoms, columns=top_symptoms, dtype=int)
for disease, g in long_df.groupby("Disease"):
    syms = [s for s in g["Symptom"].dropna().unique().tolist() if s in top_symptoms]
    for i in range(len(syms)):
        for j in range(i, len(syms)):
            a,b = syms[i], syms[j]
            cooc.loc[a,b]+=1
            if a!=b: cooc.loc[b,a]+=1
figD = go.Figure(data=go.Heatmap(z=cooc.values, x=cooc.columns.tolist(), y=cooc.index.tolist(),
                                 colorbar=dict(title="Co-occurrences")))
figD.update_layout(title="Matrice de co-occurrence des symptômes (Top 25)")

# Plot H: top precautions
prec_cols = [c for c in long_df.columns if str(c).lower().startswith("precaution")]
prec_series = pd.Series(dtype=str)
for c in prec_cols:
    prec_series = pd.concat([prec_series, long_df[c].dropna().astype(str)])
prec_counts = (prec_series.str.strip().str.lower().value_counts().reset_index())
prec_counts.columns = ["precaution", "count"]
figH = px.bar(prec_counts.head(20), x="precaution", y="count",
              title="Top 20 précautions recommandées")
figH.update_layout(xaxis_tickangle=-45)

# -------------------------
# 3) Symptom Families (taxonomy)
# -------------------------
def guess_family(sym: str) -> str:
    s = str(sym).lower()
    # Dermatology
    if any(k in s for k in ["skin", "rash", "itch", "ulcer", "blister"]):
        return "Dermatologique"
    # Respiratory
    if any(k in s for k in ["cough", "breath", "sneeze", "wheez", "phlegm", "throat", "chest_pain"]):
        return "Respiratoire"
    # Gastrointestinal
    if any(k in s for k in ["nausea", "vomit", "diarr", "stomach", "abdominal", "appetite", "constipation"]):
        return "Gastro‑intestinal"
    # Neurological
    if any(k in s for k in ["headache", "dizziness", "loss_of_balance", "tremor", "seizure"]):
        return "Neurologique"
    # Systemic / Metabolic
    if any(k in s for k in ["fever", "fatigue", "weight_loss", "sweating", "malaise", "dehydration"]):
        return "Systémique"
    # Musculoskeletal
    if any(k in s for k in ["joint", "muscle", "back_pain", "knee", "hip", "bone"]):
        return "Musculo‑squelettique"
    # Uro/renal
    if any(k in s for k in ["urine", "urination", "dysuria"]):
        return "Urologique/Rénal"
    # Ophthalmology
    if any(k in s for k in ["eye", "vision", "conjunctivitis"]):
        return "Ophtalmologique"
    # ORL (ear, nose, throat) beyond above
    if any(k in s for k in ["ear", "nose", "sinus"]):
        return "ORL"
    return "Autre"

taxo_map = (long_df[["Symptom"]]
            .dropna().drop_duplicates()
            .assign(Famille=lambda d: d["Symptom"].map(guess_family)))
taxo_path = Path("/mnt/data/symptom_taxonomy.csv")
taxo_map.to_csv(taxo_path, index=False)

long_df_tax = long_df.merge(taxo_map, on="Symptom", how="left")

# Family plots
fam_counts = (long_df_tax.groupby(["Famille","Symptom"])["Disease"]
              .nunique().reset_index(name="Nombre_de_maladies"))
fam_sev = (long_df_tax.groupby(["Famille","Symptom"])["weight"]
           .mean().reset_index(name="Severite_moyenne"))

fam_merged = fam_counts.merge(fam_sev, on=["Famille","Symptom"], how="left")

figI = px.treemap(fam_merged, path=["Famille","Symptom"],
                  values="Nombre_de_maladies",
                  color="Severite_moyenne",
                  title="Treemap — Familles → Symptômes (taille = nb maladies, couleur = sévérité moy.)")

# Stacked bar per disease: counts per family
per_dis_fam = (long_df_tax.groupby(["Disease","Famille"])["Symptom"]
               .nunique().reset_index(name="Nb_symptomes"))
top_dis15 = (long_df.groupby("Disease")["Symptom"].nunique()
             .sort_values(ascending=False).head(15).index.tolist())
per_dis_fam_top = per_dis_fam[per_dis_fam["Disease"].isin(top_dis15)]

figJ = px.bar(per_dis_fam_top, x="Disease", y="Nb_symptomes", color="Famille",
              title="Top 15 maladies — répartition des symptômes par famille",
              barmode="relative")
figJ.update_layout(xaxis_tickangle=-45)

# Heatmap disease vs family (mean severity)
fam_heat = (long_df_tax.groupby(["Disease","Famille"])["weight"]
            .mean().reset_index())
fam_pivot = fam_heat.pivot(index="Disease", columns="Famille", values="weight").fillna(0)
figK = go.Figure(data=go.Heatmap(
    z=fam_pivot.values,
    x=list(fam_pivot.columns),
    y=list(fam_pivot.index),
    colorbar=dict(title="Sévérité moyenne"),
))
figK.update_layout(title="Carte thermique — Sévérité moyenne par famille de symptômes")

# -------------------------
# Compose single HTML report
# -------------------------
def to_div(fig, div_id):
    return f"<div id='{div_id}' style='height:520px;'></div>", f"""
    Plotly.newPlot("{div_id}", {fig.to_json()}, {{responsive:true}});
    """

divs = []
scripts = []

for idx,(fig,name) in enumerate([
    (fig1, "fig1_severity"),
    (fig2, "fig2_disease_counts"),
    (fig3, "fig3_heatmap_disease_symptom"),
    (figA, "figA_scatter_reg"),
    (figB, "figB_hist_box"),
    (figC, "figC_top_symptoms"),
    (figD, "figD_cooccurrence"),
    (figH, "figH_top_precautions"),
    (figI, "figI_treemap_families"),
    (figJ, "figJ_stack_families"),
    (figK, "figK_heatmap_families"),
]):
    d, s = to_div(fig, name)
    divs.append(d)
    scripts.append(s)

report_html = f"""<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8"/>
  <title>Rapport — Analyse maladies & symptômes</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 24px; }}
    h1, h2 {{ margin: 0 0 12px; }}
    .section {{ margin-bottom: 40px; }}
    .card {{ border: 1px solid #eee; border-radius: 12px; padding: 16px; box-shadow: 0 2px 6px rgba(0,0,0,0.04); }}
  </style>
</head>
<body>
  <h1>Rapport complet — Maladies, symptômes, sévérité & précautions</h1>
  <p class="muted">Fichier auto‑généré (Plotly). Chaque figure est interactive.</p>
  <div class="section card"><h2>Distribution par sévérité</h2>{divs[0]}</div>
  <div class="section card"><h2>Top maladies par nombre de symptômes</h2>{divs[1]}</div>
  <div class="section card"><h2>Carte thermique Disease × Symptom</h2>{divs[2]}</div>
  <div class="section card"><h2>Corrélation & régression</h2>{divs[3]}</div>
  <div class="section card"><h2>Distribution de la sévérité moyenne</h2>{divs[4]}</div>
  <div class="section card"><h2>Top symptômes (couverture maladies)</h2>{divs[5]}</div>
  <div class="section card"><h2>Co-occurrence des symptômes</h2>{divs[6]}</div>
  <div class="section card"><h2>Top précautions</h2>{divs[7]}</div>
  <div class="section card"><h2>Treemap par familles de symptômes</h2>{divs[8]}</div>
  <div class="section card"><h2>Répartition par familles (Top 15 maladies)</h2>{divs[9]}</div>
  <div class="section card"><h2>Sévérité moyenne par famille (Heatmap)</h2>{divs[10]}</div>
  <script>
    {"".join(scripts)}
  </script>
</body>
</html>
"""
report_path = Path("/mnt/data/report_all_figures.html")
report_path.write_text(report_html, encoding="utf-8")

# Standalone family analysis page (optional)
family_page = f"""<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8"/>
  <title>Analyse par familles de symptômes</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
  <h1>Analyse par familles de symptômes</h1>
  <div id="treemap" style="height:520px;"></div>
  <div id="stack" style="height:520px;"></div>
  <div id="heat" style="height:520px;"></div>
  <script>
    Plotly.newPlot("treemap", {figI.to_json()}, {{responsive:true}});
    Plotly.newPlot("stack", {figJ.to_json()}, {{responsive:true}});
    Plotly.newPlot("heat", {figK.to_json()}, {{responsive:true}});
  </script>
</body>
</html>
"""
family_path = Path("/mnt/data/symptom_family_analysis.html")
family_path.write_text(family_page, encoding="utf-8")

print({
    "disease_profile_selector": str(profile_html_path),
    "report_all_figures": str(report_path),
    "symptom_taxonomy_csv": str(taxo_path),
    "symptom_family_analysis": str(family_path)
})
