# Update the integrated report so Plot 1 and Plot 2 use multi-color scales (RdYlGn_r) like before.
import json
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

# Load
long_df = pd.read_csv("/mnt/data/analysis_ready_long.csv")

# Rebuild colored figures
# Plot 1: severity distribution (sorted by weight 1->5) + multi-color by weight
sev_counts = long_df.groupby("weight", dropna=False)["Symptom"].nunique().reset_index()
sev_counts = sev_counts.rename(columns={"Symptom": "Nombre_de_symptomes"}).dropna(subset=["weight"]).sort_values("weight")
fig1 = px.bar(
    sev_counts,
    x="weight",
    y="Nombre_de_symptomes",
    color="weight",
    color_continuous_scale="RdYlGn_r",
    title="Distribution des symptômes par sévérité — triée par poids (1→5)",
    labels={"weight": "Poids (sévérité)", "Nombre_de_symptomes": "Nombre de symptômes"}
)
fig1.update_layout(xaxis=dict(dtick=1))

# Plot 2: top diseases + multi-color by average severity
per_disease = (long_df.groupby("Disease")
               .agg(Nombre_de_symptomes=("Symptom","nunique"),
                    Severite_moyenne=("weight","mean"))
               .reset_index()
               .sort_values("Nombre_de_symptomes", ascending=False))
fig2 = px.bar(
    per_disease.head(25),
    x="Disease",
    y="Nombre_de_symptomes",
    color="Severite_moyenne",
    color_continuous_scale="RdYlGn_r",
    title="Top 25 maladies par nombre de symptômes (couleur = sévérité moyenne)",
    labels={"Disease":"Maladie","Nombre_de_symptomes":"Nombre de symptômes","Severite_moyenne":"Sévérité moyenne"}
)
fig2.update_layout(xaxis_tickangle=-45)

# Recreate the rest from the last integrated report build
# Heatmap disease x symptom
top_dis = per_disease.head(20)["Disease"].tolist()
common_symptoms = (long_df.groupby("Symptom")["Disease"].nunique()
                   .sort_values(ascending=False).head(30).index.tolist())
hm = (long_df[long_df["Disease"].isin(top_dis) & long_df["Symptom"].isin(common_symptoms)]
      .groupby(["Disease", "Symptom"])["weight"].max().reset_index())
heatmap_pivot = hm.pivot(index="Disease", columns="Symptom", values="weight").fillna(0)
fig3 = go.Figure(data=go.Heatmap(z=heatmap_pivot.values, x=list(heatmap_pivot.columns), y=list(heatmap_pivot.index),
                                 colorbar=dict(title="Poids")))
fig3.update_layout(title="Carte thermique (Disease × Symptom) — poids")

# Scatter + regression
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

# Histogram + box
figB = px.histogram(per_disease, x="Severite_moyenne", nbins=20, marginal="box",
                    title="Distribution de la sévérité moyenne (hist + box)")

# Top symptoms by diseases
symptom_breadth = (long_df.groupby("Symptom")["Disease"]
                   .nunique().reset_index()
                   .rename(columns={"Disease":"Nombre_de_maladies"})
                   .sort_values("Nombre_de_maladies", ascending=False))
figC = px.bar(symptom_breadth.head(30), x="Symptom", y="Nombre_de_maladies",
              title="Top symptômes par nb de maladies")
figC.update_layout(xaxis_tickangle=-45)

# Co-occurrence heatmap (top 25)
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

# Top precautions
prec_cols = [c for c in long_df.columns if str(c).lower().startswith("precaution")]
prec_series = pd.Series(dtype=str)
for c in prec_cols:
    prec_series = pd.concat([prec_series, long_df[c].dropna().astype(str)])
prec_counts = (prec_series.str.strip().str.lower().value_counts().reset_index())
prec_counts.columns = ["precaution", "count"]
figH = px.bar(prec_counts.head(20), x="precaution", y="count",
              title="Top 20 précautions recommandées")
figH.update_layout(xaxis_tickangle=-45)

# Families taxonomy (reuse quick heuristic)
def guess_family(sym: str) -> str:
    s = str(sym).lower()
    if any(k in s for k in ["skin", "rash", "itch", "ulcer", "blister"]): return "Dermatologique"
    if any(k in s for k in ["cough", "breath", "sneeze", "wheez", "phlegm", "throat", "chest_pain"]): return "Respiratoire"
    if any(k in s for k in ["nausea", "vomit", "diarr", "stomach", "abdominal", "appetite", "constipation"]): return "Gastro‑intestinal"
    if any(k in s for k in ["headache", "dizziness", "loss_of_balance", "tremor", "seizure"]): return "Neurologique"
    if any(k in s for k in ["fever", "fatigue", "weight_loss", "sweating", "malaise", "dehydration"]): return "Systémique"
    if any(k in s for k in ["joint", "muscle", "back_pain", "knee", "hip", "bone"]): return "Musculo‑squelettique"
    if any(k in s for k in ["urine", "urination", "dysuria"]): return "Urologique/Rénal"
    if any(k in s for k in ["eye", "vision", "conjunctivitis"]): return "Ophtalmologique"
    if any(k in s for k in ["ear", "nose", "sinus"]): return "ORL"
    return "Autre"

taxo_map = (long_df[["Symptom"]].dropna().drop_duplicates()
            .assign(Famille=lambda d: d["Symptom"].map(guess_family)))
long_df_tax = long_df.merge(taxo_map, on="Symptom", how="left")

figI = px.treemap(
    (long_df_tax.groupby(["Famille","Symptom"])["Disease"].nunique()
     .reset_index(name="Nombre_de_maladies"))
    .merge(long_df_tax.groupby(["Famille","Symptom"])["weight"].mean()
           .reset_index(name="Severite_moyenne"), on=["Famille","Symptom"], how="left"),
    path=["Famille","Symptom"],
    values="Nombre_de_maladies",
    color="Severite_moyenne",
    title="Treemap — Familles → Symptômes (taille = nb maladies, couleur = sévérité moy.)"
)

per_dis_fam = (long_df_tax.groupby(["Disease","Famille"])["Symptom"].nunique().reset_index(name="Nb_symptomes"))
top_dis15 = (long_df.groupby("Disease")["Symptom"].nunique().sort_values(ascending=False).head(15).index.tolist())
per_dis_fam_top = per_dis_fam[per_dis_fam["Disease"].isin(top_dis15)]
figJ = px.bar(per_dis_fam_top, x="Disease", y="Nb_symptomes", color="Famille",
              title="Top 15 maladies — répartition des symptômes par famille",
              barmode="relative")
figJ.update_layout(xaxis_tickangle=-45)

fam_heat = (long_df_tax.groupby(["Disease","Famille"])["weight"].mean().reset_index())
fam_pivot = fam_heat.pivot(index="Disease", columns="Famille", values="weight").fillna(0)
figK = go.Figure(data=go.Heatmap(z=fam_pivot.values, x=list(fam_pivot.columns),
                                 y=list(fam_pivot.index), colorbar=dict(title="Sévérité moyenne")))
figK.update_layout(title="Carte thermique — Sévérité moyenne par famille de symptômes")

# Rebuild the integrated HTML but only update plots (selector stays the same from previous file generation)
def to_div(fig, div_id):
    return f"<div id='{div_id}' style='height:520px;'></div>", f"Plotly.newPlot('{div_id}', {fig.to_json()}, {{responsive:true}});"

divs = []
scripts = []
for fig, name in [
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
]:
    d, s = to_div(fig, name)
    divs.append(d)
    scripts.append(s)

# Grab previous selector payload and build integrated page again (with selector up top)
# Reuse same selector rendering JS from the last build
selector_js = f"""
(function() {{
  const DATA = {json.dumps({k: v for k, v in disease_profile_payload(long_df).items()})};
  const diseases = Object.keys(DATA).sort();
  const select = document.getElementById('diseaseSelect');
  diseases.forEach(d => {{ const opt = document.createElement('option'); opt.value = d; opt.textContent = d; select.appendChild(opt); }});
  function sanitizeY(arr) {{ return arr.map(v => (Number.isFinite(v) ? v : null)); }}
  function render(disease) {{
    const d = DATA[disease]; if (!d) return;
    document.getElementById('desc').textContent = d.description || '(Pas de description)';
    const ul = document.getElementById('precautions'); ul.innerHTML = '';
    if (Array.isArray(d.precautions) && d.precautions.length) {{
      d.precautions.forEach(p => {{ const li = document.createElement('li'); li.textContent = p; ul.appendChild(li); }});
    }} else {{ const li = document.createElement('li'); li.textContent = '(Aucune précaution disponible)'; ul.appendChild(li); }}
    const x = (d.symptoms || []).map(s => s.name);
    const y = sanitizeY((d.symptoms || []).map(s => s.weight));
    const trace = {{ type: 'bar', x: x, y: y, hovertemplate: '%{{x}}: %{{y}}<extra></extra>' }};
    const layout = {{ margin: {{t: 10, r: 10, b: 120, l: 50}}, xaxis: {{tickangle: -45, title: 'Symptôme'}}, yaxis: {{title: 'Poids (sévérité)', rangemode: 'tozero'}} }};
    Plotly.newPlot('chart', [trace], layout, {{responsive: true}});
  }}
  select.addEventListener('change', (e) => render(e.target.value));
  if (diseases.length) {{ select.value = diseases[0]; render(diseases[0]); }}
}})();
"""

full_html = f"""<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8"/>
  <title>Rapport complet — Profil + Analyses (couleurs corrigées)</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 24px; }}
    .layout {{ display: grid; grid-template-columns: 320px 1fr; gap: 24px; }}
    .card {{ border: 1px solid #eee; border-radius: 12px; padding: 16px; box-shadow: 0 2px 6px rgba(0,0,0,0.04); }}
    h1, h2 {{ margin: 0 0 12px; }}
    .section {{ margin-bottom: 40px; }}
    select {{ width: 100%; padding: 8px; border-radius: 8px; border: 1px solid #ddd; }}
    ul {{ margin: 8px 0 0 18px; }}
    .muted {{ color: #666; }}
  </style>
</head>
<body>
  <h1>Profil interactif + Rapport complet</h1>
  <p class="muted">Sélecteur en tête, suivi de toutes les analyses (figures interactives). Couleurs mises à jour.</p>

  <!-- Selector -->
  <div class="section card">
    <h2>Profil interactif d'une maladie</h2>
    <div class="layout">
      <div>
        <select id="diseaseSelect"></select>
        <div style="margin-top:16px">
          <h3>Description</h3>
          <div id="desc" class="muted">(sélectionne une maladie)</div>
          <h3 style="margin-top:16px">Précautions</h3>
          <ul id="precautions"></ul>
        </div>
      </div>
      <div>
        <h3>Symptômes triés par sévérité</h3>
        <div id="chart" style="height:520px;"></div>
      </div>
    </div>
  </div>

  <!-- Plots -->
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
    {selector_js}
    {"".join(scripts)}
  </script>
</body>
</html>
"""

out_path = Path("/mnt/data/report_full_with_selector.html")
out_path.write_text(full_html, encoding="utf-8")

str(out_path)
