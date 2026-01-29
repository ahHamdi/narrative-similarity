import pandas as pd

# Charger les fichiers
df_fr = pd.read_csv("plots_fr_en.autotrain_NE_masked.csv")
df_pt = pd.read_csv("plots_pt_en.autotrain_NE_masked.csv")
df_es = pd.read_csv("plots_es_en.autotrain_NE_masked.csv")
print(len(df_fr))
print(len(df_pt))
print(len(df_es))

# Renommer les colonnes pour les harmoniser
df_fr = df_fr.rename(columns={"plots_fr_en": "plots_lang_en"})
df_pt = df_pt.rename(columns={"plots_pt_en": "plots_lang_en"})
df_es = df_es.rename(columns={"plots_es_en": "plots_lang_en"})

# Garder uniquement les colonnes utiles (sécurité)
df_fr = df_fr[["plots_lang_en", "plots_en"]]
df_pt = df_pt[["plots_lang_en", "plots_en"]]
df_es = df_es[["plots_lang_en", "plots_en"]]

# Concaténer
df_all = pd.concat([df_fr, df_pt, df_es], ignore_index=True)
print(len(df_all))

# (Optionnel mais recommandé) supprimer lignes vides
df_all = df_all.dropna(subset=["plots_lang_en", "plots_en"])

# Sauvegarder
df_all.to_csv("plots_all_langs_en.autotrain_NE_masked.csv", index=False)
