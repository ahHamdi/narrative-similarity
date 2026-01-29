import pandas as pd
import re

MIN_TOKENS = 10

#The last sentence with at least MIN_TOKENS will be the query
def split_last_sentence(text):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if len(sentences) < 2:
        return None, None

    last = sentences[-1]
    if len(last.split()) < MIN_TOKENS:
        return None, None

    doc = " ".join(sentences[:-1])
    return last, doc

#Loading lang_en pairs 
df_fr = pd.read_csv("plots_fr_en.autotrain.csv").dropna(subset=["plots_en", "plots_fr_en"])
df_pt = pd.read_csv("plots_pt_en.autotrain.csv").dropna(subset=["plots_en", "plots_pt_en"])
df_es = pd.read_csv("plots_es_en.autotrain.csv").dropna(subset=["plots_en", "plots_es_en"])

common_plots = (
    set(df_fr["plots_en"])
    & set(df_pt["plots_en"])
    & set(df_es["plots_en"])
)

print(f"Common en plots for test : {len(common_plots)}")

# Build TEST IR
df_fr_c = df_fr[df_fr["plots_en"].isin(common_plots)].set_index("plots_en")
df_pt_c = df_pt[df_pt["plots_en"].isin(common_plots)].set_index("plots_en")
df_es_c = df_es[df_es["plots_en"].isin(common_plots)].set_index("plots_en")

test_rows = []

for plot_en in common_plots:
    query, doc_en = split_last_sentence(plot_en)
    if query is None:
        continue

    test_rows.append({
        "query": query,
        "doc_en": doc_en,
        "doc_fr": df_fr_c.loc[plot_en]["plots_fr_en"],
        "doc_pt": df_pt_c.loc[plot_en]["plots_pt_en"],
        "doc_es": df_es_c.loc[plot_en]["plots_es_en"],
    })

df_test = pd.DataFrame(test_rows)
df_test.to_csv("IR_testset.csv", index=False)

print(f"Generated test : {len(df_test)} queries")

# Build TRAIN IR and exclude common plots 
def build_train(df, col_lang):
    df_train = df[~df["plots_en"].isin(common_plots)].copy()
    df_train = df_train.rename(columns={col_lang: "plots_lang_en"})
    df_train = df_train[["plots_lang_en", "plots_en"]]
    return df_train.reset_index(drop=True)

train_fr = build_train(df_fr, "plots_fr_en")
train_pt = build_train(df_pt, "plots_pt_en")
train_es = build_train(df_es, "plots_es_en")

train_fr.to_csv("IR_trainset_fr_en.csv", index=False)
train_pt.to_csv("IR_trainset_pt_en.csv", index=False)
train_es.to_csv("IR_trainset_es_en.csv", index=False)

train_all = pd.concat([train_fr, train_pt, train_es], ignore_index=True)
train_all.to_csv("IR_trainset_all_langs_en.csv", index=False)

print("Generated TRAIN")
print(f"FR: {len(train_fr)} | PT: {len(train_pt)} | ES: {len(train_es)}")
print(f"ALL: {len(train_all)}")
