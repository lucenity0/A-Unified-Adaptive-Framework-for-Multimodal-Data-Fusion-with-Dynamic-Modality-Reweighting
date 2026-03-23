import pandas as pd

s = pd.read_csv("../results/test_predictions_static.csv")
d = pd.read_csv("../results/test_predictions_dynamic.csv")

m = s.merge(d, on="id", suffixes=("_static", "_dynamic"))
m["label_changed"] = m["predicted_label_static"] != m["predicted_label_dynamic"]
m["prob_delta"] = m["prob_hateful_dynamic"] - m["prob_hateful_static"]
m["conf_delta"] = m["confidence_dynamic"] - m["confidence_static"]

print(f"Total matched rows: {len(m)}")
print(f"Label disagreements: {m['label_changed'].sum()} ({100*m['label_changed'].mean():.2f}%)")
print(f"Mean prob delta (dyn-static): {m['prob_delta'].mean():.4f}")
print(f"Mean conf delta (dyn-static): {m['conf_delta'].mean():.4f}")

cols = [
    "id", "text_static", "predicted_label_static", "predicted_label_dynamic",
    "prob_hateful_static", "prob_hateful_dynamic", "prob_delta",
    "dominant_modality_static", "dominant_modality_dynamic",
    "alpha_mean_static", "alpha_mean_dynamic"
]
top = m.reindex(columns=cols).iloc[m["prob_delta"].abs().sort_values(ascending=False).index[:10]]
print("\nTop 10 biggest probability shifts:")
print(top.to_string(index=False))

m.to_csv("../results/static_dynamic_comparison.csv", index=False)
print("\nSaved: ../results/static_dynamic_comparison.csv")
