import matplotlib.pyplot as plt
import numpy as np

models = ["MF", "LightGCN", "HeteroGNN"]


recall = [0.0373, 0.0397, 0.0171]
ndcg   = [0.0191, 0.0203, 0.0081]

x = np.arange(len(models))
width = 0.35 

fig, ax = plt.subplots()

ax.bar(x - width / 2, recall, width, label="Recall@10")
ax.bar(x + width / 2, ndcg,   width, label="NDCG@10")

ax.set_ylabel("Score")
ax.set_title("Model Performance on Yelp-Philadelphia (Validation)")
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(0, 0.045) 
ax.legend()


plt.show()
