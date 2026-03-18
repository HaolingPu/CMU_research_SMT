import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'

# Data: (StreamLAAL ms, BLEU)
data = {
    "EAST": [
        (1264, 39.74),
        (1978, 45.74),
        (2767, 47.52),
        (3599, 49.94),
    ],
    "Refined-EAST": [
        (1247, 40.20),
        (1962, 45.31),
        (2864, 48.17),
        (3541, 49.59),
    ],
    "Simul-MuST-C": [
        (1125, 38.32),
        (2006, 46.34),
        (2607, 48.62),
        (3057, 48.26),
    ],
    "Word-Alignment": [
        (1180, 42.33),
        (1808, 45.60),
        (2251, 48.19),
        (2615, 48.48),
    ],
}

markers = ['o', 's', '^', 'D']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

fig, ax = plt.subplots(figsize=(7, 5))

for (name, points), marker, color in zip(data.items(), markers, colors):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ax.plot(xs, ys, marker=marker, color=color, label=name, linewidth=2, markersize=7)

ax.set_xlabel("Latency — StreamLAAL (ms)", fontsize=13)
ax.set_ylabel("Quality — BLEU", fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_title("Latency–Quality Trade-off (ACL 6060 dev En→Zh)", fontsize=14)

plt.tight_layout()
plt.savefig("/home/siqiouya/code/CMU_research_SMT/scripts/infer/latency_quality_tradeoff.png", dpi=150)
plt.savefig("/home/siqiouya/code/CMU_research_SMT/scripts/infer/latency_quality_tradeoff.pdf")
print("Saved plot.")
