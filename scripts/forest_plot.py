import numpy as np
import matplotlib.pyplot as plt

# ---------- LMM RESULTS ----------
# Order: by significance of language gap (matching your bubble plot order)
models = [
    "Llama-3.1-405B-Instruct",
    "GPT-4-Turbo",
    "BioMistral-7B",
    "DeepSeek-R1",
    "o3"
]

# Mean difference (EN - FR) from LMM
estimates = [0.91, 0.49, 0.78, 0.37, 0.08]

# 95% CI [lower, upper] from LMM
ci_lower = [0.66, 0.25, 0.28, 0.10, -0.12]
ci_upper = [1.17, 0.73, 1.28, 0.64, 0.27]

# Adjusted p-values
p_adj = [
    r"$< 0.001$",
    r"$< 0.001$",
    r"0.006",
    r"0.0207",
    r"1.000"
]

# ---------- FOREST PLOT ----------
fig, ax = plt.subplots(figsize=(8, 3.5))
fig.subplots_adjust(right=0.50)

y_pos = np.arange(len(models))[::-1]  # top to bottom

# Plot CIs as horizontal lines
for i in range(len(models)):
    ax.plot([ci_lower[i], ci_upper[i]], [y_pos[i], y_pos[i]],
            color='#1f77b4', linewidth=2, solid_capstyle='round')
    # CI caps
    cap_h = 0.15
    ax.plot([ci_lower[i], ci_lower[i]],
            [y_pos[i] - cap_h, y_pos[i] + cap_h],
            color='#1f77b4', linewidth=1.5)
    ax.plot([ci_upper[i], ci_upper[i]],
            [y_pos[i] - cap_h, y_pos[i] + cap_h],
            color='#1f77b4', linewidth=1.5)

# Plot point estimates as diamonds
ax.scatter(estimates, y_pos, s=100, c='#1f77b4', zorder=5,
           marker='D', edgecolors='white', linewidth=0.5)

# Null effect line
ax.axvline(x=0, color='grey', linestyle='--', linewidth=1, zorder=0)

# Model labels on left
ax.set_yticks(y_pos)
ax.set_yticklabels(models)

# Annotate with estimate [95% CI] and p-value on the right
for i in range(len(models)):
    text = f"{estimates[i]:.2f} [{ci_lower[i]:.2f}, {ci_upper[i]:.2f}]"
    ax.annotate(text, xy=(1.08, y_pos[i]), xycoords=('axes fraction', 'data'),
                fontsize=8, va='center', ha='left', annotation_clip=False)
    ax.annotate(f"p = {p_adj[i]}", xy=(1.55, y_pos[i]),
                xycoords=('axes fraction', 'data'),
                fontsize=8, va='center', ha='left', annotation_clip=False)

# Column headers
header_y = max(y_pos) + 0.7
ax.annotate("Mean diff [95% CI]", xy=(1.08, header_y),
            xycoords=('axes fraction', 'data'),
            fontsize=8, fontweight='bold', va='center', ha='left',
            annotation_clip=False)
ax.annotate("Adj. p", xy=(1.55, header_y),
            xycoords=('axes fraction', 'data'),
            fontsize=8, fontweight='bold', va='center', ha='left',
            annotation_clip=False)

# Axis labels
ax.set_xlabel("Mean difference in score (EN − FR)", fontsize=10)
ax.set_xlim(-0.5, 1.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("../figures/Figure2_forest.png", dpi=300, bbox_inches="tight")
plt.close(fig)