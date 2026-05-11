import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from utils import get_data, couleurs

dict_name_legend = {
    'o3_FR': "o3 - French",
    'o3_EN': "o3 - English",
    'DeepSeek_FR': "DeepSeek-R1 - French",
    'DeepSeek_EN': "DeepSeek-R1 - English",
    'GPT_FR': "GPT-4-Turbo - French",
    'GPT_EN': "GPT-4-Turbo - English",
    'Llama_FR': "Llama-3.1-405B-Instruct - French",
    'Llama_EN': "Llama-3.1-405B-Instruct - English",
    'Biomistral_FR': "BioMistral-7B - French",
    'Biomistral_EN': "BioMistral-7B - English",
}

# Each row is (EN_key, FR_key)
model_pairs = [
    ('o3_EN', 'o3_FR'),
    ('DeepSeek_EN', 'DeepSeek_FR'),
    ('GPT_EN', 'GPT_FR'),
    ('Llama_EN', 'Llama_FR'),
    ('Biomistral_EN', 'Biomistral_FR'),
]

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12

fig, axs = plt.subplots(5, 2, figsize=(12, 12))
bins = [x - 0.5 for x in range(0, 20)]

for row, (model_en, model_fr) in enumerate(model_pairs):
    for col, model in enumerate((model_en, model_fr)):
        ax = axs[row, col]
        notes = np.array(get_data(model).Note)

        sns.histplot(notes, bins=bins, ax=ax, color=couleurs[model],
                     shrink=1, stat='percent')
        ax.axvline(x=np.mean(notes), color='red', linestyle='dotted', alpha=0.5)
        ax.set_title(dict_name_legend[model])
        ax.set_xlim(-0.5, 18.5)
        ax.set_ylim(0, 60)
        ax.set_xlabel('Score')
        ax.set_xticks(range(0, 19))
        ax.set_ylabel('Frequency (%)' if col == 0 else '')

plt.tight_layout()
plt.savefig("../figures/Figure4_hist.png", dpi=300, bbox_inches="tight")