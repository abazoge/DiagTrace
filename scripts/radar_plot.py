import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

from utils import get_data, llm_types, couleurs

# ---- Radar factory (Matplotlib docs) ----
def radar_factory(num_vars, frame='circle'):
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):
        def transform_path_non_affine(self, path):
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):
        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)
            return lines

        def _close_line(self, line):
            x, y = line.get_data()
            if len(x) == 0:
                return
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars, radius=.5, edgecolor="k")
            raise ValueError(f"Unknown frame: {frame}")

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                spine = Spine(axes=self, spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5) + self.transAxes)
                return {'polar': spine}
            raise ValueError(f"Unknown frame: {frame}")

    register_projection(RadarAxes)
    return theta

# ---- Data ----
labels = [
    "Final diagnosis", "Internal\nvalidity", "External\nvalidity",
    "Differential diagnoses", "Logical\nstructure", "Expression"
]

# Define consistent FR/EN colors
COLOR_FR = '#2C5F8A'   # dark blue for French
COLOR_EN = '#8FBCDB'   # light blue for English

# Load all models
models = {
    'o3_FR': get_data('o3_FR'),
    'o3_EN': get_data('o3_EN'),
    'DeepSeek_FR': get_data('DeepSeek_FR'),
    'DeepSeek_EN': get_data('DeepSeek_EN'),
    'GPT_FR': get_data('GPT_FR'),
    'GPT_EN': get_data('GPT_EN'),
    'Llama_FR': get_data('Llama_FR'),
    'Llama_EN': get_data('Llama_EN'),
    'Biomistral_FR': get_data('Biomistral_FR'),
    'Biomistral_EN': get_data('Biomistral_EN'),
}

# Normalize all scores to 0-3 scale
# D: 0-3 (no normalization), VI: 0-5, VE: 0-3, H: 0-1, L: 0-4, E: 0-2
# Alternative (0-1 scale)
def normalize(data):
    return [
        np.mean(np.array(data.D)) / 3,                # D: 0-3 -> 0-1
        np.mean(np.array(data.VI)) / 5,                # VI: 0-5 -> 0-1
        np.mean(np.array(data.VE)) / 3,                # VE: 0-3 -> 0-1
        np.mean(np.array(data.H)),                     # H: 0-1 -> already 0-1
        np.mean(np.array(data.L)) / 4,                 # L: 0-4 -> 0-1
        np.mean(np.array(data.E)) / 2,                 # E: 0-2 -> 0-1
    ]

scores = {k: normalize(v) for k, v in models.items()}

# ---- Panel definitions (3 top, 2 bottom) ----
panels = [
    # Top row
    ("o3",           [("French", scores['o3_FR'],        None),
                      ("English", scores['o3_EN'],       None)]),
    ("DeepSeek-R1",  [("French", scores['DeepSeek_FR'],  None),
                      ("English", scores['DeepSeek_EN'], None)]),
    ("GPT-4-Turbo",        [("French", scores['GPT_FR'],       None),
                      ("English", scores['GPT_EN'],      None)]),
    # Bottom row
    ("Llama-3.1-405B-Instruct",   [("French", scores['Llama_FR'],     None),
                      ("English", scores['Llama_EN'],    None)]),
    ("BioMistral-7B",   [("French", scores['Biomistral_FR'], None),
                      ("English", scores['Biomistral_EN'], None)]),
]

# ---- Plot: manual positioning for equal size + centered bottom row ----
theta = radar_factory(len(labels), frame='polygon')

fig = plt.figure(figsize=(20, 14))

# Each panel size
scale = 0.70
w, h = 0.28 * scale, 0.40 * scale

# Top row: 3 panels evenly spaced
top_y = 0.52
top_xs = [0.04, 0.36, 0.68]

# Bottom row: 2 panels centered
bot_y = 0.15
bot_xs = [0.20, 0.52]

ax_positions = []
for x in top_xs:
    ax_positions.append(fig.add_axes([x, top_y, w, h], projection='radar'))
for x in bot_xs:
    ax_positions.append(fig.add_axes([x, bot_y, w, h], projection='radar'))

all_axes = ax_positions

# Style all axes
for ax in all_axes:
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1"], fontsize=8)
    ax.set_rlabel_position(180)
    ax.grid(color="#999999", alpha=0.5, linewidth=0.8)
    ax.spines["polar"].set_color("#888888")
    ax.spines["polar"].set_linewidth(1.0)
    ax.set_varlabels(labels)
    ax.tick_params(axis='x', pad=20, labelsize=11)

    # Adjust individual label positions
    for label, angle in zip(ax.get_xticklabels(), np.degrees(theta)):
        # "Final diagnosis" is at 0° (top), "Differential diagnoses" is at 180° (bottom)
        if label.get_text() == "Final diagnosis" or label.get_text() == "Differential diagnoses":
            label.set_y(label.get_position()[1] + 0.10)  # pull closer

# Plot data
for ax, (title, series) in zip(all_axes, panels):
    for i, (name, values, _) in enumerate(series):
        color = COLOR_FR if i == 0 else COLOR_EN
        ax.plot(theta, values, color=color, linewidth=2.5)
    ax.set_title(title, fontsize=16, pad=20, fontweight='bold')

# Panel letters
panel_letters = ['(a)', '(b)', '(c)', '(d)', '(e)']
for ax, letter in zip(all_axes, panel_letters):
    ax.text(-0.05, 1.08, letter, transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='bottom')

# Legend
legend_handles = [
    Line2D([0], [0], color=COLOR_FR, lw=3, label='French'),
    Line2D([0], [0], color=COLOR_EN, lw=3, label='English'),
]
fig.legend(handles=legend_handles, loc="lower center",
           bbox_to_anchor=(0.46, 0.08), ncol=2, frameon=False, fontsize=14)

plt.savefig("../figures/Figure3_radar.png", dpi=300, bbox_inches="tight")
plt.close(fig)