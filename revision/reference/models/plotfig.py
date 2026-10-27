
# import matplotlib.pyplot as plt
# from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter

# # -----------------------------
# # Data
# # -----------------------------
# ratios = [5, 10, 15, 20, 25]

# oa = {
#     "HC":  [98.63, 99.10, 99.28, 99.69, 99.53],
#     "HH":  [98.63, 98.70, 98.00, 99.18, 99.22],
#     "PI":  [98.84, 98.51, 99.29, 99.34, 99.35],
#     "TH":  [99.31, 99.47, 99.74, 99.38, 99.63],
#     "QUH": [98.37, 98.98, 99.20, 99.34, 99.59],
# }

# train_time = {
#     "HC":  [1613, 3515, 5271, 6892, 8013],
#     "HH":  [2407, 5469, 7838, 10593, 12154],
#     "PI":  [7391, 16135, 23627, 29925, 45229],
#     "TH":  [6400, 8002, 10888, 14073, 22187],
#     "QUH": [11386, 13691, 18734, 25701, 56625],
# }

# # -----------------------------
# # Colorblind-safe Okabe-Ito palette
# # -----------------------------
# colors = {
#     "HC":  "#0072B2",
#     "HH":  "#D55E00",
#     "PI":  "#009E73",
#     "TH":  "#CC79A7",
#     "QUH": "#E69F00",
# }

# markers = {
#     "HC":  "o",
#     "HH":  "s",
#     "PI":  "^",
#     "TH":  "D",
#     "QUH": "v",
# }

# linestyles = {
#     "HC":  "-",
#     "HH":  "--",
#     "PI":  "-.",
#     "TH":  ":",
#     "QUH": "-",
# }

# # -----------------------------
# # IEEE-style font and export settings
# # -----------------------------
# plt.rcParams.update({
#     "font.family": "serif",
#     "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
#     "mathtext.fontset": "stix",
#     "font.size": 8,
#     "axes.labelsize": 8,
#     "axes.titlesize": 8,
#     "xtick.labelsize": 7,
#     "ytick.labelsize": 7,
#     "legend.fontsize": 7,
#     "axes.linewidth": 0.7,
#     "pdf.fonttype": 42,
#     "ps.fonttype": 42,
# })

# fig, axes = plt.subplots(
#     1, 2,
#     figsize=(7.16, 2.75),
#     constrained_layout=False
# )

# # -----------------------------
# # Panel (a): OA
# # -----------------------------
# ax = axes[0]

# for name in oa:
#     ax.plot(
#         ratios,
#         oa[name],
#         label=name,
#         color=colors[name],
#         marker=markers[name],
#         linestyle=linestyles[name],
#         linewidth=1.35,
#         markersize=4.2,
#         markerfacecolor="white",
#         markeredgewidth=1.0,
#     )

# ax.set_xlabel("Training Ratio (%)")
# ax.set_ylabel("OA (%)")
# ax.set_xticks(ratios)
# ax.set_ylim(98.0, 100.0)
# ax.set_yticks([98.0, 98.5, 99.0, 99.5, 100.0])

# ax.grid(True, which="major", linestyle="-", linewidth=0.35, alpha=0.35)
# ax.minorticks_on()
# ax.grid(True, which="minor", linestyle=":", linewidth=0.25, alpha=0.18)

# ax.spines["top"].set_visible(False)
# ax.spines["right"].set_visible(False)
# # ax.text(
# #     0.02, 0.96, "(a)",
# #     transform=ax.transAxes,
# #     fontsize=8,
# #     fontweight="bold",
# #     va="top",
# #     ha="left"
# # )

# # -----------------------------
# # Panel (b): Training time
# # -----------------------------
# ax = axes[1]

# for name in train_time:
#     ax.plot(
#         ratios,
#         train_time[name],
#         label=name,
#         color=colors[name],
#         marker=markers[name],
#         linestyle=linestyles[name],
#         linewidth=1.35,
#         markersize=4.2,
#         markerfacecolor="white",
#         markeredgewidth=1.0,
#     )

# ax.set_xlabel("Training Ratio (%)")
# ax.set_ylabel("Training Time (s)")
# ax.set_xticks(ratios)

# # Log scale, but with normal readable labels
# ax.set_yscale("log")
# ax.set_ylim(1200, 70000)

# major_ticks = [2000, 5000, 10000, 20000, 50000]
# ax.yaxis.set_major_locator(FixedLocator(major_ticks))
# ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))

# # Minor ticks/grid but no minor labels
# ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=[2, 3, 4, 5, 6, 7, 8, 9]))
# ax.yaxis.set_minor_formatter(NullFormatter())

# # Extra protection: hide offset/scientific notation text
# ax.yaxis.get_offset_text().set_visible(False)

# ax.grid(True, which="major", linestyle="-", linewidth=0.35, alpha=0.35)
# ax.grid(True, which="minor", linestyle=":", linewidth=0.25, alpha=0.14)

# ax.spines["top"].set_visible(False)
# ax.spines["right"].set_visible(False)
# # ax.text(
# #     0.02, 0.96, "(b)",
# #     transform=ax.transAxes,
# #     fontsize=8,
# #     fontweight="bold",
# #     va="top",
# #     ha="left"
# # )

# # -----------------------------
# # Shared legend
# # -----------------------------
# handles, labels = axes[0].get_legend_handles_labels()

# fig.legend(
#     handles,
#     labels,
#     loc="upper center",
#     ncol=5,
#     frameon=False,
#     bbox_to_anchor=(0.5, 0.92),
#     columnspacing=1.4,
#     handlelength=2.0,
#     handletextpad=0.4,
#     borderaxespad=0.0,
# )

# # -----------------------------
# # Layout
# # -----------------------------
# plt.subplots_adjust(
#     left=0.075,
#     right=0.985,
#     bottom=0.20,
#     top=0.78,
#     wspace=0.30,
# )

# # -----------------------------
# # Save
# # -----------------------------
# plt.savefig("fig_diff_ratio.pdf", bbox_inches="tight", pad_inches=0.02)
# plt.savefig("fig_diff_ratio.png", dpi=600, bbox_inches="tight", pad_inches=0.02)

# # plt.show()



# Different attention mechanisms comparison figure

# import matplotlib.pyplot as plt
# from matplotlib.ticker import FuncFormatter

# # -----------------------------
# # Data
# # -----------------------------
# x = [1, 2, 3, 4, 5]
# xlabels = ["NA", "MHSA", "CA", "DMHSA", "FSSA"]

# oa = {
#     "HC":  [77.53, 95.25, 90.00, 92.35, 98.63],
#     "HH":  [87.92, 96.39, 95.06, 94.15, 98.63],
#     "PI":  [93.47, 97.65, 95.58, 97.62, 98.84],
#     "TH":  [94.23, 97.70, 97.21, 97.62, 99.31],
#     "QUH": [91.46, 95.86, 94.78, 96.53, 98.37],
# }

# kappa = {
#     "HC":  [73.43, 94.44, 88.28, 91.06, 98.40],
#     "HH":  [84.71, 95.43, 93.74, 92.61, 98.27],
#     "PI":  [90.25, 96.50, 93.40, 96.46, 98.27],
#     "TH":  [93.43, 97.38, 96.83, 97.30, 99.21],
#     "QUH": [88.67, 94.51, 93.08, 95.41, 97.85],
# }

# aa = {
#     "HC":  [57.99, 91.74, 79.99, 87.00, 97.43],
#     "HH":  [73.59, 84.11, 93.36, 87.29, 95.50],
#     "PI":  [82.57, 93.92, 87.87, 93.85, 96.93],
#     "TH":  [89.36, 93.67, 93.86, 96.15, 98.50],
#     "QUH": [83.99, 91.15, 89.30, 95.04, 96.93],
# }

# # -----------------------------
# # Style
# # -----------------------------
# colors = {
#     "HC":  "#0072B2",
#     "HH":  "#D55E00",
#     "PI":  "#009E73",
#     "TH":  "#CC79A7",
#     "QUH": "#E69F00",
# }

# linestyles = {
#     "HC":  "-",
#     "HH":  "--",
#     "PI":  "-.",
#     "TH":  ":",
#     "QUH": (0, (5, 2, 1, 2)),
# }

# markers = {
#     "HC":  "o",
#     "HH":  "s",
#     "PI":  "^",
#     "TH":  "X",
#     "QUH": "P",
# }

# plt.rcParams.update({
#     "font.family": "serif",
#     "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
#     "mathtext.fontset": "stix",
#     "font.size": 8,
#     "axes.labelsize": 8,
#     "axes.titlesize": 8,
#     "xtick.labelsize": 7,
#     "ytick.labelsize": 7,
#     "legend.fontsize": 7,
#     "axes.linewidth": 0.7,
#     "pdf.fonttype": 42,
#     "ps.fonttype": 42,
# })

# # -----------------------------
# # Figure
# # -----------------------------
# fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.1), constrained_layout=False)

# panels = [
#     ("OA (%)", oa),
#     (r"Kappa ($\kappa$)", kappa),
#     ("AA (%)", aa),
# ]

# for ax, (ylabel, data) in zip(axes, panels):
#     for name in data:
#         ax.plot(
#             x,
#             data[name],
#             label=name,
#             color=colors[name],
#             marker=markers[name],
#             linestyle=linestyles[name],
#             linewidth=1.35,
#             markersize=4.2,
#             markerfacecolor="white",
#             markeredgewidth=1.0,
#         )

#     ax.set_xticks(x)
#     ax.set_xticklabels(xlabels)
#     ax.set_ylabel(ylabel)

#     ax.grid(True, which="major", linestyle="-", linewidth=0.35, alpha=0.35)
#     ax.minorticks_on()
#     ax.grid(True, which="minor", linestyle=":", linewidth=0.25, alpha=0.15)

#     ax.spines["top"].set_visible(False)
#     ax.spines["right"].set_visible(False)

#     ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f}"))

# # -----------------------------
# # Y-axis limits
# # -----------------------------
# axes[0].set_ylim(75, 100)
# axes[1].set_ylim(70, 100)
# axes[2].set_ylim(55, 100)

# # -----------------------------
# # Legend inside the first plot
# # -----------------------------
# axes[0].legend(
#     loc="lower right",
#     ncol=1,
#     frameon=True,
#     fancybox=True,
#     framealpha=0.85,
#     edgecolor="0.75",
#     fontsize=7,
#     handlelength=1.7,
#     handletextpad=0.4,
#     borderpad=0.3,
# )

# # -----------------------------
# # Layout
# # -----------------------------
# plt.subplots_adjust(
#     left=0.055,
#     right=0.995,
#     bottom=0.20,
#     top=0.94,
#     wspace=0.28,
# )

# # -----------------------------
# # Save
# # -----------------------------
# plt.savefig("fig_diff_attn_combine.pdf", bbox_inches="tight", pad_inches=0.02)
# plt.savefig("fig_diff_attn_combine.png", dpi=600, bbox_inches="tight", pad_inches=0.02)

# plt.show()



# import matplotlib.pyplot as plt
# from matplotlib.ticker import FuncFormatter

# # -----------------------------
# # Data
# # -----------------------------
# x = [1, 2, 3, 4, 5]
# xlabels = ["NA", "MHSA", "CA", "DMHSA", "FSSA"]

# oa = {
#     "HC":  [77.53, 95.25, 90.00, 92.35, 98.63],
#     "HH":  [87.92, 96.39, 95.06, 94.15, 98.63],
#     "PI":  [93.47, 97.65, 95.58, 97.62, 98.84],
#     "TH":  [94.23, 97.70, 97.21, 97.62, 99.31],
#     "QUH": [91.46, 95.86, 94.78, 96.53, 98.37],
# }
# colors = {
#     "HC":  "#0072B2",
#     "HH":  "#D55E00",
#     "PI":  "#009E73",
#     "TH":  "#CC79A7",
#     "QUH": "#E69F00",
# }

# linestyles = {
#     "HC":  "-",
#     "HH":  "--",
#     "PI":  "-.",
#     "TH":  ":",
#     "QUH": (0, (5, 2, 1, 2)),
# }

# markers = {
#     "HC":  "o",
#     "HH":  "s",
#     "PI":  "^",
#     "TH":  "X",
#     "QUH": "P",
# }

# # -----------------------------
# # IEEE-style colors
# # -----------------------------
# # colors = {
# #     "HC":  "#0072B2",
# #     "HH":  "#D55E00",
# #     "PI":  "#009E73",
# #     "TH":  "#CC79A7",
# #     "QUH": "#E69F00",
# # }

# # colors = {
# #     "HC":  "#1f77b4",  # blue
# #     "HH":  "#ff7f0e",  # orange
# #     "PI":  "#2ca02c",  # green
# #     "TH":  "#d62728",  # red
# #     "QUH": "#9467bd",  # purple
# # }


# # markers = {
# #     "HC":  "o",
# #     "HH":  "s",
# #     "PI":  "^",
# #     "TH":  "D",
# #     "QUH": "v",
# # }

# # linestyles = {
# #     "HC":  "-",
# #     "HH":  "--",
# #     "PI":  "-.",
# #     "TH":  ":",
# #     "QUH": "-",
# # }
# # --------------------------
# # IEEE font settings
# # -----------------------------
# plt.rcParams.update({
#     "font.family": "serif",
#     "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
#     "mathtext.fontset": "stix",
#     "font.size": 8,
#     "axes.labelsize": 8,
#     "axes.titlesize": 8,
#     "xtick.labelsize": 7,
#     "ytick.labelsize": 7,
#     "legend.fontsize": 7,
#     "axes.linewidth": 0.7,
#     "pdf.fonttype": 42,
#     "ps.fonttype": 42,
# })

# # -----------------------------
# # OA-only figure
# # -----------------------------
# plt.figure(figsize=(3.55, 2.65))  # good for one-column IEEE figure

# for name, values in oa.items():
#     plt.plot(
#         x,
#         values,
#         label=name,
#         color=colors[name],
#         marker=markers[name],
#         linestyle=linestyles[name],
#         linewidth=1.35,
#         markersize=4.2,
#         markerfacecolor="white",
#         markeredgewidth=1.0,
#     )

# plt.xlabel("Attention Mechanism")
# plt.ylabel("OA (%)")
# plt.xticks(x, xlabels)
# plt.ylim(75, 100)
# plt.yticks([75, 80, 85, 90, 95, 100])
# plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f}"))

# plt.grid(True, which="major", linestyle="-", linewidth=0.35, alpha=0.35)
# plt.minorticks_on()
# plt.grid(True, which="minor", linestyle=":", linewidth=0.25, alpha=0.15)

# ax = plt.gca()
# ax.spines["top"].set_visible(False)
# ax.spines["right"].set_visible(False)

# plt.legend(
#     ncol=3,
#     frameon=False,
#     loc="lower right",
#     fontsize=7,
#     columnspacing=0.8,
#     handlelength=1.8,
#     handletextpad=0.4,
# )

# plt.tight_layout()

# plt.savefig("fig_diff_attn_oa.pdf", bbox_inches="tight", pad_inches=0.02)
# plt.savefig("fig_diff_attn_oa.png", dpi=600, bbox_inches="tight", pad_inches=0.02)

# # plt.show()


# import numpy as np
# import matplotlib.pyplot as plt

# # -----------------------------
# # Data
# # -----------------------------
# heads = [2, 4, 6, 8]

# data = {
#     "HC":  [95.944, 96.845, 95.557, 98.639],
#     "HH":  [96.065, 98.235, 97.011, 98.634],
#     "PI":  [97.556, 98.837, 97.920, 98.871],
#     "TH":  [98.429, 99.442, 98.565, 99.311],
#     "QUH": [97.119, 98.298, 97.915, 98.379],
# }

# # -----------------------------
# # Cleaner publication colors
# # -----------------------------
# colors = {
#       "HC":  "#0072B2",
#     "HH":  "#D55E00",
#     "PI":  "#009E73",
#     "TH":  "#CC79A7",
#     "QUH": "#E69F00",
#     # "HC":  "#4C78A8",  # muted blue
#     # "HH":  "#F58518",  # muted orange
#     # "PI":  "#54A24B",  # muted green
#     # "TH":  "#B279A2",  # muted purple
#     # "QUH": "#ECA82C",  # golden yellow
# }

# # -----------------------------
# # IEEE-style font settings
# # -----------------------------
# plt.rcParams.update({
#     "font.family": "serif",
#     "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
#     "mathtext.fontset": "stix",
#     "font.size": 8,
#     "axes.labelsize": 8,
#     "axes.titlesize": 8,
#     "xtick.labelsize": 7,
#     "ytick.labelsize": 7,
#     "legend.fontsize": 7,
#     "axes.linewidth": 0.7,
#     "pdf.fonttype": 42,
#     "ps.fonttype": 42,
# })

# # -----------------------------
# # Plot
# # -----------------------------
# fig, ax = plt.subplots(figsize=(4.8, 2.9))

# x = np.arange(len(heads))
# dataset_names = list(data.keys())

# bar_width = 0.145
# group_width = bar_width * len(dataset_names)

# offsets = np.linspace(
#     -group_width / 2 + bar_width / 2,
#     group_width / 2 - bar_width / 2,
#     len(dataset_names)
# )

# for offset, name in zip(offsets, dataset_names):
#     values = data[name]

#     bars = ax.bar(
#         x + offset,
#         values,
#         width=bar_width,
#         label=name,
#         color=colors[name],
#         edgecolor="white",
#         linewidth=0.7,
#         alpha=0.95,
#     )

#     # Clean compact labels
#     for bar, value in zip(bars, values):
#         ax.text(
#             bar.get_x() + bar.get_width() / 2,
#             value + 0.055,
#             f"{value:.1f}",
#             ha="center",
#             va="bottom",
#             fontsize=5.8,
#             rotation=90,
#             color="black"
#         )

# # -----------------------------
# # Axes formatting
# # -----------------------------
# ax.set_xlabel("Number of Attention Heads")
# ax.set_ylabel("OA (%)")

# ax.set_xticks(x)
# ax.set_xticklabels([str(h) for h in heads])

# ax.set_ylim(95, 100)
# ax.set_yticks([95, 96, 97, 98, 99, 100])

# ax.grid(
#     True,
#     axis="y",
#     linestyle="-",
#     linewidth=0.35,
#     alpha=0.28
# )
# ax.set_axisbelow(True)

# ax.spines["top"].set_visible(False)
# ax.spines["right"].set_visible(False)

# # -----------------------------
# # Legend
# # -----------------------------
# ax.legend(
#     loc="upper center",
#     bbox_to_anchor=(0.5, 1.17),
#     ncol=5,
#     frameon=False,
#     columnspacing=1.0,
#     handlelength=1.2,
#     handletextpad=0.35,
# )

# plt.tight_layout()

# # -----------------------------
# # Save
# # -----------------------------
# plt.savefig("fig_diff_head_clean.pdf", bbox_inches="tight", pad_inches=0.02)
# plt.savefig("fig_diff_head_clean.png", dpi=600, bbox_inches="tight", pad_inches=0.02)

# plt.show()



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# ratios = np.array([5, 10, 15, 20, 25])

# oa = {
#     "HC":  np.array([98.63, 99.10, 99.30, 99.68, 99.53]),
#     "HH":  np.array([98.63, 98.70, 98.00, 99.18, 99.22]),
#     "PI":  np.array([98.85, 98.52, 99.30, 99.34, 99.35]),
#     "TH":  np.array([99.31, 99.46, 99.74, 99.39, 99.61]),
#     "QUH": np.array([98.37, 98.99, 99.20, 99.34, 99.59]),
# }

# training_time = {
#     "HC":  np.array([1613, 3600, 5200, 6900, 8100]),
#     "HH":  np.array([2407, 5500, 7600, 10600, 12100]),
#     "PI":  np.array([7200, 15500, 23000, 28000, 45000]),
#     "TH":  np.array([6400, 7900, 10800, 13500, 22000]),
#     "QUH": np.array([11500, 13500, 18500, 24500, 55000]),
# }

# markers = {
#     "HC": "o",
#     "HH": "s",
#     "PI": "^",
#     "TH": "D",
#     "QUH": "v",
# }

# line_styles = {
#     "HC": "-",
#     "HH": "--",
#     "PI": "-.",
#     "TH": ":",
#     "QUH": "-",
# }

# # Manual offsets to reduce overlapping labels.
# annotation_offsets = {
#     "HC":  [(5, 5), (5, 5), (5, 5), (5, 5), (5, -14)],
#     "HH":  [(5, 5), (5, 5), (5, 5), (5, 5), (5, 5)],
#     "PI":  [(5, 5), (5, -15), (5, 5), (5, 5), (5, 5)],
#     "TH":  [(5, -15), (5, 5), (5, 5), (5, 5), (5, 5)],
#     "QUH": [(5, 5), (5, 5), (5, 5), (5, 5), (5, 5)],
# }

# plt.rcParams.update({
#     "font.family": "serif",
#     "font.size": 11,
#     "axes.labelsize": 13,
#     "xtick.labelsize": 10,
#     "ytick.labelsize": 10,
#     "legend.fontsize": 11,
#     "pdf.fonttype": 42,
#     "ps.fonttype": 42,
# })

# fig, ax = plt.subplots(figsize=(9.2, 5.8))

# for dataset in oa:
#     ax.plot(
#         training_time[dataset],
#         oa[dataset],
#         marker=markers[dataset],
#         linestyle=line_styles[dataset],
#         linewidth=2.1,
#         markersize=7.5,
#         markerfacecolor="white",
#         markeredgewidth=1.7,
#         label=dataset,
#         zorder=3,
#     )

#     for i, (x, y, ratio) in enumerate(
#         zip(training_time[dataset], oa[dataset], ratios)
#     ):
#         ax.annotate(
#             f"{ratio}%",
#             xy=(x, y),
#             xytext=annotation_offsets[dataset][i],
#             textcoords="offset points",
#             fontsize=8.5,
#             annotation_clip=False,
#         )

# # Keep logarithmic spacing but display actual seconds.
# ax.set_xscale("log")
# ax.set_xlim(1400, 60000)

# # Explicit, readable training-time values.
# time_ticks = [2000, 5000, 10000, 20000, 50000]
# ax.set_xticks(time_ticks)

# ax.xaxis.set_major_formatter(
#     FuncFormatter(lambda value, position: f"{int(value):,}")
# )

# ax.set_xlabel("Training Time (s)")
# ax.set_ylabel("OA (%)")
# ax.set_ylim(97.8, 100.0)

# ax.grid(
#     True,
#     which="major",
#     linestyle="--",
#     linewidth=0.6,
#     alpha=0.35,
#     zorder=0,
# )

# ax.legend(
#     ncol=5,
#     loc="upper center",
#     bbox_to_anchor=(0.5, 1.15),
#     frameon=False,
#     handlelength=2.6,
#     columnspacing=1.7,
# )

# ax.text(
#     0.99,
#     0.02,
#     transform=ax.transAxes,
#     ha="right",
#     va="bottom",
#     fontsize=8.5,
# )

# for spine in ax.spines.values():
#     spine.set_linewidth(0.9)

# plt.tight_layout()

# plt.savefig(
#     "fig_accuracy_time_frontier_seconds.pdf",
#     bbox_inches="tight",
# )

# plt.savefig(
#     "fig_accuracy_time_frontier_seconds.png",
#     dpi=600,
#     bbox_inches="tight",
# )

# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.ticker import (
#     FuncFormatter,
#     MultipleLocator,
#     NullLocator,
# )

# # ---------------------------------------------------------
# # 1. Experimental results
# # ---------------------------------------------------------
# ratios = np.array([5, 10, 15, 20, 25])

# oa = {
#     "HC":  np.array([98.63, 99.10, 99.30, 99.68, 99.53]),
#     "HH":  np.array([98.63, 98.70, 98.00, 99.18, 99.22]),
#     "PI":  np.array([98.85, 98.52, 99.30, 99.34, 99.35]),
#     "TH":  np.array([99.31, 99.46, 99.74, 99.39, 99.61]),
#     "QUH": np.array([98.37, 98.99, 99.20, 99.34, 99.59]),
# }

# training_time = {
#     "HC":  np.array([1613, 3600, 5200, 6900, 8100]),
#     "HH":  np.array([2407, 5500, 7600, 10600, 12100]),
#     "PI":  np.array([7200, 15500, 23000, 28000, 45000]),
#     "TH":  np.array([6400, 7900, 10800, 13500, 22000]),
#     "QUH": np.array([11500, 13500, 18500, 24500, 55000]),
# }

# markers = {
#     "HC": "o",
#     "HH": "s",
#     "PI": "^",
#     "TH": "D",
#     "QUH": "v",
# }

# line_styles = {
#     "HC": "-",
#     "HH": "--",
#     "PI": "-.",
#     "TH": ":",
#     "QUH": "-",
# }

# # Only annotate the first and last training ratios.
# endpoint_offsets = {
#     "HC": {
#         "start": (7, 6),
#         "end": (7, -16),
#     },
#     "HH": {
#         "start": (7, 6),
#         "end": (7, 6),
#     },
#     "PI": {
#         "start": (7, 6),
#         "end": (7, 6),
#     },
#     "TH": {
#         "start": (7, -17),
#         "end": (7, 6),
#     },
#     "QUH": {
#         "start": (7, 6),
#         "end": (7, 6),
#     },
# }

# # ---------------------------------------------------------
# # 2. Figure style
# # ---------------------------------------------------------
# plt.rcParams.update({
#     "font.family": "serif",
#     "font.size": 11,
#     "axes.labelsize": 14,
#     "xtick.labelsize": 11,
#     "ytick.labelsize": 11,
#     "legend.fontsize": 11,
#     "pdf.fonttype": 42,
#     "ps.fonttype": 42,
# })

# fig, ax = plt.subplots(figsize=(9.4, 5.7))

# # ---------------------------------------------------------
# # 3. Plot dataset trajectories
# # ---------------------------------------------------------
# for dataset in oa:
#     times = training_time[dataset]
#     scores = oa[dataset]

#     ax.plot(
#         times,
#         scores,
#         marker=markers[dataset],
#         linestyle=line_styles[dataset],
#         linewidth=2.2,
#         markersize=7.5,
#         markerfacecolor="white",
#         markeredgewidth=1.7,
#         label=dataset,
#         zorder=3,
#     )

#     # Annotate only 5% and 25%.
#     ax.annotate(
#         "5%",
#         xy=(times[0], scores[0]),
#         xytext=endpoint_offsets[dataset]["start"],
#         textcoords="offset points",
#         fontsize=9,
#         fontweight="semibold",
#         annotation_clip=False,
#         zorder=4,
#     )

#     ax.annotate(
#         "25%",
#         xy=(times[-1], scores[-1]),
#         xytext=endpoint_offsets[dataset]["end"],
#         textcoords="offset points",
#         fontsize=9,
#         fontweight="semibold",
#         annotation_clip=False,
#         zorder=4,
#     )

# # ---------------------------------------------------------
# # 4. Training-time axis
# # ---------------------------------------------------------
# ax.set_xscale("log")
# ax.set_xlim(1400, 60000)

# time_ticks = [2000, 5000, 10000, 20000, 50000]
# ax.set_xticks(time_ticks)

# ax.xaxis.set_major_formatter(
#     FuncFormatter(lambda value, position: f"{int(value):,}")
# )

# # Remove dense logarithmic minor ticks.
# ax.xaxis.set_minor_locator(NullLocator())

# ax.set_xlabel("Training Time (s)")

# # ---------------------------------------------------------
# # 5. OA axis
# # ---------------------------------------------------------
# ax.set_ylabel("OA (%)")
# ax.set_ylim(97.8, 100.0)

# # Major horizontal ticks every 0.25%.
# ax.yaxis.set_major_locator(MultipleLocator(0.25))

# # One minor line between adjacent major grid lines.
# ax.yaxis.set_minor_locator(MultipleLocator(0.125))

# ax.yaxis.set_major_formatter(
#     FuncFormatter(lambda value, position: f"{value:.2f}")
# )

# # ---------------------------------------------------------
# # 6. Cleaner grid
# # ---------------------------------------------------------
# ax.set_axisbelow(True)

# # Major grid on both axes.
# ax.grid(
#     True,
#     which="major",
#     axis="both",
#     linestyle="--",
#     linewidth=0.65,
#     alpha=0.38,
# )

# # Minor grid only on the OA axis.
# ax.grid(
#     True,
#     which="minor",
#     axis="y",
#     linestyle=":",
#     linewidth=0.45,
#     alpha=0.18,
# )

# ax.tick_params(
#     axis="both",
#     which="major",
#     direction="out",
#     length=5,
#     width=0.9,
# )

# ax.tick_params(
#     axis="y",
#     which="minor",
#     direction="out",
#     length=3,
#     width=0.6,
# )

# # ---------------------------------------------------------
# # 7. Legend and direction note
# # ---------------------------------------------------------
# ax.legend(
#     ncol=5,
#     loc="lower center",
#     bbox_to_anchor=(0.5, 1.02),
#     frameon=False,
#     handlelength=2.8,
#     columnspacing=1.7,
# )

# ax.text(
#     0.985,
#     0.025,
#     r"Trajectories progress from 5\% to 25\%",
#     transform=ax.transAxes,
#     ha="right",
#     va="bottom",
#     fontsize=8.5,
# )

# # ---------------------------------------------------------
# # 8. Border formatting
# # ---------------------------------------------------------
# for spine in ax.spines.values():
#     spine.set_linewidth(0.9)

# plt.tight_layout()

# # ---------------------------------------------------------
# # 9. Save publication-quality outputs
# # ---------------------------------------------------------
# plt.savefig(
#     "fig_accuracy_time_frontier_clean.pdf",
#     bbox_inches="tight",
# )

# plt.savefig(
#     "fig_accuracy_time_frontier_clean.png",
#     dpi=600,
#     bbox_inches="tight",
# )

# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt

# from matplotlib.colors import LogNorm
# from matplotlib.cm import ScalarMappable
# from matplotlib.ticker import FuncFormatter
# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
# from mpl_toolkits.mplot3d.art3d import PolyCollection


# # =========================================================
# # 1. Experimental results
# # =========================================================
# ratios = np.array([5, 10, 15, 20, 25])

# dataset_keys = [
#     "HC",
#     "HH",
#     "PI",
#     "TH",
#     "QUH",
# ]

# dataset_labels = [
#     "HC",
#     "HH",
#     "PI",
#     "TH",
#     "QUH",
# ]

# oa = {
#     "HC": np.array([
#         98.63,
#         99.10,
#         99.30,
#         99.68,
#         99.53,
#     ]),
#     "HH": np.array([
#         98.63,
#         98.70,
#         98.00,
#         99.18,
#         99.22,
#     ]),
#     "PI": np.array([
#         98.85,
#         98.52,
#         99.30,
#         99.34,
#         99.35,
#     ]),
#     "TH": np.array([
#         99.31,
#         99.46,
#         99.74,
#         99.39,
#         99.61,
#     ]),
#     "QUH": np.array([
#         98.37,
#         98.99,
#         99.20,
#         99.34,
#         99.59,
#     ]),
# }

# training_time = {
#     "HC": np.array([
#         1613,
#         3600,
#         5200,
#         6900,
#         8100,
#     ]),
#     "HH": np.array([
#         2407,
#         5500,
#         7600,
#         10600,
#         12100,
#     ]),
#     "PI": np.array([
#         7200,
#         15500,
#         23000,
#         28000,
#         45000,
#     ]),
#     "TH": np.array([
#         6400,
#         7900,
#         10800,
#         13500,
#         22000,
#     ]),
#     "QUH": np.array([
#         11500,
#         13500,
#         18500,
#         24500,
#         55000,
#     ]),
# }


# # =========================================================
# # 2. Configuration
# # =========================================================
# oa_floor = 97.8

# plt.rcParams.update({
#     "font.family": "serif",
#     "font.size": 10,
#     "axes.labelsize": 12,
#     "axes.titlesize": 13,
#     "xtick.labelsize": 9,
#     "ytick.labelsize": 9,
#     "pdf.fonttype": 42,
#     "ps.fonttype": 42,
# })


# # =========================================================
# # 3. Validate input
# # =========================================================
# for key in dataset_keys:
#     if len(oa[key]) != len(ratios):
#         raise ValueError(
#             f"{key}: invalid number of OA values."
#         )

#     if len(training_time[key]) != len(ratios):
#         raise ValueError(
#             f"{key}: invalid number of training-time values."
#         )

#     if np.any(training_time[key] <= 0):
#         raise ValueError(
#             f"{key}: training times must be positive."
#         )


# # =========================================================
# # 4. Prepare ribbon polygons
# # =========================================================
# dataset_positions = np.arange(len(dataset_keys))

# ribbon_vertices = []

# for key in dataset_keys:
#     values = oa[key]

#     # Start from the OA floor, follow the OA curve,
#     # and return to the floor to form a closed ribbon.
#     vertices = [
#         (ratios[0], oa_floor),
#         *list(zip(ratios, values)),
#         (ratios[-1], oa_floor),
#     ]

#     ribbon_vertices.append(vertices)


# # Use restrained shades for the ribbons.
# ribbon_shades = plt.cm.Greys(
#     np.linspace(0.22, 0.48, len(dataset_keys))
# )

# ribbons = PolyCollection(
#     ribbon_vertices,
#     facecolors=ribbon_shades,
#     edgecolors=(0.25, 0.25, 0.25, 0.75),
#     linewidths=0.9,
#     alpha=0.45,
# )


# # =========================================================
# # 5. Create figure
# # =========================================================
# fig = plt.figure(
#     figsize=(11.5, 7.6),
# )

# ax = fig.add_subplot(
#     111,
#     projection="3d",
# )

# ax.add_collection3d(
#     ribbons,
#     zs=dataset_positions,
#     zdir="y",
# )


# # =========================================================
# # 6. Training-time normalization
# # =========================================================
# all_times = np.concatenate([
#     training_time[key]
#     for key in dataset_keys
# ])

# time_norm = LogNorm(
#     vmin=all_times.min(),
#     vmax=all_times.max(),
# )

# time_cmap = plt.get_cmap("viridis")

# scatter_handle = None


# # =========================================================
# # 7. Draw OA profiles and time-coded markers
# # =========================================================
# for dataset_index, key in enumerate(dataset_keys):

#     oa_values = oa[key]
#     time_values = training_time[key]

#     y_values = np.full(
#         len(ratios),
#         dataset_index,
#         dtype=float,
#     )

#     # Main OA profile.
#     profile_line, = ax.plot(
#         ratios,
#         y_values,
#         oa_values,
#         linewidth=2.1,
#         marker=None,
#         zorder=4,
#     )

#     # Markers are colored according to training time.
#     scatter_handle = ax.scatter(
#         ratios,
#         y_values,
#         oa_values,
#         c=time_values,
#         cmap=time_cmap,
#         norm=time_norm,
#         s=82,
#         edgecolors="white",
#         linewidths=1.0,
#         depthshade=False,
#         zorder=6,
#     )

#     # Outline the best OA setting.
#     best_index = int(
#         np.argmax(oa_values)
#     )

#     ax.scatter(
#         [ratios[best_index]],
#         [dataset_index],
#         [oa_values[best_index]],
#         s=170,
#         facecolors="none",
#         edgecolors="black",
#         linewidths=1.6,
#         depthshade=False,
#         zorder=7,
#     )

#     # Subtle vertical stems.
#     for ratio, oa_value in zip(
#         ratios,
#         oa_values,
#     ):
#         ax.plot(
#             [ratio, ratio],
#             [dataset_index, dataset_index],
#             [oa_floor, oa_value],
#             color=profile_line.get_color(),
#             linestyle=":",
#             linewidth=0.55,
#             alpha=0.22,
#             zorder=1,
#         )


# # =========================================================
# # 8. Axes
# # =========================================================
# ax.set_xlabel(
#     "Training Ratio (%)",
#     labelpad=13,
# )

# ax.set_ylabel(
#     "Dataset",
#     labelpad=15,
# )

# ax.set_zlabel(
#     "OA (%)",
#     labelpad=11,
# )

# ax.set_xticks(ratios)
# ax.set_xlim(4, 26)

# ax.set_yticks(dataset_positions)
# ax.set_yticklabels(dataset_labels)
# ax.set_ylim(-0.45, len(dataset_keys) - 0.55)

# ax.set_zlim(oa_floor, 100.0)

# ax.set_zticks([
#     98.0,
#     98.5,
#     99.0,
#     99.5,
#     100.0,
# ])

# ax.zaxis.set_major_formatter(
#     FuncFormatter(
#         lambda value, position: f"{value:.1f}"
#     )
# )


# # =========================================================
# # 9. View and aspect ratio
# # =========================================================
# ax.view_init(
#     elev=28,
#     azim=-58,
# )

# ax.set_box_aspect(
#     (
#         1.55,   # training-ratio dimension
#         1.20,   # dataset dimension
#         0.80,   # OA dimension
#     )
# )


# # =========================================================
# # 10. Pane and grid styling
# # =========================================================
# ax.xaxis.pane.set_facecolor(
#     (0.98, 0.98, 0.98, 1.0)
# )

# ax.yaxis.pane.set_facecolor(
#     (0.98, 0.98, 0.98, 1.0)
# )

# ax.zaxis.pane.set_facecolor(
#     (1.00, 1.00, 1.00, 1.0)
# )

# ax.xaxis.pane.set_alpha(0.18)
# ax.yaxis.pane.set_alpha(0.18)
# ax.zaxis.pane.set_alpha(0.06)

# for axis in (
#     ax.xaxis,
#     ax.yaxis,
#     ax.zaxis,
# ):
#     axis._axinfo["grid"]["color"] = (
#         0.55,
#         0.55,
#         0.55,
#         0.28,
#     )
#     axis._axinfo["grid"]["linewidth"] = 0.55
#     axis._axinfo["grid"]["linestyle"] = "--"

# ax.grid(True)


# # =========================================================
# # 11. Color bar
# # =========================================================
# scalar_mappable = ScalarMappable(
#     norm=time_norm,
#     cmap=time_cmap,
# )

# scalar_mappable.set_array([])

# colorbar = fig.colorbar(
#     scalar_mappable,
#     ax=ax,
#     shrink=0.68,
#     pad=0.09,
#     aspect=24,
# )

# colorbar.set_label(
#     "Training Time (s)",
# )

# time_ticks = np.array([
#     2000,
#     5000,
#     10000,
#     20000,
#     50000,
# ])

# colorbar.set_ticks(time_ticks)

# colorbar.ax.yaxis.set_major_formatter(
#     FuncFormatter(
#         lambda value, position: f"{int(value):,}"
#     )
# )


# # =========================================================
# # 12. Title
# # =========================================================
# # ax.set_title(
# #     "Training-Ratio Accuracy Profiles and Computational Cost",
# #     fontweight="bold",
# #     pad=22,
# # )

# # fig.text(
# #     0.50,
# #     0.94,
# #     "Ribbon height represents OA, marker color represents "
# #     "training time, and outlined markers indicate the "
# #     "highest OA for each dataset.",
# #     ha="center",
# #     fontsize=9.5,
# # )


# # =========================================================
# # 13. Save
# # =========================================================
# fig.subplots_adjust(
#     left=0.02,
#     right=0.88,
#     bottom=0.04,
#     top=0.90,
# )

# fig.savefig(
#     "fig_training_ratio_3d_waterfall.pdf",
#     bbox_inches="tight",
# )

# fig.savefig(
#     "fig_training_ratio_3d_waterfall.png",
#     dpi=600,
#     bbox_inches="tight",
# )

# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt

# from matplotlib.colors import to_rgba
# from matplotlib.lines import Line2D
# from matplotlib.patches import Patch
# from matplotlib.ticker import FuncFormatter, MultipleLocator


# # =========================================================
# # 1. Experimental results
# # =========================================================
# ratios = np.array([5, 10, 15, 20, 25])
# ratio_labels = [f"{ratio}%" for ratio in ratios]

# datasets = [
#     "HC",
#     "HH",
#     "PI",
#     "TH",
#     "QUH",
# ]

# dataset_names = {
#     "HC": "WHU-Hi-HanChuan",
#     "HH": "WHU-Hi-HongHu",
#     "PI": "Ping'an",
#     "TH": "Tangdaowan",
#     "QUH": "Qingyun",
# }

# oa = {
#     "HC": np.array([
#         98.63,
#         99.10,
#         99.30,
#         99.68,
#         99.53,
#     ]),
#     "HH": np.array([
#         98.63,
#         98.70,
#         98.00,
#         99.18,
#         99.22,
#     ]),
#     "PI": np.array([
#         98.85,
#         98.52,
#         99.30,
#         99.34,
#         99.35,
#     ]),
#     "TH": np.array([
#         99.31,
#         99.46,
#         99.74,
#         99.39,
#         99.61,
#     ]),
#     "QUH": np.array([
#         98.37,
#         98.99,
#         99.20,
#         99.34,
#         99.59,
#     ]),
# }

# training_time = {
#     "HC": np.array([
#         1613,
#         3600,
#         5200,
#         6900,
#         8100,
#     ]),
#     "HH": np.array([
#         2407,
#         5500,
#         7600,
#         10600,
#         12100,
#     ]),
#     "PI": np.array([
#         7200,
#         15500,
#         23000,
#         28000,
#         45000,
#     ]),
#     "TH": np.array([
#         6400,
#         7900,
#         10800,
#         13500,
#         22000,
#     ]),
#     "QUH": np.array([
#         11500,
#         13500,
#         18500,
#         24500,
#         55000,
#     ]),
# }


# # =========================================================
# # 2. Validate the results
# # =========================================================
# for dataset in datasets:

#     if dataset not in oa:
#         raise KeyError(
#             f"Missing OA results for dataset '{dataset}'."
#         )

#     if dataset not in training_time:
#         raise KeyError(
#             f"Missing training-time results for '{dataset}'."
#         )

#     if len(oa[dataset]) != len(ratios):
#         raise ValueError(
#             f"{dataset}: expected {len(ratios)} OA values, "
#             f"but received {len(oa[dataset])}."
#         )

#     if len(training_time[dataset]) != len(ratios):
#         raise ValueError(
#             f"{dataset}: expected {len(ratios)} training-time "
#             f"values, but received {len(training_time[dataset])}."
#         )

#     if np.any(training_time[dataset] <= 0):
#         raise ValueError(
#             f"{dataset}: training times must be positive."
#         )

#     if np.any((oa[dataset] < 0) | (oa[dataset] > 100)):
#         raise ValueError(
#             f"{dataset}: OA values must be between 0 and 100."
#         )


# # =========================================================
# # 3. Colorblind-safe dataset encoding
# # =========================================================
# dataset_colors = {
#     "HC": "#0072B2",   # blue
#     "HH": "#E69F00",   # orange
#     "PI": "#009E73",   # bluish green
#     "TH": "#D55E00",   # vermilion
#     "QUH": "#CC79A7",  # reddish purple
# }

# dataset_markers = {
#     "HC": "o",
#     "HH": "s",
#     "PI": "^",
#     "TH": "D",
#     "QUH": "v",
# }

# dataset_line_styles = {
#     "HC": "-",
#     "HH": "--",
#     "PI": "-.",
#     "TH": (0, (3, 1, 1, 1)),
#     "QUH": (0, (6, 2)),
# }

# # Bar hatches improve grayscale readability.
# dataset_hatches = {
#     "HC": "",
#     "HH": "//",
#     "PI": "\\\\",
#     "TH": "xx",
#     "QUH": "..",
# }


# # =========================================================
# # 4. Figure controls
# # =========================================================

# # OA bars use a truncated baseline because all results
# # lie in a narrow range. This must be stated in the caption.
# oa_floor = 97.8

# # Value annotations are disabled by default because 25 OA
# # labels and 25 time labels would make the figure crowded.
# show_oa_labels = False
# show_time_labels = False

# # IEEE papers generally do not require an internal title.
# show_internal_title = False

# output_pdf = "fig_all_datasets_oa_time_combo.pdf"
# output_png = "fig_all_datasets_oa_time_combo.png"


# # =========================================================
# # 5. Publication typography
# # =========================================================
# plt.rcParams.update({
#     "font.family": "serif",
#     "font.size": 10,
#     "axes.labelsize": 13,
#     "axes.titlesize": 13,
#     "xtick.labelsize": 10,
#     "ytick.labelsize": 10,
#     "legend.fontsize": 9.2,
#     "pdf.fonttype": 42,
#     "ps.fonttype": 42,
# })


# # =========================================================
# # 6. Create figure and secondary axis
# # =========================================================
# fig, ax_oa = plt.subplots(
#     figsize=(12.8, 6.2),
# )

# ax_time = ax_oa.twinx()

# fig.patch.set_facecolor("white")
# ax_oa.set_facecolor("white")


# # =========================================================
# # 7. Grouped-bar positions
# # =========================================================
# x = np.arange(len(ratios))

# group_width = 0.78
# bar_width = group_width / len(datasets)

# offsets = (
#     np.arange(len(datasets))
#     - (len(datasets) - 1) / 2
# ) * bar_width


# # =========================================================
# # 8. Plot OA bars
# # =========================================================
# bar_containers = {}

# for dataset_index, dataset in enumerate(datasets):

#     color = dataset_colors[dataset]
#     bar_positions = x + offsets[dataset_index]

#     bar_heights = oa[dataset] - oa_floor

#     bars = ax_oa.bar(
#         bar_positions,
#         bar_heights,
#         width=bar_width * 0.88,
#         bottom=oa_floor,
#         facecolor=to_rgba(color, 0.34),
#         edgecolor=color,
#         linewidth=1.15,
#         hatch=dataset_hatches[dataset],
#         label=dataset_names[dataset],
#         zorder=3,
#     )

#     bar_containers[dataset] = bars

#     if show_oa_labels:
#         for bar, value in zip(bars, oa[dataset]):
#             ax_oa.annotate(
#                 f"{value:.2f}",
#                 xy=(
#                     bar.get_x() + bar.get_width() / 2,
#                     value,
#                 ),
#                 xytext=(0, 3),
#                 textcoords="offset points",
#                 ha="center",
#                 va="bottom",
#                 fontsize=7.2,
#                 color=color,
#                 rotation=90,
#                 annotation_clip=False,
#             )


# # =========================================================
# # 9. Plot training-time lines
# # =========================================================
# for dataset in datasets:

#     color = dataset_colors[dataset]

#     ax_time.plot(
#         x,
#         training_time[dataset],
#         color=color,
#         linestyle=dataset_line_styles[dataset],
#         linewidth=2.25,
#         marker=dataset_markers[dataset],
#         markersize=7.0,
#         markerfacecolor="white",
#         markeredgecolor=color,
#         markeredgewidth=1.65,
#         zorder=5,
#     )

#     if show_time_labels:
#         for x_position, value in zip(
#             x,
#             training_time[dataset],
#         ):
#             ax_time.annotate(
#                 f"{value:,}",
#                 xy=(x_position, value),
#                 xytext=(4, 5),
#                 textcoords="offset points",
#                 ha="left",
#                 va="bottom",
#                 fontsize=7.0,
#                 color=color,
#                 annotation_clip=False,
#             )


# # =========================================================
# # 10. X-axis configuration
# # =========================================================
# ax_oa.set_xticks(x)
# ax_oa.set_xticklabels(ratio_labels)

# ax_oa.set_xlim(
#     -0.55,
#     len(ratios) - 0.45,
# )

# ax_oa.set_xlabel(
#     "Training Ratio (%)"
# )


# # =========================================================
# # 11. Left axis: OA
# # =========================================================
# ax_oa.set_ylim(
#     oa_floor,
#     100.0,
# )

# ax_oa.set_ylabel(
#     "Overall Accuracy (%)"
# )

# ax_oa.yaxis.set_major_locator(
#     MultipleLocator(0.25)
# )

# ax_oa.yaxis.set_minor_locator(
#     MultipleLocator(0.125)
# )

# ax_oa.yaxis.set_major_formatter(
#     FuncFormatter(
#         lambda value, position: f"{value:.2f}"
#     )
# )


# # =========================================================
# # 12. Right axis: training time
# # =========================================================
# ax_time.set_ylim(
#     0,
#     60000,
# )

# ax_time.set_ylabel(
#     "Training Time (s)"
# )

# ax_time.set_yticks([
#     0,
#     10000,
#     20000,
#     30000,
#     40000,
#     50000,
#     60000,
# ])

# ax_time.yaxis.set_major_formatter(
#     FuncFormatter(
#         lambda value, position: f"{int(value):,}"
#     )
# )


# # =========================================================
# # 13. Grid
# # =========================================================
# ax_oa.set_axisbelow(True)

# ax_oa.grid(
#     True,
#     which="major",
#     axis="y",
#     color="0.78",
#     linestyle="--",
#     linewidth=0.62,
#     alpha=0.58,
# )

# ax_oa.grid(
#     True,
#     which="minor",
#     axis="y",
#     color="0.88",
#     linestyle=":",
#     linewidth=0.38,
#     alpha=0.52,
# )

# # Vertical separators between ratio groups.
# for separator in np.arange(len(ratios) - 1) + 0.5:
#     ax_oa.axvline(
#         separator,
#         color="0.90",
#         linewidth=0.55,
#         zorder=0,
#     )


# # =========================================================
# # 14. Dataset legend
# # =========================================================
# dataset_handles = [
#     Line2D(
#         [0],
#         [0],
#         color=dataset_colors[dataset],
#         linestyle=dataset_line_styles[dataset],
#         linewidth=2.2,
#         marker=dataset_markers[dataset],
#         markerfacecolor="white",
#         markeredgecolor=dataset_colors[dataset],
#         markeredgewidth=1.4,
#         markersize=6.5,
#         label=dataset_names[dataset],
#     )
#     for dataset in datasets
# ]

# dataset_legend = ax_oa.legend(
#     handles=dataset_handles,
#     title="Dataset",
#     loc="lower center",
#     bbox_to_anchor=(0.5, 1.015),
#     ncol=5,
#     frameon=False,
#     handlelength=2.8,
#     handletextpad=0.55,
#     columnspacing=1.35,
# )

# ax_oa.add_artist(dataset_legend)


# # =========================================================
# # 15. Measurement legend
# # =========================================================
# measurement_handles = [
#     Patch(
#         facecolor=to_rgba("0.40", 0.28),
#         edgecolor="0.25",
#         linewidth=1.0,
#         label="OA (bars, left axis)",
#     ),
#     Line2D(
#         [0],
#         [0],
#         color="0.25",
#         linestyle="-",
#         linewidth=2.1,
#         marker="o",
#         markerfacecolor="white",
#         markeredgecolor="0.25",
#         markersize=6,
#         label="Training time (lines, right axis)",
#     ),
# ]

# ax_oa.legend(
#     handles=measurement_handles,
#     loc="upper left",
#     frameon=False,
#     handlelength=2.5,
#     borderaxespad=0.7,
# )


# # =========================================================
# # 16. Optional internal title
# # =========================================================
# if show_internal_title:
#     ax_oa.set_title(
#         "Effect of the Training Ratio on OA and "
#         "Training Time Across Five HSI Datasets",
#         fontweight="bold",
#         pad=42,
#     )


# # =========================================================
# # 17. Borders and tick formatting
# # =========================================================
# for spine in ax_oa.spines.values():
#     spine.set_linewidth(0.9)
#     spine.set_color("0.18")

# for spine in ax_time.spines.values():
#     spine.set_linewidth(0.9)
#     spine.set_color("0.18")

# ax_oa.tick_params(
#     axis="both",
#     which="major",
#     direction="out",
#     length=5,
#     width=0.9,
# )

# ax_oa.tick_params(
#     axis="y",
#     which="minor",
#     direction="out",
#     length=3,
#     width=0.55,
# )

# ax_time.tick_params(
#     axis="y",
#     which="major",
#     direction="out",
#     length=5,
#     width=0.9,
# )


# # =========================================================
# # 18. Final layout and save
# # =========================================================
# fig.subplots_adjust(
#     left=0.08,
#     right=0.91,
#     bottom=0.13,
#     top=0.83,
# )

# fig.savefig(
#     output_pdf,
#     bbox_inches="tight",
# )

# fig.savefig(
#     output_png,
#     dpi=600,
#     bbox_inches="tight",
# )

# plt.show()



import math

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MultipleLocator


# =========================================================
# 1. Experimental data
# =========================================================
RATIOS = np.array([5, 10, 15, 20, 25])
DATASETS = ("HC", "HH", "PI", "TH", "QUH")

DISPLAY_NAMES = {
    "HC": "WHU-HC",
    "HH": "WHU-HH",
    "PI": "Ping'an",
    "TH": "Tangdaowan",
    "QUH": "Qingyun",
}

OA = {
    "HC": np.array([98.63, 99.10, 99.30, 99.68, 99.53]),
    "HH": np.array([98.63, 98.70, 98.00, 99.18, 99.22]),
    "PI": np.array([98.85, 98.52, 99.30, 99.34, 99.35]),
    "TH": np.array([99.31, 99.46, 99.74, 99.39, 99.61]),
    "QUH": np.array([98.37, 98.99, 99.20, 99.34, 99.59]),
}

TRAINING_TIME = {
    "HC": np.array([1613, 3600, 5200, 6900, 8100]),
    "HH": np.array([2407, 5500, 7600, 10600, 12100]),
    "PI": np.array([7200, 15500, 23000, 28000, 45000]),
    "TH": np.array([6400, 7900, 10800, 13500, 22000]),
    "QUH": np.array([11500, 13500, 18500, 24500, 55000]),
}


# =========================================================
# 2. Visual encoding
# =========================================================
# Colorblind-safe palette.
COLORS = {
    "HC": "#0072B2",
    "HH": "#E69F00",
    "PI": "#009E73",
    "TH": "#D55E00",
    "QUH": "#CC79A7",
}

MARKERS = {
    "HC": "o",
    "HH": "s",
    "PI": "^",
    "TH": "D",
    "QUH": "P",  # pentagon-like filled-plus marker; avoids arrow appearance
}

LINE_STYLES = {
    "HC": "-",
    "HH": "--",
    "PI": "-.",
    "TH": (0, (3, 1, 1, 1)),
    "QUH": (0, (6, 2)),
}


# =========================================================
# 3. Figure configuration
# =========================================================
OA_FLOOR = 97.8
OA_LIMIT = 100.0

TIME_TICK_STEP = 10_000
TIME_HEADROOM = 1.20

OUTPUT_PDF = "fig_training_ratio_oa_time_clean.pdf"
OUTPUT_PNG = "fig_training_ratio_oa_time_clean.png"


# =========================================================
# 4. Validation
# =========================================================
def validate_data() -> None:
    """Validate result dimensions and numerical ranges."""

    expected_length = len(RATIOS)

    for dataset in DATASETS:
        if dataset not in OA:
            raise KeyError(f"Missing OA results for {dataset}.")

        if dataset not in TRAINING_TIME:
            raise KeyError(
                f"Missing training-time results for {dataset}."
            )

        oa_values = np.asarray(OA[dataset], dtype=float)
        time_values = np.asarray(
            TRAINING_TIME[dataset],
            dtype=float,
        )

        if len(oa_values) != expected_length:
            raise ValueError(
                f"{dataset}: expected {expected_length} OA values, "
                f"received {len(oa_values)}."
            )

        if len(time_values) != expected_length:
            raise ValueError(
                f"{dataset}: expected {expected_length} training-time "
                f"values, received {len(time_values)}."
            )

        if not np.all(np.isfinite(oa_values)):
            raise ValueError(
                f"{dataset}: OA contains non-finite values."
            )

        if not np.all(np.isfinite(time_values)):
            raise ValueError(
                f"{dataset}: training time contains "
                "non-finite values."
            )

        if np.any((oa_values < 0) | (oa_values > 100)):
            raise ValueError(
                f"{dataset}: OA must lie within [0, 100]."
            )

        if np.any(time_values <= 0):
            raise ValueError(
                f"{dataset}: training time must be positive."
            )


# =========================================================
# 5. Utilities
# =========================================================
def calculate_time_limit() -> int:
    """
    Return a rounded training-time limit with sufficient
    headroom above the largest observed value.
    """

    maximum_time = max(
        int(np.max(TRAINING_TIME[dataset]))
        for dataset in DATASETS
    )

    target_limit = maximum_time * TIME_HEADROOM

    return int(
        math.ceil(target_limit / TIME_TICK_STEP)
        * TIME_TICK_STEP
    )


def configure_style() -> None:
    """Configure publication-oriented typography."""

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9.4,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def create_legend_handles() -> list[Line2D]:
    """Construct compact dataset legend entries."""

    return [
        Line2D(
            [0],
            [0],
            color=COLORS[dataset],
            linestyle=LINE_STYLES[dataset],
            linewidth=2.1,
            marker=MARKERS[dataset],
            markerfacecolor="white",
            markeredgecolor=COLORS[dataset],
            markeredgewidth=1.4,
            markersize=6.5,
            label=DISPLAY_NAMES[dataset],
        )
        for dataset in DATASETS
    ]


# =========================================================
# 6. Main plotting function
# =========================================================
def plot_results() -> None:
    """Plot grouped OA bars and training-time trajectories."""

    validate_data()
    configure_style()

    time_limit = calculate_time_limit()

    fig, ax_oa = plt.subplots(
        figsize=(12.0, 5.7),
    )

    ax_time = ax_oa.twinx()

    fig.patch.set_facecolor("white")
    ax_oa.set_facecolor("white")

    # Keep the time curves above the bars without hiding them.
    ax_time.set_zorder(ax_oa.get_zorder() + 1)
    ax_time.patch.set_visible(False)

    x_positions = np.arange(len(RATIOS))

    group_width = 0.76
    bar_width = group_width / len(DATASETS)

    dataset_offsets = (
        np.arange(len(DATASETS))
        - (len(DATASETS) - 1) / 2
    ) * bar_width

    # -----------------------------------------------------
    # OA bars
    # -----------------------------------------------------
    for dataset_index, dataset in enumerate(DATASETS):
        values = OA[dataset]
        positions = (
            x_positions
            + dataset_offsets[dataset_index]
        )

        bars = ax_oa.bar(
            positions,
            values - OA_FLOOR,
            bottom=OA_FLOOR,
            width=bar_width * 0.80,
            facecolor=to_rgba(
                COLORS[dataset],
                0.38,
            ),
            edgecolor=COLORS[dataset],
            linewidth=0.95,
            zorder=3,
        )

        # Highlight the highest OA obtained for this dataset.
        best_index = int(np.argmax(values))

        bars[best_index].set_edgecolor("black")
        bars[best_index].set_linewidth(1.45)
        bars[best_index].set_alpha(1.0)

    # -----------------------------------------------------
    # Training-time lines
    # -----------------------------------------------------
    for dataset in DATASETS:
        ax_time.plot(
            x_positions,
            TRAINING_TIME[dataset],
            color=COLORS[dataset],
            linestyle=LINE_STYLES[dataset],
            linewidth=2.05,
            marker=MARKERS[dataset],
            markersize=6.8,
            markerfacecolor="white",
            markeredgecolor=COLORS[dataset],
            markeredgewidth=1.55,
            clip_on=True,
            zorder=5,
        )

    # -----------------------------------------------------
    # Horizontal axis
    # -----------------------------------------------------
    ax_oa.set_xticks(x_positions)

    ax_oa.set_xticklabels([
        f"{ratio}%"
        for ratio in RATIOS
    ])

    ax_oa.set_xlim(
        -0.55,
        len(RATIOS) - 0.45,
    )

    ax_oa.set_xlabel(
        "Training Ratio (%)"
    )

    # -----------------------------------------------------
    # OA axis
    # -----------------------------------------------------
    ax_oa.set_ylim(
        OA_FLOOR,
        OA_LIMIT,
    )

    ax_oa.set_ylabel(
        "Overall Accuracy (%)"
    )

    ax_oa.yaxis.set_major_locator(
        MultipleLocator(0.25)
    )

    ax_oa.yaxis.set_minor_locator(
        MultipleLocator(0.125)
    )

    ax_oa.yaxis.set_major_formatter(
        FuncFormatter(
            lambda value, _: f"{value:.2f}"
        )
    )

    # -----------------------------------------------------
    # Training-time axis
    # -----------------------------------------------------
    ax_time.set_ylim(
        0,
        time_limit,
    )

    ax_time.set_yticks(
        np.arange(
            0,
            time_limit + TIME_TICK_STEP,
            TIME_TICK_STEP,
        )
    )

    ax_time.set_ylabel(
        "Training Time (s)"
    )

    ax_time.yaxis.set_major_formatter(
        FuncFormatter(
            lambda value, _: f"{int(value):,}"
        )
    )

    # -----------------------------------------------------
    # Grid
    # -----------------------------------------------------
    ax_oa.set_axisbelow(True)

    ax_oa.grid(
        axis="y",
        which="major",
        linestyle="--",
        linewidth=0.58,
        color="0.80",
        alpha=0.55,
    )

    ax_oa.grid(
        axis="y",
        which="minor",
        linestyle=":",
        linewidth=0.34,
        color="0.90",
        alpha=0.48,
    )

    # -----------------------------------------------------
    # Legend placed directly above the plotting area
    # -----------------------------------------------------
    ax_oa.legend(
        handles=create_legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 1.008),
        ncol=5,
        frameon=False,
        handlelength=2.6,
        handletextpad=0.50,
        columnspacing=1.25,
        borderaxespad=0.0,
    )

    # -----------------------------------------------------
    # Axis appearance
    # -----------------------------------------------------
    for axis in (ax_oa, ax_time):
        for spine in axis.spines.values():
            spine.set_linewidth(0.85)
            spine.set_color("0.20")

    ax_oa.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=5,
        width=0.85,
    )

    ax_oa.tick_params(
        axis="y",
        which="minor",
        direction="out",
        length=3,
        width=0.50,
    )

    ax_time.tick_params(
        axis="y",
        which="major",
        direction="out",
        length=5,
        width=0.85,
    )

    # Less whitespace above the legend.
    fig.subplots_adjust(
        left=0.08,
        right=0.91,
        bottom=0.14,
        top=0.895,
    )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------
    fig.savefig(
        OUTPUT_PDF,
        bbox_inches="tight",
    )

    fig.savefig(
        OUTPUT_PNG,
        dpi=600,
        bbox_inches="tight",
    )

    plt.show()


if __name__ == "__main__":
    plot_results()