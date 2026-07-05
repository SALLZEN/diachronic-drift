import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

mpl.rcParams["font.family"] = "Stix Two Math"

colors = {
    "A": "#2a9d8f",
    "B": "#DFAA27",
    "C": "#e76f51",
    "A_secondary": "#3ac7b6",
    "B_secondary": "#e9c46a",
    "C_secondary": "#eb8b73",
    "none": "#cccccc",
}

gruvbox_l = ["#ea6962", "#e78a4e", "#a9b665", "#7daea3", "#d3869b", "#d8a657", "#89b482"]
gruvbox_d = ["#c14a4a", "#b47109", "#4c7a5d", "#45707a", "#945e80", "#c35e0a", "#6c782e"]

# blueprint palette
blueprint_l = [
    "#7FA6BF",  # pale print blue
    "#4F7FA3",  # washed cyan blue
    "#2E5D87",  # mid cyanotype blue
    "#163E63",  # classic blueprint blue
    "#0B1F33",  # deep archival blue
]
blueprint_d = [
    "#A9C0D1",  # brighter pale accent for visibility on dark bg
    "#7FA6BF",
    "#4F7FA3",
    "#2E5D87",
    "#163E63",
]

DARK_BG = "#1b1b1b"
LIGHT_BG = "#fcf8e6"

BLUEPRINT_BG = "#0B1F33"
BLUEPRINT_FG = "#F3F7F8"
BLUEPRINT_FG_ALT = "#F6F1E8"
BLUEPRINT_BG_ALT = "#163E63"

gbox_light = ListedColormap(gruvbox_l, name="gbox_light").reversed()
gbox_dark = ListedColormap(gruvbox_d, name="gbox_dark").reversed()

blueprint_light = ListedColormap(blueprint_l, name="blueprint_light")
blueprint_dark = ListedColormap(blueprint_d, name="blueprint_dark")

themes = {
    "dark": dict(
        cmap=gbox_dark,
        bg=DARK_BG,
        fg=LIGHT_BG,
        fg_alt="#f9f5d7",
        bg_alt="#282828",
    ),
    "light": dict(
        cmap=gbox_light,
        bg=LIGHT_BG,
        fg=DARK_BG,
        bg_alt="#f9f5d7",
        fg_alt="#282828",
    ),
    "paper": dict(
        cmap=gbox_light,
        bg="white",
        fg="black",
        bg_alt="#f9f5d7",
        fg_alt="#282828",
    ),
    "blueprint": dict(
        cmap=blueprint_dark,
        bg=BLUEPRINT_BG,
        fg=BLUEPRINT_FG,
        fg_alt=BLUEPRINT_FG_ALT,
        bg_alt=BLUEPRINT_BG_ALT,
        grid_alpha=0.16,
        tick_length=2,
        tick_width=0.6,
    ),
    "backup": dict(
        cmap=plt.get_cmap("tab10"),
        bg="white",
        fg="black",
        bg_alt="#f2f2f2",
        fg_alt="#222222",
    ),
}

CAT_LABELS = {
    "A": "A | Reference",
    "B": "B | Role",
    "C": "C | Parameter",
}


def apply_theme_rcparams(theme):
    grid_alpha = theme.get("grid_alpha", 0.20)

    mpl.rcParams["figure.facecolor"] = theme["bg"]
    mpl.rcParams["savefig.facecolor"] = theme["bg"]
    mpl.rcParams["axes.facecolor"] = theme["bg"]
    mpl.rcParams["axes.edgecolor"] = theme["fg"]
    mpl.rcParams["axes.labelcolor"] = theme["fg"]
    mpl.rcParams["text.color"] = theme["fg"]
    mpl.rcParams["xtick.color"] = theme["fg"]
    mpl.rcParams["ytick.color"] = theme["fg"]
    mpl.rcParams["grid.color"] = theme["fg"]
    mpl.rcParams["grid.alpha"] = grid_alpha
    mpl.rcParams["axes.grid"] = True


def activate_theme(mode="dark"):
    if mode not in themes:
        valid = ", ".join(themes.keys())
        raise ValueError(f"Unknown theme '{mode}'. Choose one of: {valid}")

    theme = themes[mode]
    apply_theme_rcparams(theme)
    return theme


def style_axis(ax, theme, *, grid_alpha=None):
    if grid_alpha is None:
        grid_alpha = theme.get("grid_alpha", 0.20)

    ax.set_facecolor(theme["bg"])

    for spine in ax.spines.values():
        spine.set_color(theme["fg"])
        spine.set_alpha(grid_alpha)

    ax.tick_params(colors=theme["fg"])
    ax.xaxis.label.set_color(theme["fg"])
    ax.yaxis.label.set_color(theme["fg"])
    ax.title.set_color(theme["fg"])

    ax.grid(True, color=theme["fg"], alpha=grid_alpha)


def style_3d_axis(ax, theme, *, pane_alpha=1.0, grid_alpha=None):
    if grid_alpha is None:
        grid_alpha = theme.get("grid_alpha", 0.20)

    bg_rgba = mcolors.to_rgba(theme["bg"], pane_alpha)
    fg_rgba = mcolors.to_rgba(theme["fg"], 1.0)

    ax.xaxis.set_pane_color(bg_rgba)
    ax.yaxis.set_pane_color(bg_rgba)
    ax.zaxis.set_pane_color(bg_rgba)

    ax.xaxis._axinfo["grid"]["color"] = (*fg_rgba[:3], grid_alpha)
    ax.yaxis._axinfo["grid"]["color"] = (*fg_rgba[:3], grid_alpha)
    ax.zaxis._axinfo["grid"]["color"] = (*fg_rgba[:3], grid_alpha)
    ax.xaxis._axinfo["grid"]["linewidth"] = 0.6
    ax.yaxis._axinfo["grid"]["linewidth"] = 0.6
    ax.zaxis._axinfo["grid"]["linewidth"] = 0.6

    ax.xaxis.line.set_color(theme["fg"])
    ax.yaxis.line.set_color(theme["fg"])
    ax.zaxis.line.set_color(theme["fg"])
    ax.xaxis.line.set_alpha(grid_alpha)
    ax.yaxis.line.set_alpha(grid_alpha)
    ax.zaxis.line.set_alpha(grid_alpha)

    ax.tick_params(
        colors=theme["fg"],
        length=theme.get("tick_length", 2),
        width=theme.get("tick_width", 0.6),
    )


__all__ = [
    "CAT_LABELS",
    "DARK_BG",
    "LIGHT_BG",
    "BLUEPRINT_BG",
    "BLUEPRINT_FG",
    "BLUEPRINT_FG_ALT",
    "BLUEPRINT_BG_ALT",
    "activate_theme",
    "apply_theme_rcparams",
    "colors",
    "gbox_dark",
    "gbox_light",
    "gruvbox_d",
    "gruvbox_l",
    "blueprint_dark",
    "blueprint_light",
    "blueprint_d",
    "blueprint_l",
    "style_3d_axis",
    "style_axis",
    "themes",
]