import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler
import os
import warnings
import numpy as np

_script_dir = os.path.dirname(os.path.realpath(__file__))

config = {
    'POSTGRES_USERNAME': 'dev',
    'POSTGRES_PASSWORD': '2q7Pt3e6KBVL77',
    'POSTGRES_DATABASE': 'development',
    'POSTGRES_HOST': 'postgres.dev.vave.corp',
    'POSTGRES_PORT': 5432
}

class PlotBase:
    def __init__(self, seaborn=True):
        self._font_prop = None
        self._use_seaborn = seaborn
        self._colours = {}
        self._cycle = []
        self._rc = {}
        self._colour_alias = {
            'w': 'white',
            'k': 'black',
            'b': 'blue',
            'g': 'green',
            'y': 'gold',
            'o': 'orange',
            'do': 'dark_orange',
            'lg': 'light_grey',
            'dg': 'dark_grey'
        }
        self.cmap = 'YlGnBu_r'
        self.colours = {
            'white': '#FFFFFF',
            'light_grey': '#F1F1F1',
            'dark_grey': '#C5C5C5',
            'black': '#000000',
            'blue': '#003366',
            'green': '#0BC73C',
            'orange': '#FFA500',
            'dark_orange': '#FF8C00',
            'gold': '#FFD700'
        }
        self.cycle = self.cycle_default
        self.rc = self.rc_default

    @property
    def colours(self):
        return self._colours

    @colours.setter
    def colours(self, dict_val):
        self._colours = dict_val
        self.set_alias()
        self.cycle = self.cycle_default

    @property
    def alias(self):
        return self._colour_alias

    @property
    def cycle_default(self):
        return [
            self.colours['blue'],
            self.colours['green'],
            self.colours['dark_grey'],
            self.colours['orange'],
            self.colours['black'],
            self.colours['gold'],
            self.colours['dark_orange']
        ]

    @property
    def cycle(self):
        return self._cycle

    @cycle.setter
    def cycle(self, list_val):
        self._cycle = list_val
        self.rc = self.rc_default

    @property
    def rc_default(self):
        return {
            'figure.figsize': np.array([16.0, 9.0]) * 0.5,
            'figure.facecolor': self.colours['white'],
            'figure.edgecolor': self.colours['white'],

            'font.family': self.font_prop.get_name(),
            'font.weight': 'light',
            'font.size': 14,

            'text.color': '#555555',

            'lines.linewidth': 1.0,

            'patch.linewidth': 0.5,
            'patch.facecolor': '#348ABD',
            'patch.edgecolor': '#EEEEEE',
            'patch.antialiased': True,

            'axes.facecolor': '#E5E5E5',
            'axes.edgecolor': self.colours['white'],
            'axes.linewidth': 1,
            'axes.grid': True,
            'axes.titlesize': 'medium',
            'axes.titleweight': 'light',
            'axes.labelsize': 'medium',
            'axes.labelweight': 'light',
            'axes.labelcolor': '#555555',
            'axes.axisbelow': True,
            'axes.prop_cycle': cycler(color=self.cycle),
            'axes.spines.left': True,
            'axes.spines.bottom': True,
            'axes.spines.right': True,
            'axes.spines.top': True,

            'xtick.color': '#555555',
            'xtick.direction': 'out',
            'xtick.labelsize': 14,

            'ytick.color': '#555555',
            'ytick.direction': 'out',
            'ytick.labelsize': 14,

            'grid.color': self.colours['white'],
            'grid.linestyle': '-',

            'legend.loc': 'best',
            'legend.borderaxespad': 0.5,
            'legend.fancybox': True,
            'legend.facecolor': '#348ABD',
            'legend.edgecolor': '#EEEEEE',
            'legend.fontsize': 'medium',

            'image.cmap': self.cmap
        }

    @property
    def rc(self):
        return self._rc

    @rc.setter
    def rc(self, dict_val):
        self._rc = dict_val
        self.set_styles()

    @property
    def cat_palette(self):
        return sns.color_palette(self.cycle)

    @property
    def seq_palette(self):
        return sns.color_palette(mpl.rcParams['image.cmap'])

    @property
    def font_prop(self):
        if not self._font_prop:
            self.setup_fonts()
        return self._font_prop

    def setup_fonts(self, path=None):
        if not path:
            path_to_font = os.path.join(_script_dir, "Calibri-Light.ttf")
            font_dir = _script_dir
        else:
            path_to_font = path
            font_dir = os.path.dirname(os.path.realpath(path))
        font_files = mpl.font_manager.findSystemFonts(
            fontpaths=[font_dir, ]
        )
        font_list = mpl.font_manager.createFontList(font_files)
        mpl.font_manager.fontManager.ttflist.extend(font_list)
        self._font_prop = mpl.font_manager.FontProperties(fname=path_to_font)

    def set_styles(self):
        mpl.rcParams.update(self.rc)

    def set_alias(self):
        for key, value in self._colour_alias.items():
            self.colours[key] = self.colours[value]
        mpl.colors.get_named_colors_mapping().update(self.colours)

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.cycle[index]
        else:
            return self.colours[index]

    def __setitem__(self, index, value):
        if isinstance(index, int):
            self.cycle[index] = value
            self.rc = self.rc_default
        else:
            if index in self._colour_alias:
                index = self._colour_alias[index]
            self.colours[index] = value
            self.set_alias()
            self.cycle = self.cycle_default


class Notebook(PlotBase):
    @property
    def rc_default(self):
        rc = super().rc_default

        rc['figure.figsize'] = np.array([16.0, 9.0])

        rc['font.weight'] = 'light'
        rc['font.size'] = 14

        rc['text.color'] = self.colours['blue']

        rc['patch.facecolor'] = self.colours['white']
        rc['patch.edgecolor'] = self.colours['white']
        rc['patch.linewidth'] = 0.5

        rc['lines.linewidth'] = 1.0

        rc['axes.facecolor'] = self.colours['white']
        rc['axes.edgecolor'] = self.colours['blue']
        rc['axes.linewidth'] = 1
        rc['axes.grid'] = True
        rc['axes.titlesize'] = 14
        rc['axes.titleweight'] = 'light'
        rc['axes.labelsize'] = 14
        rc['axes.labelweight'] = 'light'
        rc['axes.labelcolor'] = self.colours['blue']
        rc['axes.spines.left'] = True
        rc['axes.spines.bottom'] = True
        rc['axes.spines.right'] = False
        rc['axes.spines.top'] = False

        rc['xtick.color'] = self.colours['blue']
        rc['xtick.labelsize'] = 14

        rc['ytick.color'] = self.colours['blue']
        rc['ytick.labelsize'] = 14

        rc['grid.color'] = self.colours['dark_grey']
        rc['grid.linestyle'] = '-'

        rc['legend.loc'] = 'best'
        rc['legend.fontsize'] = 14
        rc['legend.facecolor'] = self.colours['white']
        rc['legend.edgecolor'] = self.colours['white']

        return rc

