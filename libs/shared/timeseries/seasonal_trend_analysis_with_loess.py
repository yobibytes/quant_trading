from statsmodels.tsa.seasonal import STL
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()

def add_stl_plot(fig, res, legend):
    """Add plots from additional STL fits"""
    axs = fig.get_axes()
    comps = ['observed', 'trend', 'seasonal', 'resid']
    for ax, comp in zip(axs[0:], comps):
        for r in res:
            series = getattr(r, comp)
            if comp == 'resid':
                ax.plot(series, marker='o', linestyle='none')
            else:
                ax.plot(series)
                if comp == 'observed':
                    ax.legend(legend, frameon=False)
                    

df1_units_sum = df1.groupby('period')[f'units'].sum()
df2_units_sum = df2.groupby('period')[f'units'].sum()

plt.figure(figsize=(12,12))
res1_units = STL(df1_units_sum, robust=True).fit()
fig = res1_units.plot()
res2_units = STL(df2_units_sum, robust=True).fit()
add_stl_plot(fig, [res2_units], ['product 1', 'product 2'])
fig.set_size_inches(12,12)