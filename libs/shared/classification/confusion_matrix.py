def plot_confusion_matrix(cm, classes, title, normalize=False):
    cmap = plt.cm.Blues
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    

def get_feature_correlation(df, filename=None):
    df_corr = df.corr().stack().reset_index()
    df_corr.columns = ['feature_x', 'feature_y', 'corr']
    mask_dups = (df_corr[['feature_x', 'feature_y']].apply(frozenset, axis=1).duplicated()) | (
            df_corr['feature_x'] == df_corr['feature_y'])
    df_corr = df_corr[~mask_dups]
    df_corr = df_corr[df_corr["corr"] < 1.0].sort_values(['corr'], ascending=False).reset_index(drop=True)
    if filename:
        df_corr.to_csv(filename, sep=";", index=None, decimal=",")
    return df_corr