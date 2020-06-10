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
