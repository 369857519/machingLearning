def normalize_feature(df):
    return df.apply(lambda column: (column-column.mean())/column.std())