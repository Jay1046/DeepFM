from Settings import settings

def post_preprocess(df, y_pred):
    df.loc[df[settings['target_column'][0]].isnull(), settings['target_column'][0]] = y_pred.reshape(-1)
    return df

