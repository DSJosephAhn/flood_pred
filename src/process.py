import pandas as pd


def x_index_process(rf_df_2012):
    index= pd.Series(range(len(rf_df_2012)), name='index')
    return pd.concat([index, rf_df_2012], axis=1)
