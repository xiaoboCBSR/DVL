import altair as alt
import pandas as pd
from os.path import join
from altair_saver import save


def per_type_plot(df_A, df_B, name_A, name_B):
    # produce plot that compare the per-type f1 of two apporaches.
    def melt_df(df):
        df = pd.melt(df,
                     id_vars=['type'],
                     value_vars=[name_A, name_B],
                     var_name='Model', value_name='F1')
        return df

    color_scale = alt.Scale(domain=[name_A, name_B], scheme='category10')
    bar_size = 5 #3
    col_width, col_height = 10, 170 #6, 100
    model_order = ['SATO', 'Sherlock', 'DVL']

    df_A = df_A.rename({'f1-score': name_A}, axis='columns')
    df_B = df_B.rename({'f1-score': name_B}, axis='columns')
    df = pd.merge(df_A, df_B, on=['type'], suffixes=("_" + name_A, "_" + name_B))

    better = df[df[name_A] > df[name_B]]  # A better
    worse = df[df[name_A] < df[name_B]]  # A worse
    equal = df[df[name_A] == df[name_B]]  # equal

    better = melt_df(better)
    worse = melt_df(worse)
    equal = melt_df(equal)

    chart1 = alt.Chart(better).mark_bar(size=bar_size).encode(
        y=alt.Y("F1:Q", title='Support F1 Score'),
        x=alt.X('Model:O', sort=model_order, axis=None),
        color=alt.Color('Model:N', sort=model_order, scale=color_scale)
    ).properties(
        width=col_width,
        height=col_height
    ).facet(column=alt.Column("type:O",
                              sort=alt.EncodingSortField('F1',
                                                         op='min',
                                                         order='descending'),
                              title=None,
                              header=alt.Header(labelAlign='right', labelAngle=275)
                              )
            )

    chart2 = alt.Chart(worse).mark_bar(size=bar_size).encode(
        y=alt.Y("F1:Q", title=None, axis=None),
        x=alt.X('Model:O', sort=model_order, title=None, axis=None),
        color=alt.Color('Model:N', sort=model_order, scale=color_scale)
    ).properties(
        width=col_width,
        height=col_height
    ).facet(column=alt.Column("type:O",
                              sort=alt.EncodingSortField('F1',
                                                         op='max',
                                                         order='descending'),
                              title=None,
                              header=alt.Header(labelAlign='right', labelAngle=275)
                              ),
            )

    chart3 = alt.Chart(equal).mark_bar(size=bar_size).encode(
        y=alt.Y("F1:Q", title=None, axis=None),
        x=alt.X('Model:O', sort=model_order, title=None, axis=None),
        color=alt.Color('Model:N', sort=model_order, scale=color_scale)
    ).properties(
        width=col_width,
        height=col_height
    ).facet(column=alt.Column("type:O",
                              sort=alt.EncodingSortField('F1',
                                                         op='max',
                                                         order='descending'),
                              title=None,
                              header=alt.Header(labelAlign='right', labelAngle=275)
                              ),
            )

    return alt.hconcat(chart1, chart3, chart2, spacing=20).configure_facet(spacing=1.6)


if __name__ == "__main__":
    # import data
    path = './plot_data'
    result_sherlock_pair = pd.read_csv(join(path, 'pairflip_result_sherlock+LDA_None.csv'))
    result_dvl_pair = pd.read_csv(join(path, 'pairflip_result_sherlock+LDA_RV-AM.csv'))
    result_sherlock_sym = pd.read_csv(join(path, 'sym_result_sherlock+LDA_None.csv'))
    result_dvl_sym = pd.read_csv(join(path, 'sym_result_sherlock+LDA_RV-AM.csv'))

    naming = {
        'SATO': 'SATO',
        'Sherlock': 'Sherlock',  # sato with only LDA(sherlock + LDA)
        'DVL': 'DVL'
    }

    f1 = per_type_plot(result_dvl_pair, result_sherlock_pair, naming['DVL'], naming['Sherlock'])  # LDA
    f2 = per_type_plot(result_dvl_sym, result_sherlock_sym, naming['DVL'], naming['SATO'])  # LDA
    save(f1, 'f1.png', scale_factor=12.0)
    save(f2, 'f2.png', scale_factor=12.0)