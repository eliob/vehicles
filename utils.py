import pandas as pd
import transformers
import numpy as np


def fill_nan_values(cars_clean):
    cars_mm_grouped = cars_clean.groupby(['manufacturer_model']).apply(
        lambda grp: pd.Series([grp['manufacturer'].max(), grp['model'].max(),
                               grp['cylinders'].value_counts(normalize=True, ascending=False, dropna=False),
                               len(set(grp['cylinders'])),
                               grp['fuel'].value_counts(normalize=True, ascending=False, dropna=False),
                               len(set(grp['fuel'])),
                               grp['drive'].value_counts(normalize=True, ascending=False, dropna=False),
                               len(set(grp['drive'])),
                               grp['size'].value_counts(normalize=True, ascending=False, dropna=False),
                               len(set(grp['size'])),
                               grp['type'].value_counts(normalize=True, ascending=False, dropna=False),
                               len(set(grp['type'])),
                               ],
                              index=['manufacturer', 'model', 'cylinders Freq', 'cylinders Count', 'Fuel Freq',
                                     'Fuel Count',
                                     'drive Freq', 'drive Count', 'size Freq', 'size Count', 'type Freq',
                                     'type Count']))

    cars_mm_grouped['cylinders Max Val'] = cars_mm_grouped['cylinders Freq'].map(
        lambda par: par.index[0] if (par.index[0] != 'other' or len(par.index) == 1) else par.index[1])
    cars_mm_grouped['cylinders Max Freq'] = cars_mm_grouped['cylinders Freq'].map(lambda par: round(par.values[0], 2))
    cars_mm_grouped['Fuel Max Val'] = cars_mm_grouped['Fuel Freq'].map(
        lambda par: par.index[0] if (par.index[0] != 'other' or len(par.index) == 1) else par.index[1])
    cars_mm_grouped['Fuel Max Freq'] = cars_mm_grouped['Fuel Freq'].map(lambda par: round(par.values[0], 2))
    cars_mm_grouped['drive Max Val'] = cars_mm_grouped['drive Freq'].map(
        lambda par: par.index[0] if (par.index[0] != 'other' or len(par.index) == 1) else par.index[1])
    cars_mm_grouped['drive Max Freq'] = cars_mm_grouped['drive Freq'].map(lambda par: round(par.values[0], 2))
    cars_mm_grouped['size Max Val'] = cars_mm_grouped['size Freq'].map(
        lambda par: par.index[0] if (par.index[0] != 'other' or len(par.index) == 1) else par.index[1])
    cars_mm_grouped['size Max Freq'] = cars_mm_grouped['size Freq'].map(lambda par: round(par.values[0], 2))
    cars_mm_grouped['type Max Val'] = cars_mm_grouped['type Freq'].map(
        lambda par: par.index[0] if (par.index[0] != 'other' or len(par.index) == 1) else par.index[1])
    cars_mm_grouped['type Max Freq'] = cars_mm_grouped['type Freq'].map(lambda par: round(par.values[0], 2))

    cars_mm_grouped.drop(
        labels=['cylinders Freq', 'cylinders Count', 'Fuel Freq', 'Fuel Count', 'drive Freq', 'drive Count',
                'size Freq', 'size Count', 'type Freq', 'type Count'], axis=1, inplace=True)

    cars_update = cars_mm_grouped.copy()

    cars_update.reset_index(inplace=True)
    cars_update.drop(
        columns=['manufacturer', 'model', 'cylinders Max Freq', 'Fuel Max Freq', 'drive Max Freq', 'size Max Freq',
                 'type Max Freq'], inplace=True)
    cars_update.set_index(['manufacturer_model'], inplace=True)
    cars_update.rename(columns={'cylinders Max Val': 'cylinders', 'Fuel Max Val': 'fuel', 'drive Max Val': 'drive',
                                'size Max Val': 'size', 'type Max Val': 'type'}, inplace=True)

    update_col_name = cars_update.columns
    for col in update_col_name:
        cars_clean.loc[cars_clean[col] == 'other', col] = cars_clean[cars_clean[col] == 'other']['manufacturer_model'] \
            .map(cars_update[col])

    return cars_clean


def remove_other_and_nan(cars_clean):
    cars_clean.dropna(
        subset=['cylinders', 'condition', 'size', 'type', 'transmission', 'drive', 'title_status', 'fuel'],
        inplace=True)

    for col_name in ['cylinders', 'size', 'type', 'transmission', 'drive', 'fuel']:
        cars_clean = cars_clean[cars_clean[col_name] != 'other']

    return cars_clean


def convert_final_df_to_knn(final_df):
    final_df_knn = final_df.copy()
    final_df_knn['price'] = np.log1p(final_df['price'])
    final_df_knn['odometer'] = np.log1p(final_df['odometer'])
    final_df_knn['year'] = 2021 - final_df['year']
    ordinal_columns = ['condition', 'fuel', 'transmission', 'drive', 'size']
    ordinal_convertor = transformers.ConvertOrdinal(ordinal_columns)
    ordinal_convertor.fit(final_df_knn).transform(final_df_knn)
    return final_df_knn


def convert_final_df_to_tree(final_df):
    final_df_tree = convert_final_df_to_knn(final_df)
    median_columns = ['cylinders', 'manufacturer', 'type', 'title_status']
    median_convertor = transformers.ConvertMedian(median_columns)
    median_convertor.fit(final_df_tree).transform(final_df_tree)
    return final_df_tree
