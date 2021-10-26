import pandas as pd


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
        cars_clean.loc[cars_clean[col] == 'other', col] = cars_clean[cars_clean[col] == 'other']['manufacturer_model']\
        .map(cars_update[col])

    return cars_clean


