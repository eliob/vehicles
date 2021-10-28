import pandas as pd
import numpy as np

import utils
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression


def knn_prediction(final_df, manufacturer, cylinders, title_status, v_type,
                   size, drive, year, odometer, transmission, fuel, condition):
    X_train, ordinal_convertor = utils.convert_final_df_to_knn(final_df)

    odometer = np.log1p(odometer)
    year = 2021 - year
    size, drive, transmission, fuel, condition = ordinal_convertor.transform(pd.DataFrame({'size': size, 'drive': drive,
                                                                                           'transmission': transmission,
                                                                                           'fuel': fuel,
                                                                                           'condition': condition},
                                                                                          index=[0])).iloc[0]

    model = KNeighborsRegressor(n_neighbors=6)
    X_train_sub = X_train[(manufacturer == X_train['manufacturer']) &
                          (cylinders == X_train['cylinders']) &
                          (title_status == X_train['title_status']) &
                          (v_type == X_train['type'])]

    if len(X_train_sub) < 6:
        return -1

    X_train_sub.drop(columns=['manufacturer', 'cylinders', 'title_status', 'type'], inplace=True)
    model.fit(X_train_sub.drop(columns=['price']), X_train_sub.price)

    y_pred = model.predict([[size, drive, year, odometer, transmission, fuel, condition]])
    return round(np.expm1(y_pred[0]), 2)


def tree_prediction(final_df, manufacturer, cylinders, title_status, v_type,
                    size, drive, year, odometer, transmission, fuel, condition):
    X_train, ordinal_convertor, median_convertor = utils.convert_final_df_to_tree(final_df)

    odometer = np.log1p(odometer)
    year = 2021 - year
    size, drive, transmission, fuel, condition = ordinal_convertor.transform(pd.DataFrame({'size': size, 'drive': drive,
                                                                                           'transmission': transmission,
                                                                                           'fuel': fuel,
                                                                                           'condition': condition},
                                                                                          index=[0])).iloc[0]
    manufacturer, v_type, cylinders, title_status = median_convertor.transform(pd.DataFrame(
        {'manufacturer': manufacturer, 'type': v_type, 'cylinders': cylinders, 'title_status': title_status},
        index=[0])).iloc[0]

    model = DecisionTreeRegressor(max_leaf_nodes=60)
    model.fit(X_train.drop(columns=['price']), X_train.price)

    y_pred = model.predict([[manufacturer, size, v_type, drive, year, odometer, transmission,
                             cylinders, fuel, condition, title_status]])
    return round(np.expm1(y_pred[0]), 2)


def linear_prediction(final_df, manufacturer, cylinders, title_status, v_type,
                     size, drive, year, odometer, transmission, fuel, condition):
    X_train, ordinal_convertor, median_convertor = utils.convert_final_df_to_tree(final_df)

    odometer = np.log1p(odometer)
    year = 2021 - year
    size, drive, transmission, fuel, condition = ordinal_convertor.transform(pd.DataFrame({'size': size, 'drive': drive,
                                                                                           'transmission': transmission,
                                                                                           'fuel': fuel,
                                                                                           'condition': condition},
                                                                                          index=[0])).iloc[0]
    manufacturer, v_type, cylinders, title_status = median_convertor.transform(pd.DataFrame(
        {'manufacturer': manufacturer, 'type': v_type, 'cylinders': cylinders, 'title_status': title_status},
        index=[0])).iloc[0]

    model = LinearRegression()
    model.fit(X_train.drop(columns=['price']), X_train.price)

    y_pred = model.predict([[manufacturer, size, v_type, drive, year, odometer, transmission,
                             cylinders, fuel, condition, title_status]])
    return round(np.expm1(y_pred[0]), 2)
