import utils
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression


def knn_prediction(final_df, manufacturer, cylinders, title_status, v_type,
                   size, drive,	year, odometer,	transmission, fuel,	condition):
    X_train = utils.convert_final_df_to_knn(final_df)
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
    return y_pred[0]


def tree_prediction(final_df, manufacturer, cylinders, title_status, v_type,
                   size, drive,	year, odometer,	transmission, fuel,	condition):
    X_train = utils.convert_final_df_to_tree(final_df)
    model = DecisionTreeRegressor(max_leaf_nodes=60)
    model.fit(X_train, y_train)

    y_pred = model.predict([[size, drive, year, odometer, transmission, fuel, condition]])