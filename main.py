import pandas as pd
import download_kaggle_dataset
import utils
import modals_predictions
import warnings

warnings.filterwarnings("ignore")
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 40)

if __name__ == '__main__':
    download_kaggle_dataset.download_dataset('austinreese/craigslist-carstrucks-data')
    cars = pd.read_csv('data/vehicles.csv')

    cars_clean = cars.drop(
        columns=['url', 'region_url', 'image_url', 'VIN', 'region', 'county', 'lat', 'long', 'posting_date', 'id',
                 'description']).copy()

    column_names = ['manufacturer', 'model', 'size', 'type', 'drive', 'year', 'odometer', 'transmission', 'cylinders',
                    'fuel', 'paint_color', 'condition', 'title_status', 'price', 'state']
    cars_clean = cars_clean.reindex(columns=column_names)
    cars_clean.drop_duplicates(ignore_index=True, inplace=True)

    for drop_column in ['price', 'odometer', 'year', 'manufacturer']:
        cars_clean.drop(index=cars_clean[cars_clean[drop_column].isnull()].index, inplace=True)
    cars_clean['year'] = cars_clean['year'].astype(int)

    cars_clean = cars_clean[(cars_clean['price'] > 1500) & (cars_clean['price'] < 40000)]

    # Remove 'Junk' prices
    cars_clean = cars_clean[cars_clean['price'].map(
        lambda par: par not in (1234, 12340, 12344, 12345, 12349, 123456, 1234567, 12345678, 123456789, 1234567890,
                                1111, 11111, 11117, 111111, 1111111, 11111111, 1111111111, 2222, 22222, 3333, 33333))]

    cars_clean = cars_clean[cars_clean['odometer'] > 2500]

    # fill NaN with "other"
    values = {"fuel": 'other', "cylinders": 'other', 'drive': 'other', 'size': 'other', 'type': 'other'}
    cars_clean.fillna(value=values, inplace=True)

    # Add 'manufacturer_model' column
    cars_clean['manufacturer_model'] = cars_clean['manufacturer'] + '@@@' + cars_clean['model']

    cars_clean = utils.fill_nan_values(cars_clean)
    cars_clean = utils.remove_other_and_nan(cars_clean)

    final_df = cars_clean.drop(columns=['model', 'paint_color', 'state', 'manufacturer_model']).copy()

    print(final_df)

    print('The Knn prediction is: ',
          modals_predictions.knn_prediction(final_df, manufacturer='jeep', size='mid-size', v_type='SUV', drive='4wd',
                                            year=1992, odometer=192000, transmission='automatic',
                                            cylinders='6 cylinders',
                                            fuel='gas', condition='excellent', title_status='clean'))

    print('The tree prediction is: ',
          modals_predictions.tree_prediction(final_df, manufacturer='jeep', size='mid-size', v_type='SUV', drive='4wd',
                                             year=1992, odometer=192000, transmission='automatic',
                                             cylinders='6 cylinders',
                                             fuel='gas', condition='excellent', title_status='clean'))

    print('The Linear prediction is: ',
          modals_predictions.linear_prediction(final_df, manufacturer='jeep', size='mid-size', v_type='SUV',
                                               drive='4wd',
                                               year=1992, odometer=192000, transmission='automatic',
                                               cylinders='6 cylinders',
                                               fuel='gas', condition='excellent', title_status='clean'))
