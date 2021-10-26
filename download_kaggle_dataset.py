import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
# importing required modules
from zipfile import ZipFile
import os.path


def download_dataset(path='austinreese/craigslist-carstrucks-data'):
    api = KaggleApi()
    api.authenticate()
    file_name = path.split('/')[1] + '.zip'
    if not os.path.isfile(file_name):
        lis1 = api.datasets_list(search=path)
        api.dataset_download_files(path)

        # opening the zip file in READ mode
        with ZipFile(file_name, 'r') as zip:
            # printing all the contents of the zip file
            zip.printdir()

            # extracting all the files
            print('Extracting all the files now...')
            zip.extractall(path='data/')
