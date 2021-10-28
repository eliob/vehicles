import uvicorn
import nest_asyncio
from enum import Enum
from fastapi import FastAPI
from fastapi.responses import JSONResponse

import modals_predictions
import warnings

warnings.filterwarnings("ignore")

app = FastAPI(
    title="Craig's list used cars prediction",
    description='Predict the price of a Car',
    version='0.1')


@app.get('/')
def _index():
    """ Health check."""
    response = {
        'message': 'OK',
        'status-code': 200,
        'data': {},
    }
    return JSONResponse(response)


class Manufacturer(str, Enum):
    gmc = "gmc"
    chevrolet = "chevrolet"
    toyota = "toyota"
    ford = "ford"
    jeep = "jeep"
    nissan = "nissan"
    ram = "ram"
    mazda = "mazda"
    cadillac = "cadillac"
    honda = "honda"
    dodge = "dodge"
    lexus = "lexus"
    jaguar = "jaguar"
    buick = "buick"
    chrysler = "chrysler"
    volvo = "volvo"
    audi = "audi"
    infiniti = "infiniti"
    lincoln = "lincoln"
    alfa_romeo = "alfa-romeo"
    subaru = "subaru"
    acura = "acura"
    hyundai = "hyundai"
    mercedes_benz = "mercedes-benz"
    bmw = "bmw"
    mitsubishi = "mitsubishi"
    volkswagen = "volkswagen"
    porsche = "porsche"
    kia = "kia"
    rover = "rover"
    ferrari = "ferrari"
    mini = "mini"
    pontiac = "pontiac"
    fiat = "fiat"
    tesla = "tesla"
    saturn = "saturn"
    mercury = "mercury"
    harley_davidson = "harley-davidson"
    datsun = "datsun"
    aston_martin = "aston-martin"
    landrover = "land rover"
    morgan = "morgan"


class Size(str, Enum):
    full_size = 'full-size'
    mid_size = 'mid-size'
    compact = 'compact'
    sub_compact = 'sub-compact'


class Type(str, Enum):
    pickup = "pickup"
    truck = "truck"
    other = "other"
    coupe = "coupe"
    SUV = "SUV"
    hatchback = "hatchback"
    mini_van = "mini-van"
    sedan = "sedan"
    offroad = "offroad"
    bus = "bus"
    van = "van"
    convertible = "convertible"
    wagon = "wagon"


class Drive(str, Enum):
    rwd = 'rwd'
    wd4 = '4wd'
    fwd = 'fwd'


class Transmission(str, Enum):
    other = 'other'
    automatic = 'automatic'
    manual = 'manual'


class Cylinders(str, Enum):
    cylinders8 = "8 cylinders"
    cylinders6 = "6 cylinders"
    cylinders4 = "4 cylinders"
    cylinders5 = "5 cylinders"
    cylinders3 = "3 cylinders"
    cylinders10 = "10 cylinders"
    cylinders12 = "12 cylinders"


class Fuel(str, Enum):
    gas = 'gas'
    other = 'other'
    diesel = 'diesel'
    hybrid = 'hybrid'
    electric = 'electric'


class Condition(str, Enum):
    good = "good"
    excellent = "excellent"
    fair = "fair"
    like_new = "like new"
    new = "new"
    salvage = "salvage"


class TitleStatus(str, Enum):
    clean = "clean"
    rebuilt = "rebuilt"
    lien = "lien"
    salvage = "salvage"
    missing = "missing"
    parts_only = "parts only"


@app.post('/predict')
def predict(manufacturer: Manufacturer, size: Size, type: Type, drive: Drive, year: int, odometer: int,
            transmission: Transmission,
            cylinders: Cylinders, fuel: Fuel, condition: Condition, title_status: TitleStatus):
    # result = float(model.predict(array.reshape(1, -1)))

    knn_prediction = modals_predictions.api_knn_prediction(manufacturer=manufacturer, size=size, v_type=type,
                                                           drive=drive, year=year, odometer=odometer,
                                                           transmission=transmission, cylinders=cylinders,
                                                           fuel=fuel, condition=condition, title_status=title_status)
    tree_prediction = modals_predictions.api_tree_prediction(manufacturer=manufacturer, size=size, v_type=type,
                                                             drive=drive, year=year, odometer=odometer,
                                                             transmission=transmission, cylinders=cylinders,
                                                             fuel=fuel, condition=condition, title_status=title_status)
    linear_prediction = modals_predictions.api_linear_prediction(manufacturer=manufacturer, size=size, v_type=type,
                                                                 drive=drive, year=year, odometer=odometer,
                                                                 transmission=transmission, cylinders=cylinders,
                                                                 fuel=fuel, condition=condition,
                                                                 title_status=title_status)

    result = {'Knn_prediction': knn_prediction, 'Tree prediction': tree_prediction,
              'Linear prediction': linear_prediction}
    return JSONResponse(content=result)


if __name__ == "__main__":
    # Allows the server to be run in this interactive environment
    nest_asyncio.apply()

    # Host depends on the setup you selected (docker or virtual env)
    host = "127.0.0.1"
    port = 8080

    # Spin up the server!
    uvicorn.run(app, host=host, port=port)
