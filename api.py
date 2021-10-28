import uvicorn
import nest_asyncio
from fastapi import FastAPI
from fastapi.responses import JSONResponse

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


@app.post('/amir')
def amir(par1):
    # result = float(model.predict(array.reshape(1, -1)))

    result = {'res1': 123, 'res2': 456, 'res3': 789}
    return JSONResponse(content=result)


if __name__ == "__main__":
    # Allows the server to be run in this interactive environment
    nest_asyncio.apply()

    # Host depends on the setup you selected (docker or virtual env)
    host = "127.0.0.1"
    port = 8080

    # Spin up the server!
    uvicorn.run(app, host=host, port=port)
