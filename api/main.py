from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
app = Flask(__name__)
scaler = MinMaxScaler()

model = tf.keras.models.load_model("H:/Python/MachineLearning/1/PROJETOS/PREVISAO_PRECOS_CARRO/modelo.h5")
dic = {
    'convertible': [1,0,0,0,0],
    'hardtop': [0,1,0,0,0],
    'hatchback':[0,0,1,0,0],
    'sedan':[0,0,0,1,0],
    'wagon':[0,0,0,0,1]
}
@app.route("/helloworld", methods=['GET'])
def helloworld():
    return 'Hello world!'


@app.route("/predict", methods=['POST'])
def predict():
    data = request.get_json()

    horsepower = data["horsepower"]
    enginesize = data["enginesize"]
    valores = dic[data["type"]] + [horsepower, enginesize]
    
    valores_reshaped = np.array(valores).reshape(-1, 1) #Converteu a matriz para 2D
    
    valores_scaled = scaler.fit_transform(valores_reshaped)
    valores_scaled = valores_scaled.reshape(1, -1)
    predict = model.predict(valores_scaled)
    predict = scaler.inverse_transform(predict)
    predict_reshaped = predict.reshape(1, -1)
    return jsonify({"data": (predict_reshaped[0]*100).tolist()}), 200


if __name__ == '__main__':
    app.run('localhost', debug=True)


#Observação: O Fit transform só funciona com matrizes 2D