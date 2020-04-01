from __future__ import absolute_import, division, print_function, unicode_literals
from flask import Flask, jsonify, request, abort
import json
import numpy as np
import tensorflow as tf

from utils.data_helper import data_loader
from utils.params import get_params
import model.FastText.FastText_params as FastText_params
import model.TextCNN.TextCNN_params as TextCNN_params


app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

err_result = {
    "errCode": "",
    "errMsg": "",
    "status": False
}


FastText_model = tf.keras.models.load_model('./results/FastText/2020-03-27-15-01')
# FastText_model = tf.keras.models.load_model('./results/FastText/model.h5')
TextCNN_model = tf.keras.models.load_model('./results/TextCNN/2020-03-27-21-41')


_, _, _, _, vocab, mlb = data_loader(get_params())
labels = np.array(mlb.classes_)

fastText_params = FastText_params.get_params()
textCNN_params = TextCNN_params.get_params()


@app.route("/FastText_service/", methods=['GET', 'POST'])
def FastText_service():
    try:
        text_list = request.json
        predict_data = convert(text_list, fastText_params)
    except Exception as e:
        return jsonify(err_result)
    else:
        preds = FastText_model.predict(predict_data)
        results = []
        for pred in preds:
            results.append(labels[np.where(pred > 0.5)].tolist())
        return jsonify(results)


@app.route("/TextCNN_service/", methods=['GET', 'POST'])
def TextCNN_service():
    try:
        text_list = request.json
        predict_data = convert(text_list, textCNN_params)
    except Exception as e:
        return jsonify(err_result)
    else:
        preds = TextCNN_model.predict(predict_data)
        results = []
        for pred in preds:
            results.append(labels[np.where(pred > 0.5)].tolist())
        return jsonify(results)


def convert(text_list, params):
    maxlen = params['maxlen']
    result = []
    for text in text_list:
        tmp = [vocab[w] if w in vocab else 1 for w in text['text']]
        if len(tmp) > maxlen:
            result.append(tmp[:maxlen])
        else:
            result.append(tmp + [0] * (maxlen - len(tmp)))
    return result


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
