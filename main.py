import flask
from flask import Flask
from flask import request, render_template
import config
from model import BERTBaseUncased

import torch
import flask
import time
from flask import Flask
from flask import request

import functools
import torch.nn as nn
import joblib

app = Flask(__name__)     # initialize flask app

MODEL = None
DEVICE = "cpu"
PREDICTION_DICT = dict()
# memory = joblib.Memory("../input/", verbose=0)

@app.route("/")
def home():
  return render_template("index.html")

def predict_from_cache(sentence):
    if sentence in PREDICTION_DICT:
        return PREDICTION_DICT[sentence]
    else:
        result = sentence_prediction(sentence)
        PREDICTION_DICT[sentence] = result
        return result


@memory.cache
def sentence_prediction(sentence):
    tokenizer = config.TOKENIZER
    max_len = config.MAX_LEN
    review = str(sentence)
    review = " ".join(review.split())     # removes all unnecessary space

    inputs = tokenizer.encode_plus(       # encode_plus can encode 2 strings at a time
        review,
        None,                             # since we use only 1 string at a time
        add_special_tokens=True,          # adds cls, sep tokens
        max_length=max_len,
        truncation=True
    )

    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]
    token_type_ids = inputs["token_type_ids"]      # since only 1 string token_type_ids are same and unnecessary

    padding_length = max_len - len(ids)            # for bert we pad on the right side
    ids = ids + ([0] * padding_length)             # zero times the padding length
    mask = mask + ([0] * padding_length)
    token_type_ids = token_type_ids + ([0] * padding_length)

    ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)    # since dataloader always returns batches
    mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).unsqueeze(0)

    ids = ids.to(DEVICE, dtype=torch.long)          # send to tpu DEVICE
    token_type_ids = token_type_ids.to(DEVICE, dtype=torch.long)
    mask = mask.to(DEVICE, dtype=torch.long)

    outputs = MODEL(ids=ids, mask=mask, token_type_ids=token_type_ids)

    outputs = torch.sigmoid(outputs).cpu().detach().numpy()
    return outputs[0][0]          # since outputs will be 2-Dimensional but there is only 1 value


@app.route("/predict", methods=['POST'])            # creating end point using flask
def predict():                                      # predict function
    sentence = [str(x) for x in request.form.values()]
    sentence = sentence[0]
    # sentence = str(request.form.values())
    start_time = time.time()
    positive_prediction = sentence_prediction(sentence)
    negative_prediction = 1 - positive_prediction
    response = {}
    response["response"] = {
        "positive": str(positive_prediction),
        "negative": str(negative_prediction),
        "sentence": str(sentence),
        "time_taken": str(time.time() - start_time),
    }
    return render_template("index.html", sentence = r"Your Sentence = '{}'".format(sentence),
    prediction_text="Sentiment of the sentence is :", positive=  "{}% POSITIVE".format(round(positive_prediction*100,2)),
    negative = "{}% NEGATIVE".format(round(negative_prediction*100,2)))




if __name__ == "__main__":
    MODEL = BERTBaseUncased()
    MODEL.load_state_dict(torch.load(config.MODEL_PATH))
    MODEL.to(DEVICE)
    MODEL.eval()
    app.run(debug=True)
