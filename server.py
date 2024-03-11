from flask import Flask, jsonify, request
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

app = Flask(__name__)
model = RandomForestClassifier()
model.load('path/to/model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_df = pd.DataFrame([data])
    prediction = model.predict(input_df)
    return jsonify({'prediction': prediction.tolist()})

from flask import Flask, request, jsonify
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer

app = Flask(__name__)
chatbot = ChatBot('my_bot')
trainer = ChatterBotCorpusTrainer(chatbot)
trainer.train('chatterbot.corpus.english')

@app.route('/api/chatbot', methods=['POST'])
def chatbot():
    data = request.json
    input_text = data['text']
    response = chatbot.get_response(input_text)
    return jsonify({'response': response.text})