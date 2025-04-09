import streamlit as st
import json
import random
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

import nltk
nltk.download('punkt')  # И другие нужные ресурсы

# Загрузка необходимых данных NLTK
nltk.download('punkt_tab')
nltk.download('wordnet')

# Инициализация лемматизатора
lemmatizer = WordNetLemmatizer()

# Загрузка обученной модели и ранее сохранённых данных
model = load_model("chatbot_model.h5")
with open("intents.json", encoding="utf-8") as file:
    intents = json.load(file)
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))


def bag_of_words(sentence, words):
    """Преобразует предложение в bag-of-words вектор."""
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    bag = [1 if w in sentence_words else 0 for w in words]
    return np.array(bag)


def predict_class(sentence, model):
    """Определяет интент на основе предсказания модели."""
    bow = bag_of_words(sentence, words)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)

    return_list = [{"intent": classes[i], "probability": str(prob)} for i, prob in results]
    return return_list


def get_response(intents_list, intents_json):
    """Выбирает ответ из интентов на основе предсказанного интента."""
    if not intents_list:
        return "Извините, я не понимаю вас."
    tag = intents_list[0]['intent']
    for intent in intents_json['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "Извините, я не понимаю вас."



st.title("Чат бот")

# Инициализация истории сообщений
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "ai", "content": "Начните общение с мной!"}]

# Отображение истории сообщений
for msg in st.session_state.messages:
    if msg.get("is_image"):
        st.chat_message(msg["role"]).image(msg["content"])
    else:
        st.chat_message(msg["role"]).write(msg["content"])

# Обработка ввода пользователя
if user_prompt := st.chat_input():
    # Показываем сообщение пользователя
    st.chat_message("user").write(user_prompt)
    st.session_state.messages.append({"role": "user", "content": user_prompt})



    ints = predict_class(user_prompt, model)
    response = get_response(ints, intents)

    # Показываем ответ бота
    st.chat_message("ai").write(response)
    st.session_state.messages.append({"role": "ai", "content": response})




