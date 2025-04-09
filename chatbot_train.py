import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Если ранее не загружали данные nltk, выполняем загрузку:
nltk.download('punkt_tab')
nltk.download('wordnet')

# Инициализация лемматизатора
lemmatizer = WordNetLemmatizer()

# Загрузка данных интентов
with open("intents.json", encoding="utf-8") as file:
    data = json.load(file)

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

# Обработка каждого интента
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        # Токенизируем каждое предложение (паттерн)
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent["tag"]))
    if intent["tag"] not in classes:
        classes.append(intent["tag"])

# Приведение слов к нижнему регистру и лемматизация
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(set(words))
classes = sorted(set(classes))

# Сохраняем обработанные данные для использования при общении
pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))

# Создаем обучающие данные: bag-of-words для каждой фразы
training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    pattern_words = document[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

# Перемешиваем данные и разбиваем их на признаки и метки
random.shuffle(training)
training = np.array(training, dtype=object)
train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

# Создаем модель нейронной сети
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Компилируем и обучаем модель
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

# Сохраняем модель в файл
model.save("chatbot_model.h5")
print("Модель обучена и сохранена!")