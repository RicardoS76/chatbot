import random
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer
from tkinter import *
from tkinter import scrolledtext
from tkinter import messagebox
from PIL import Image, ImageTk


lemmatizer = WordNetLemmatizer()
with open('intents.json') as file:
    intents = json.load(file)

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = tf.keras.models.load_model('chatbot_model.h5')


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    if intents_list:
        tag = intents_list[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                result = random.choice(i['responses'])
                break
        return result
    return "No entiendo, por favor intenta de nuevo."

def send_message(event=None):
    message = user_input.get()
    if message:
        chat_window.config(state=NORMAL)
        chat_window.insert(END, "Tú: " + message + '\n', 'user')
        chat_window.config(state=DISABLED)
        chat_window.yview(END)

        ints = predict_class(message)
        res = get_response(ints, intents)

        chat_window.config(state=NORMAL)
        chat_window.insert(END, "Bot: " + res + '\n\n', 'bot')
        chat_window.config(state=DISABLED)
        chat_window.yview(END)

        user_input.delete(0, END)
    else:
        messagebox.showwarning("Entrada vacía", "Por favor, escribe un mensaje para enviar.")


root = Tk()
root.title("Chatbot del Instituto Tecnológico de Pochutla")
root.geometry("500x550")
root.resizable(False, False)


logo_image = Image.open("assets/Logo.jpeg")
logo_image = logo_image.resize((100, 100), Image.LANCZOS)
logo_photo = ImageTk.PhotoImage(logo_image)
logo_label = Label(root, image=logo_photo)
logo_label.pack(pady=10)


title = Label(root, text="Chatbot del Instituto Tecnológico de Pochutla", font=("Helvetica", 16, "bold"))
title.pack(pady=10)


input_frame = Frame(root)
input_frame.pack(padx=10, pady=10, fill=X)


user_input = Entry(input_frame, font=("Helvetica", 14))
user_input.pack(side=LEFT, padx=5, fill=X, expand=True)
user_input.bind("<Return>", send_message)  


send_button = Button(input_frame, text="Enviar", command=send_message, font=("Helvetica", 14))
send_button.pack(side=RIGHT, padx=5)


chat_window = scrolledtext.ScrolledText(root, wrap=WORD, state=DISABLED, font=("Helvetica", 12))
chat_window.pack(padx=10, pady=10, fill=BOTH, expand=True)
chat_window.tag_config('user', foreground='blue')
chat_window.tag_config('bot', foreground='green')


root.mainloop()