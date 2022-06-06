import random
import json

import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"
print("Let's chat! (type 'quit' to exit)")
while True:
    # sentence = "do you use credit cards?"
    sentence = input("You: ")
    if sentence == "quit": # exit
        break
        
    sentence = tokenize(sentence) # tokenize
    X = bag_of_words(sentence, all_words) # bag of words
    X = X.reshape(1, X.shape[0]) # shape for PyTorch
    X = torch.from_numpy(X).to(device) # convert to PyTorch

    output = model(X) # get output
    _, predicted = torch.max(output, dim=1) # get predicted class

    tag = tags[predicted.item()] # get predicted tag

    probs = torch.softmax(output, dim=1) # get class probabilities
    prob = probs[0][predicted.item()]   # get probability of the predicted tag
    if prob.item() > 0.75:           # if greater than 0.75, we will use the prediction
        for intent in intents['intents']: # get the intent
            if tag == intent["tag"]: # get the intent tag
                print(f"{bot_name}: {random.choice(intent['responses'])}") # get a random response
    else:                        # otherwise, we will use the user's input
        print(f"{bot_name}: I do not understand...")  # print a response