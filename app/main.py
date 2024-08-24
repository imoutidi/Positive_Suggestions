from fastapi import HTTPException, Depends, status
from fastapi import FastAPI
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from pydantic import BaseModel, constr

import re
import os
import json
import string

import langid
from transformers import BertTokenizer, BertForMaskedLM, pipeline
import torch



app = FastAPI()

# Initializing models
# Load pre-trained model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# This model is the most used one in Hugging face for Masked Language Modeling
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
# Sentiment analysis model
sentiment_pipeline = pipeline("sentiment-analysis")


# Define a Pydantic model for the input data
class TextInput(BaseModel):
    # The sentence should not be more than 80 characters, to avoid huge entries (DOS).
    # And it should be a string.
    text: constr(max_length=80)

security = HTTPBasic()

def verify_credentials(credentials: HTTPBasicCredentials):

    # Read credentials from file
    credentials_file = open(os.path.join(os.path.dirname(__file__), "credentials.json"))
    credentials_dict = json.load(credentials_file)
    auth_username = credentials_dict["username"]
    auth_password = credentials_dict["password"]

    # auth_username = "user"
    # auth_password = "31415"
    # Check if the provided username exists
    if credentials.username != auth_username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )

    # Check if the provided password matches
    if credentials.password != auth_password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )

# Define a POST endpoint that accepts a string
@app.post("/sentence_sentiment/")
async def process_text(input_sentence: TextInput, credentials: HTTPBasicCredentials = Depends(security) ):
    verify_credentials(credentials)
    check_result = check_input(input_sentence)
    if not check_result[0]:
        return {"response": check_result[1]}
    else:
        # Return the processed string
        suggested_words = generate_text_suggestions(input_sentence)
        return {"output1": suggested_words}

# I haven't done excessive input checks.
# I assume that there should be only one <blank> tag and that there should be more than
# two words in an input sentence.
def check_input(sent_input):
    pattern_plain = r"(?=.*\b\w+\b)+(?=.*<blank>)"
    pattern_spaces = r"(?=.*\b\w+\b)(?=.*(?<!\w)<blank>(?!\w))"

    if re.search(pattern_plain, sent_input.text):
        if not re.search(pattern_spaces, sent_input.text):
            return False, "The <blank> tag shouldn't be adjacent with other characters!"
        if len(sent_input.text.split()) < 2:
            return False, "The input string should have more than one words (including the <blank> tag)."
        if sent_input.text.count("<blank>") > 1:
            return False, "The input string should have only one <blank> tag."
        # Detecting the language of the input text I remove the <blank> tag because it messes up the language detection.
        language = langid.classify(sent_input.text.replace("<blank>", "").strip())
        # print("The language is", language)
        # print(sent_input.text.replace("<blank>", "").strip())
        if language[0] != "en":
            return False, "The input language should be in English."
        else:
            return True, " is valid!"
    else:
        return False, "The input string should contain at least one word and the tag <blank> in it!"

def generate_text_suggestions(sent_input, results_num=3):

    # Preprocessing
    text = prepare_text_for_classifier(sent_input.text)

    # Tokenize input
    inputs = tokenizer(text, return_tensors='pt')

    # Get the index of the masked token
    mask_token_index = torch.where(inputs['input_ids'] == tokenizer.mask_token_id)[1]

    # Predict all tokens
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    mask_token_logits = logits[0, mask_token_index, :]

    # Get the top predictions for the masked token
    top_tokens = torch.topk(mask_token_logits, results_num, dim=1).indices[0].tolist()
    suggested_words = sentiment_evaluation_of_suggestions(top_tokens, tokenizer)
    return suggested_words

def prepare_text_for_classifier(input_text):
    p_text = input_text.replace("<blank>", "[MASK]")
    p_text = p_text.strip()
    # I exclude the ] character from the punctuation list because it is part of the [MASK] tag
    # of the classifier and it prevents the addition of a full stop later in this function.
    custom_punctuation = string.punctuation.replace("]", "")

    # If the input text does not have a punctuation mark in the end we add a fullstop.
    # That way the model will not return punctuation marks as results.
    if p_text[-1] not in custom_punctuation:
        p_text += "."
    return p_text

# Concatenate the suggested words into a string
def sentiment_evaluation_of_suggestions(t_tokens, tokenizer):
    non_negative_words = list()
    for token in t_tokens:
        word = tokenizer.decode([token])
        sentiment = sentiment_pipeline(word)
        # Discarding any negative suggestions.
        if sentiment[0]["label"] != "NEGATIVE":
            non_negative_words.append(word)
    # Concatenating all words into one string, coma separated.
    suggested_words = ""
    for p_word in non_negative_words[:-1]:
        suggested_words += p_word + ", "
    suggested_words += non_negative_words[-1]
    if len(non_negative_words) == 0:
        return "No non negative words were found."
    else:
        return suggested_words


if __name__ =="__main__":
    print("So it begins")