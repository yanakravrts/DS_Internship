from transformers import BertTokenizerFast, BertForTokenClassification
import torch

model_directory = "Task1/model_save/"

tokenizer = BertTokenizerFast.from_pretrained(model_directory)
model = BertForTokenClassification.from_pretrained(model_directory)

id_to_tag = {0: 'O', 1: 'B-MOUNTAIN', 2: 'I-MOUNTAIN'}

def predict(sentence):
    """
    Predict token labels for the given sentence.
    
    Args:
        sentence (str): Input sentence to analyze.

    Returns:
        list: List of tuples containing tokens and their predicted labels.
    """
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze().tolist())
    predictions = outputs.logits.argmax(dim=-1).squeeze().tolist()
    predicted_labels = [id_to_tag[pred] for pred in predictions]

    token_label_pairs = list(zip(tokens, predicted_labels))
    return token_label_pairs

sentence = "Mount Everest is Earth's highest mountain above sea level."

get_prediction = predict(sentence)

output_file = 'Task1/model_result.txt'

with open(output_file, 'w') as f:
    f.write("Token\t\tLabel\n")
    f.write("-" * 30 + "\n") 
    for token, label in get_prediction:
        f.write(f'{token:<15}: {label}\n')







