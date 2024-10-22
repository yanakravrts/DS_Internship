from transformers import BertTokenizerFast, BertForTokenClassification, Trainer, TrainingArguments
import torch
import pandas as pd
import ast

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=3)

tag_to_id = {'O': 0, 'B-MOUNTAIN': 1, 'I-MOUNTAIN': 2}

data_frame = pd.read_csv("Task1/data_for_dataset/ner_dataset.csv")

# Convert string representations of lists back to lists
data_frame['tokens'] = data_frame['tokens'].apply(ast.literal_eval)
data_frame['tags'] = data_frame['tags'].apply(ast.literal_eval)

def map_labels_to_tokens(labels, word_ids):
    """
    Map labels to tokenized inputs.

    Args:
        labels (list): List of label IDs.
        word_ids (list): List of word IDs from tokenization.

    Returns:
        list: Aligned labels with tokenized inputs.
    """
    aligned_labels = [-100] * len(word_ids)  
    label_index = 0  

    for i, word_id in enumerate(word_ids):
        if word_id is not None: 
            if label_index < len(labels):
                aligned_labels[i] = labels[label_index]  
            if i == 0 or word_id != word_ids[i - 1]:  
                label_index += 1

    return aligned_labels

# Tokenize the inputs and align labels
tokenized_inputs = tokenizer(
    data_frame['tokens'].tolist(),
    is_split_into_words=True,
    padding=True,
    truncation=True,
    return_offsets_mapping=True,
    max_length=512
)

# Prepare labels for the tokenized inputs
aligned_labels = []
for i in range(len(data_frame)):
    word_ids = tokenized_inputs.word_ids(batch_index=i)
    labels = [tag_to_id[tag] for tag in data_frame['tags'][i]]
    aligned_labels.append(map_labels_to_tokens(labels, word_ids))

tokenized_inputs['labels'] = aligned_labels

class NERDataset(torch.utils.data.Dataset):
    """
    Custom dataset class for NER.
    """
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, index):
        item = {key: torch.tensor(val[index]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.encodings['labels'][index])
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

ner_dataset = NERDataset(tokenized_inputs)

# Setting up training arguments
training_arguments = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='logs',
    learning_rate=2e-4,
    adam_epsilon=1e-8,
    save_total_limit=3,
    evaluation_strategy='steps',
    save_strategy='steps',
    load_best_model_at_end=True,
)

# Initialization of the Trainer
trainer = Trainer(
    model=model,
    args=training_arguments,
    train_dataset=ner_dataset,
)

trainer.train()

output_directory = "Task1/model_save"
model.save_pretrained(output_directory)
tokenizer.save_pretrained(output_directory)


