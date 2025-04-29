import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)
import matplotlib.pyplot as plt
import seaborn as sns

# === Load and preprocess dataset ===
df = pd.read_csv('ISEAR.csv', encoding='latin1')
df = df[['content', 'sentiment']].dropna()

# Encode labels
label2id = {label: idx for idx, label in enumerate(df['sentiment'].unique())}
id2label = {v: k for k, v in label2id.items()}
df['label'] = df['sentiment'].map(label2id)

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Dataset class
class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors='pt')
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.labels[idx]
        }

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['content'], df['label'], test_size=0.2, stratify=df['label'], random_state=42)
train_dataset = EmotionDataset(list(X_train), list(y_train), tokenizer)
test_dataset = EmotionDataset(list(X_test), list(y_test), tokenizer)

# Load model
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

# Training arguments (removed fp16 for safety)
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=50,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100
)

# Compute metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    report = classification_report(labels, preds, output_dict=True, zero_division=0)
    return {
        'accuracy': report['accuracy'],
        'macro_f1': report['macro avg']['f1-score'],
        'macro_precision': report['macro avg']['precision'],
        'macro_recall': report['macro avg']['recall']
    }

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# Train
trainer.train()

# Final Evaluation
print("\n=== Final Evaluation ===")
eval_results = trainer.evaluate()
print(eval_results)

# Detailed classification report
def print_classification_report(model, dataset):
    model.eval()
    loader = DataLoader(dataset, batch_size=16)
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
    print(classification_report(all_labels, all_preds, target_names=[id2label[i] for i in range(len(id2label))]))

print("\n=== Detailed Classification Report ===")
print_classification_report(model, test_dataset)

# === Confusion matrix display ===
def show_confusion_matrix(model, dataset, tokenizer, id2label):
    model.eval()
    loader = DataLoader(dataset, batch_size=16)
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    cm = confusion_matrix(all_labels, all_preds)
    labels = [id2label[i] for i in range(len(id2label))]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

show_confusion_matrix(model, test_dataset, tokenizer, id2label)

# === Error Analysis: Confusion Matrix ===
def show_confusion_matrix_for_error_analysis(model, dataset, id2label):
    model.eval()
    loader = DataLoader(dataset, batch_size=16)
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    cm = confusion_matrix(all_labels, all_preds)
    labels = [id2label[i] for i in range(len(id2label))]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Emotion')
    plt.ylabel('Actual Emotion')
    plt.title('Confusion Matrix - Error Analysis')
    plt.show()

# Generate confusion matrix for error analysis
show_confusion_matrix_for_error_analysis(model, test_dataset, id2label)

# === LIVE CHAT LOOP ===
# Prediction function
def predict_bert_sentiment(text):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=1).item()
    return id2label[predicted_class_id]

# Predefined simple responses
responses = {
    "joy": "Yay! I'm happy to hear that! 😊",
    "anger": "I'm sorry you're feeling angry. 😠",
    "sadness": "Sending virtual hugs. 🤗",
    "fear": "Stay strong! You got this. 💪",
    "love": "Love is a beautiful thing! ❤",
    "surprise": "Wow, that’s surprising!😲"
}

# Chat loop
print("\nHi! I’m your upgraded BERT-powered chatbot. Type something (or 'quit' to stop).")
while True:
    user_input = input("> ")
    if user_input.lower() == 'quit':
        print("Goodbye!")
        break
    sentiment = predict_bert_sentiment(user_input)
    response = responses.get(sentiment, "Hmm, I’m not sure how to respond to that!")
    print(f"Chatbot: {response} (Detected emotion: {sentiment})")
