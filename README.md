# Sentiment-Aware Chatbot (BERT)

## Overview
This repository contains the assets for an HCI course project that explores how sentiment-aware responses can make conversations with chatbots feel more empathetic. The core deliverable is a command line application that fine-tunes a `bert-base-uncased` model on the ISEAR emotion dataset, evaluates the classifier, visualizes performance with confusion matrices, and then runs an interactive chat session that echoes the detected emotion with a matching response.

## Features
- Fine-tunes BERT for multi-class emotion classification on the ISEAR dataset.
- Reports accuracy, macro-averaged F1, precision, and recall, alongside a detailed classification report.
- Generates two confusion matrix heatmaps for quick error analysis.
- Provides a lightweight CLI chat loop that predicts the user's emotion in real time and replies with pre-scripted empathetic messages.

## Project Layout
- `Source Code/main.py` – end-to-end training, evaluation, visualization, and chat loop.
- `ISEAR.csv` – required dataset (not included; must be supplied by the user).
- `HCI Final PPT.pptx` – presentation slide deck that accompanies the project.
- `REPORT.pdf` – written report describing design decisions and findings.

## Requirements
- Python 3.8 or newer
- A machine with enough memory/compute to fine-tune BERT (a GPU is recommended but the script also runs on CPU at reduced speed)
- Python packages: `pandas`, `torch`, `scikit-learn`, `transformers`, `matplotlib`, `seaborn`

### Suggested environment setup
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install --upgrade pip
pip install pandas torch scikit-learn transformers matplotlib seaborn
```

## Dataset Preparation
1. Download the ISEAR (International Survey on Emotion Antecedents and Reactions) dataset from an authorized source.
2. Save it as `ISEAR.csv` in the project root (same folder as `Source Code/`).
3. Ensure the file includes at least two columns named `content` (the text) and `sentiment` (the emotion label). Example:

   | content                      | sentiment |
   |------------------------------|-----------|
   | I laughed when I saw my dog. | joy       |

The script automatically maps the unique sentiment values it finds to numerical labels, so you can extend the dataset with additional emotions if they are consistently labeled.

## Running the Project
1. From the `Source Code/` directory, execute the main script:
   ```bash
   cd "Source Code"
   python main.py
   ```
2. The script performs the following stages:
   - Loads and preprocesses the dataset.
   - Splits the data (80/20 stratified train/test split).
   - Fine-tunes `bert-base-uncased` for two epochs with the Hugging Face Trainer API.
   - Prints evaluation metrics and a scikit-learn classification report.
   - Displays confusion matrix heatmaps using Matplotlib/Seaborn.
   - Launches an interactive chat loop where you can type messages and see the predicted emotion.

### Notes on running
- The confusion matrix plots use `plt.show()`, which opens a window when running in an environment with GUI support. If you are running headless (e.g., over SSH), set a non-interactive backend (for example, `matplotlib.use("Agg")`) before executing the script.
- Training time scales with dataset size. On CPU the run can take several minutes; on GPU it is substantially faster.

## Interactive Chatbot Usage
Once training and evaluation finish, the terminal prints a prompt similar to:
```
Hi! I’m your upgraded BERT-powered chatbot. Type something (or 'quit' to stop).
```
Type any message to receive an emotion-aware response. Enter `quit` to exit.

Default emotions and replies are defined in the `responses` dictionary inside `main.py`. You can customize them to better fit your use case.

## Customization Tips
- **Change model/parameters:** Modify the `TrainingArguments` block to adjust epochs, batch size, or logging configuration. Swap in a different pretrained model by updating the `from_pretrained` calls.
- **Augment responses:** Extend the `responses` dictionary to cover additional emotions returned by your dataset or to produce more elaborate replies.
- **Persist trained weights:** Add `trainer.save_model("./models/emotive-chatbot")` after training if you want to reuse the fine-tuned checkpoint without retraining.

## Troubleshooting
- *CUDA errors*: Ensure you installed the PyTorch build that matches your CUDA toolkit—or install the CPU-only build if you do not have a compatible GPU.
- *Dataset issues*: Confirm that `ISEAR.csv` is encoded in UTF-8 or Latin-1 and contains the expected column names. Missing or differently named columns will raise a KeyError when the script selects `['content', 'sentiment']`.
- *Matplotlib backend*: If the script hangs after training, it is likely waiting for the confusion matrix window to close. Close the figure or switch to a non-interactive backend as mentioned above.

## Acknowledgements
- The ISEAR dataset was created by the Swiss Center for Affective Sciences.
- Transformers and Trainer utilities are provided by the Hugging Face ecosystem.
- Visualization uses Matplotlib and Seaborn.
