# Sentiment Analysis with DistilBERT and LoRA

This repository contains a sentiment analysis model fine-tuned using the `distilbert-base-uncased-finetuned-sst-2-english` model from Hugging Face. The model is further enhanced with LoRA (Low-Rank Adaptation) for efficient fine-tuning and supports hyperparameter tuning with Optuna. It is trained on the IMDb dataset.

## Features

- **LoRA Integration:** Efficient fine-tuning using the PEFT (Parameter-Efficient Fine-Tuning) library.
- **Hyperparameter Optimization:** Uses Optuna for tuning batch size and learning rate.
- **Logging & Monitoring:** Utilizes TensorBoard for tracking training metrics.
- **Model Checkpointing:** Saves model checkpoints at each epoch.
- **Robust Evaluation Metrics:** Computes accuracy, precision, recall, F1-score, and confusion matrix.
- **Interactive Inference:** Allows real-time sentiment classification of user-provided text.

## Data

The model is fine-tuned using the IMDb dataset, which is publicly available from [Hugging Face Datasets](https://huggingface.co/datasets/imdb). Ensure you adhere to the dataset's license and usage terms.

## Setup

#### Requirements

Ensure you have the following dependencies installed:

```bash
pip install torch transformers datasets optuna scikit-learn tensorboard peft jupyter
```

#### Clone the Repository

```bash
git clone https://github.com/Ziicco/LoRA-Sentiment-Analysis.git
cd LoRA-Sentiment-Analysis
```

## Running the Notebook

Open the Jupyter Notebook to start training and evaluation:

```bash
jupyter notebook sentiment_analysis.ipynb
```

The notebook will:
1. Load a pre-trained DistilBERT model.
2. Integrate LoRA for efficient fine-tuning.
3. Perform hyperparameter tuning with Optuna.
4. Train and evaluate the model on a subset of the IMDb dataset.
5. Save model checkpoints for future use.

## Customizing the Model
You can replace the DistilBERT model with any other Hugging Face model by changing the `MODEL_NAME` variable in the notebook:

```python
MODEL_NAME = "bert-base-uncased"
```
To use a different dataset, modify the dataset loading parts of the code in `def objective(trial)` and `if __name__ == "__main__"`:
```python
train_full = load_dataset("new_dataset_name", split="train")
    .
    .
    .
test_full = load_dataset("new_dataset_name", split="test")
```

## Hyperparameter Tuning

The model uses Optuna to optimize learning rate and batch size. You can customize the number of trials by modifying:

```python
study.optimize(objective, n_trials=10)
```

## Model Evaluation

After training, the notebook computes evaluation metrics:
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **Confusion Matrix**

## Inference

To classify text interactively, run the relevant cell in the notebook and enter text when prompted:

```bash
jupyter notebook sentiment_analysis.ipynb
```

Example:

```
Enter text for sentiment analysis (or type 'exit' to quit): The movie was fantastic!
Sentiment: Positive
Probabilities: [0.1, 0.9]
```

## Logging with TensorBoard

You can visualize training progress with TensorBoard:

```bash
tensorboard --logdir=runs
```

Then open `http://localhost:6006/` in your browser.

## Checkpoints

Model checkpoints are saved in `model_checkpoints/` after each epoch:

```bash
model_checkpoints/model_epoch_1.pt
model_checkpoints/model_epoch_2.pt
```

## Contact
For any issues or questions, please open an issue on the [GitHub repository](https://github.com/Ziicco/LoRA-Sentiment-Analysis/issues).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 

## Author

Isaac Ojiaku (@Ziicco)


Feel free to modify further as needed!
