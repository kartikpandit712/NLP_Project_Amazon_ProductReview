import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

def load_model_and_tokenizer(model_name):
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

def abstractive_summary(text, model_name, max_input_length=10000, max_output_length=5000):
    tokenizer, model = load_model_and_tokenizer(model_name)
    
    # Tokenize and truncate the input text
    input_ids = tokenizer.encode("summarize: " + text, max_length=max_input_length, return_tensors="pt", truncation=True)

    # Generate a summary
    with torch.no_grad():
        summary_ids = model.generate(input_ids, max_length=max_output_length, num_return_sequences=1, early_stopping=True)

    # Decode and return the summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return summary

if __name__ == "__main__":
    model_name_input = input("Enter the model name (e.g., t5-small, facebook/bart-large-cnn, google/pegasus-large, google/bigbird-roberta-base, allenai/longformer-base-4096): ")
    input_text = input("Enter the text to be summarized:\n")
    summary = abstractive_summary(input_text, model_name_input)
    print("\nSummary:")
    print(summary)
