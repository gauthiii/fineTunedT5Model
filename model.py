from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the tokenizer and model for T5
tokenizer = AutoTokenizer.from_pretrained("t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

# Input text for summarization
text = """
Hugging Face is an AI company that develops tools for natural language processing.
It has become widely known for its open-source transformers library, which provides
pre-trained models for various NLP tasks such as text generation, summarization, 
and translation. The company also offers cloud-based services for AI model deployment.
"""

# Prepend the task keyword (T5 requires explicit task description)
input_text = "summarize: " + text

# Tokenize the input text
inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)

# Generate the summary
outputs = model.generate(**inputs, max_length=100, num_beams=4, early_stopping=True)

# Decode and print the summary
summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated Summary:")
print(summary)
