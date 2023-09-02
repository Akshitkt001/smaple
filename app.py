import random
import os
import requests
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from flask import Flask, request, jsonify, render_template

# Initialize Flask app
app = Flask(__name__)

# Load GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Function to generate a shuffled creative caption
def generate_shuffled_creative_caption_gpt2(text, category):
    prompt = f"Given the text: '{text}' and category: '{category}', generate a unique and creative caption:"
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2)
    
    generated_caption = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Split the generated caption into words and shuffle them
    words = generated_caption.split()
    random.shuffle(words)
    shuffled_caption = ' '.join(words)
    
    # Limit the caption to 50 words
    caption_words = shuffled_caption.split()[:50]
    final_caption = ' '.join(caption_words)
    
    return final_caption

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle form submission and generate caption
@app.route('/get_caption', methods=['POST'])
def get_caption():
    input_type = request.form['input_type']
    user_input = request.form['user_input']
    category = request.form['category']
    
    # Process input based on the selected input type
    if input_type == 'text':
        # If the input is plain text, generate a caption directly
        caption = generate_shuffled_creative_caption_gpt2(user_input, category)
    elif input_type == 'url':
        # If the input is a URL, fetch the content from the URL and generate a caption
        try:
            response = requests.get(user_input)
            if response.status_code == 200:
                text_content = response.text
                caption = generate_shuffled_creative_caption_gpt2(text_content, category)
            else:
                caption = "Error: Unable to fetch content from the URL."
        except Exception as e:
            caption = f"Error: {str(e)}"
    elif input_type == 'pdf':
        # If the input is a PDF file, you need to handle PDF parsing and text extraction here
        # Replace the following code with your PDF processing logic
        caption = "PDF processing not implemented yet."
    else:
        caption = "Invalid input type selected."
    
    return render_template('index.html', caption=caption)

if __name__ == '__main__':
    app.run(debug=True)
