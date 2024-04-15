from transformers import GPT3Tokenizer, GPT3Model

# Initialize the huggingface tokenizer and transformer library to get the pretrained GPT3 model
tokenizer = GPT3Tokenizer.from_pretrained('gpt-3')
model = GPT3Model.from_pretrained('gpt-3')

# create user inputs and generate responses
def generate_response(input_text):
    inputs = tokenizer(input_text, return_tensors='pt')
    outputs = model(**inputs)
    return tokenizer.decode(outputs.logits[0], skip_special_tokens=True)

# sample
user_input = input("You: ")
response = generate_response(user_input)
print("Chatbot:", response)
