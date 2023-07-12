import os
import openai
import faiss
import numpy as np

# Set up OpenAI API credentials
openai.api_key = '<YOUR_API_KEY>'


###############################################
# Read the text file and split it into chunks
###############################################
def split_file_content(file_path, chunk_size):
    # Check if the file exists
    if not os.path.isfile(file_path):
        return None

    # Read the file content
    with open(file_path, 'r') as file:
        content = file.read()

    # Split the content into chunks of specified size
    chunks = [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]

    return chunks

###############################################
# Fetch Embeddings for Each Chunk & Store
###############################################


def fetch_embeddings(input_string):
    embedding_vectors = openai.Embedding.create(
        input=input_string,
        model="text-embedding-ada-002"
    )["data"][0]["embedding"]

    return embedding_vectors

# Split the file content
split_text_content = split_file_content('./resources/sample_data.txt', 200)
vectors_array = []

# Build an array of vectors for each chunk of text
for split_text in split_text_content:
    vectors = fetch_embeddings(split_text)
    vectors_array.append(vectors)

###############################################
# Build FAISS Database for Similarity Search
###############################################

dimensionality = len(vectors_array[0]) # dimensionality based on size of vector array.

# Create an empty FAISS index
vector_index = faiss.IndexFlatL2(dimensionality)

# Convert the list of vectors into a NumPy array
vectors__numpy_array = np.array(vectors_array, dtype=np.float32)

# Add vectors to the index
vector_index.add(vectors__numpy_array)

###############################################
# Ask for User Input / Convert User Input to Vector
###############################################

user_query = input("What is your question? ")

user_query_vector_representation = fetch_embeddings(user_query)
query_numpy_array = np.array(user_query_vector_representation, dtype=np.float32)
###############################################
# Perform Similarity Search
###############################################

k = 2 # Number of Nearest neighbors to retrieve

distances, indices = vector_index.search(query_numpy_array.reshape(1, -1), k)

fetched_context = []
# Append all results to fetched context
for i in range(k):
    # Uncomment to see more information about the nearest neighbors & corresponding text chunks
    # print("Nearest neighbor", i+1)
    # print("Distance:", distances[0][i])
    # print("Index:", indices[0][i])
    # print("Vector:", vectors[indices[0][i]])
    # print("Text Chunk:", split_text_content[indices[0][i]])
    fetched_context.append(split_text_content[indices[0][i]])

###############################################
# Prepare Template & Send Data to AI for Answers.
###############################################

Template = """
context: {context}
question: {question}

""".format(question = user_query, context = fetched_context[0])

# Uncomment to see full template that is provided to chatGPT
#print(Template)

# Define your conversation history
conversation = [
    {'role': 'system', 'content': 'You are a helpful assistant who will answer questions based on context provided. If you do not have enough information from the context reply that you have insufficient information.'},
    {'role': 'user', 'content': 'context: In 2020 the Los Angeles Dodgers won the World series.\n question: Who won the World Series in 2020?'},
    {'role': 'assistant', 'content': 'The Los Angeles Dodgers won the World Series in 2020.'},
    {'role': 'user', 'content': 'context: In 2020 the Los Angeles Dodgers won the World series.\n question: Who won the World Series in 2001?'},
    {'role': 'assistant', 'content': 'I do not have enough information from the provided context to answer the question.'},
    {'role': 'user', 'content': Template}
]

# Send a message to ChatGPT
response = openai.ChatCompletion.create(
    model='gpt-3.5-turbo',
    messages=conversation
)

# Retrieve the assistant's reply
reply = response.choices[0].message['content']

# Print the assistant's reply
print("Assistant:", reply)
