import time
print("Importing sentence_transformers...")
start = time.time()
from sentence_transformers import SentenceTransformer
print(f"Imported in {time.time() - start:.2f}s")
print("Loading model...")
start = time.time()
model = SentenceTransformer('intfloat/multilingual-e5-small')
print(f"Loaded in {time.time() - start:.2f}s")
