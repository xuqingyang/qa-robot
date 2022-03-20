from flask import jsonify
from flask import request, Flask, render_template
import pandas as pd
import torch
import faiss
import numpy as np
import os.path
from sentence_transformers import SentenceTransformer


def try_gpu(i=0):
    """try to get gpu"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def create_index(model, data_list, index_file):
    """encode and index questions"""
    print("start indexing")
    # encoding
    encoded_data = model.encode(data_list)

    # make index
    index = faiss.IndexIDMap(faiss.IndexFlatIP(768))
    total = len(data_list)
    index.add_with_ids(encoded_data, np.array(range(0, total)))
    faiss.write_index(index, index_file)
    print(f"created index for {total} questions")


CSV_FILE = "data/demo.csv"
INDEX_FILE = "demo.index"
app = Flask(__name__)

# store data in database when you have big data set.
df = pd.read_csv(CSV_FILE)
question_data = df['question'].tolist()
answer_data = df['answer'].tolist()
net = SentenceTransformer('all-mpnet-base-v2', device=try_gpu())

if not os.path.exists(INDEX_FILE):
    create_index(net, question_data, INDEX_FILE)


@app.route("/")
def hello_world():
    return render_template('index.html')


@app.route("/search", methods=['POST'])
def search():
    data = request.get_json()
    print(data)
    query = data["query"]
    number = int(data["number"])
    index = faiss.read_index(INDEX_FILE)
    q_vec = net.encode([query])
    D, I = index.search(q_vec, number)
    print(D)
    print(I)

    ids = I.flatten()
    for i in ids:
        d = df[df['id'] == i]
        print(f"{i}: question: {d['question'].values}\n{d['answer'].values}")
    return df[df["id"].isin(ids)].to_json(orient='records')