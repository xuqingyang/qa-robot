# qa-robot

A simple Question-Answer robot. The embedding model uses [SentenceTransformers](https://www.sbert.net/). The search engine uses [faiss](https://github.com/facebookresearch/faiss). The service framework uses [Flask](https://flask.palletsprojects.com/en/2.0.x/). The demo data uses [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) dev set.

## requirement
```
pip install -r requirements.txt
```

## run demo
```
flask run
```
Then open browser http://127.0.0.1:5000/ 