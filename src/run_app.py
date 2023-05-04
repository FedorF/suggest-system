from flask import Flask, request, jsonify
import json

from src.config import Config
from src.server import QueryHandler


app = Flask(__name__)

conf = Config()
handler = QueryHandler(conf.EMB_PATH_KNRM, conf.MLP_PATH, conf.VOCAB_PATH)


@app.route('/ping', methods=['GET'])
def ping():
    if handler.model_is_ready:
        return jsonify(status='ok')

    return jsonify(status='wait')


@app.route('/query', methods=['POST'])
def query():
    if handler.index_is_ready:
        queries = json.loads(request.json)['queries']
        lang_check, suggestions = handler.suggest_candidates(queries)
        return jsonify(lang_check=lang_check, suggestions=suggestions)

    return jsonify(status='FAISS is not initialized!')


@app.route('/update_index', methods=['POST'])
def update_index():
    documents = json.loads(request.json)['documents']
    handler.update_index(documents)
    index_size = handler.index_size

    return jsonify(status="ok", index_size=index_size)


handler.build_knrm_model()
