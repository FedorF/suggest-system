from flask import Flask, request, jsonify
import json

from src.config import DevConfig
from src.utils import QueryHandler

cfg = DevConfig()

app = Flask(__name__)
handler = QueryHandler()


@app.route('/ping', methods=['POST'])
def ping():
    if not handler.model_is_ready:
        return jsonify(status='wait')
    return jsonify(status='ok')


@app.route('/query', methods=['POST'])
def query():
    if not handler.model_is_ready:
        return jsonify(status='FAISS is not initialized!')

    queries = json.loads(request.json)['queries']  # (Dict[str, List[str]])
    lang_check, suggestions = handler.get_suggestion(queries)

    return jsonify(lang_check=lang_check, suggestions=suggestions)


@app.route('/update_index', methods=['POST'])
def update_index():
    documents = json.loads(request.json)['documents']  # Dict[str, str], (id:text)
    handler.update_index(documents)
    index_size = handler.index_size

    return jsonify(status="ok", index_size=index_size)


if __name__ == '__main__':
    app.run(debug=True)
