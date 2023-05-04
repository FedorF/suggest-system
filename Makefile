prepare_model:
	unzip binary/model_artefacts.zip -d binary/

prepare_embeddings:
	curl https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip --output binary/glove.zip
	unzip binary/glove.zip -d binary/

run_app:
	FLASK_APP=src/run_app.py flask run --port 11000

ping_server:
	python3 test/send_request.py --route ping

update_index:
	python3 test/send_request.py --route update_index --data-path binary/vocab.json

send_query:
	python3 test/send_request.py --route query --data-path binary/query.json
