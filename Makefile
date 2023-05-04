run_app:
	FLASK_APP=src/run_app.py flask run --port 11000

ping_server:
	python3 test/send_request.py --route ping

update_index:
	python3 test/send_request.py --route update_index --data-path binary/vocab.json

send_query:
	python3 test/send_request.py --route query --data-path binary/query.json
