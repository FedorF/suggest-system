# Suggesting system
System of prompts for similar questions on User’s query. <br>
The pipeline is following: <br>
1) The query is filtered by language using the LangDetect [library](https://pypi.org/project/langdetect/). All requests in a language other than English are filtered.
2) Candidates search by the similarity using [faiss](https://github.com/facebookresearch/faiss). In this part pre-trained GLOVE [embeddings](https://nlp.stanford.edu/projects/glove/) has been used.
3) Previously obtained candidates are re-ranked by the [K-NRM](https://arxiv.org/pdf/1706.06613.pdf) model. In the end, the most relevant candidates are returned as a response. K-NRM model was trained on the Quora Question Pairs [dataset](https://gluebenchmark.com/tasks/).

The system is implemented as a micro-service based on Flask. 