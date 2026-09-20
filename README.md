# itwiki-gemini-embeddings

Script per generare gli embedding di **Wikipedia in italiano** con `gemini-embedding-2` (Google Vertex AI) e una demo di ricerca semantica.

- **Dataset**: https://huggingface.co/datasets/zionino/itwiki-gemini-embeddings
- **Demo**: LINK_SPACE

## Uso

```bash
pip install -r requirements.txt
export GCP_PROJECT=mio-progetto GCP_LOCATION=global
python embed_wiki.py --stima                 # stima token e costo
LIMIT_ARTICLES=2000 python embed_wiki.py     # test
MAX_SHARDS=50 THREADS=40 python embed_wiki.py  # run riprendibile
```

Serve il ruolo `Vertex AI User` (`roles/aiplatform.user`) sull'account che esegue lo script.
Con la quota standard (6000 richieste/min) servono circa 11-12 ore per l'intera Wikipedia IT.

`app.py` è la demo Gradio per Hugging Face Spaces (secret: `GEMINI_API_KEY`).

## Licenza

Codice: Apache-2.0. Dati: CC BY-SA 4.0 (derivati da Wikipedia).
