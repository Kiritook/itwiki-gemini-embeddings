"""Demo di ricerca semantica (Hugging Face Space, Gradio).
Secrets del Space: GEMINI_API_KEY (Google AI Studio, tier gratuito).
requirements.txt: gradio datasets faiss-cpu numpy google-genai
"""
import os
import numpy as np, faiss, gradio as gr
from datasets import load_dataset
from google import genai
from google.genai.types import EmbedContentConfig

REPO = os.environ.get("DATASET", "zionino/itwiki-gemini-embeddings")
MAX = int(os.environ.get("MAX_ROWS", "1000000"))  # limita la RAM del Space

# Solo il primo chunk di ogni voce (incipit): indice leggero ma utile
ds = load_dataset(REPO, split="train").filter(lambda r: r["id"].endswith("_0"))
ds = ds.select(range(min(MAX, len(ds))))
X = np.stack(ds.with_format("numpy")["embedding"]).astype("float32")
faiss.normalize_L2(X)
index = faiss.IndexFlatIP(X.shape[1])
index.add(X)

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
cfg = EmbedContentConfig(task_type="RETRIEVAL_QUERY", output_dimensionality=X.shape[1])


def search(q, k=5):
    v = client.models.embed_content(model="gemini-embedding-2", contents=q, config=cfg)
    v = np.array([v.embeddings[0].values], dtype="float32")
    faiss.normalize_L2(v)
    D, I = index.search(v, int(k))
    out = []
    for d, i in zip(D[0], I[0]):
        r = ds[int(i)]
        out.append(f"**[{r['title']}]({r['url']})** · {d:.3f}\n\n{r['text'][:400]}…")
    return "\n\n---\n\n".join(out)


gr.Interface(search, [gr.Textbox(label="Domanda"), gr.Slider(1, 20, 5, step=1, label="Risultati")],
             gr.Markdown(), title="Ricerca semantica su Wikipedia IT").launch()
