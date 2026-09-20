"""Embedding di Wikipedia in italiano con Vertex AI (gemini-embedding-2).

Uso:
  export GCP_PROJECT=mio-progetto
  python embed_wiki.py --stima              # stima token/costo, nessuna chiamata API
  LIMIT_ARTICLES=2000 python embed_wiki.py  # test su pochi articoli
  MAX_SHARDS=50 python embed_wiki.py        # run (riprendibile, si ferma dopo N shard)
"""
import os, sys
from concurrent.futures import ThreadPoolExecutor
import pyarrow as pa, pyarrow.parquet as pq
from datasets import load_dataset
from tenacity import retry, wait_exponential, stop_after_attempt

MODEL = "gemini-embedding-2"
DIM = 768              # Matryoshka: 768 basta e riduce lo spazio di 4x
CHUNK_WORDS = 300
BATCH = 1              # gemini-embedding-2 fonde più testi in un solo vettore: 1 per richiesta
THREADS = int(os.environ.get("THREADS", "8"))
SHARD_SIZE = 50_000    # chunk per file parquet
LIMIT = int(os.environ.get("LIMIT_ARTICLES", "0"))
OUT = "out"
PRICE_PER_M = 0.20     # $/1M token (standard)


def chunks(text):
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    buf, n, i = [], 0, 0
    for p in paras:
        w = len(p.split())
        if buf and n + w > CHUNK_WORDS:
            yield i, "\n".join(buf)
            i, buf, n = i + 1, [], 0
        buf.append(p)
        n += w
    if buf:
        yield i, "\n".join(buf)


def rows():
    ds = load_dataset("wikimedia/wikipedia", "20231101.it", split="train", streaming=True)
    for k, a in enumerate(ds):
        if LIMIT and k >= LIMIT:
            break
        for i, c in chunks(a["text"]):
            yield {"id": f'{a["id"]}_{i}', "title": a["title"], "url": a["url"], "text": c}


def stima():
    chars = n = 0
    for r in rows():
        chars += len(r["title"]) + len(r["text"]) + 1
        n += 1
    tok = chars / 4  # approssimazione; verifica sul billing dopo il test
    print(f"chunk: {n:,}  token≈{tok/1e6:,.0f}M  costo≈${tok/1e6*PRICE_PER_M:,.0f}")


def main():
    from google import genai
    from google.genai.types import EmbedContentConfig

    client = genai.Client(vertexai=True, project=os.environ["GCP_PROJECT"],
                          location=os.environ.get("GCP_LOCATION", "global"))
    cfg = EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT", output_dimensionality=DIM)

    @retry(wait=wait_exponential(min=2, max=60), stop=stop_after_attempt(10))
    def embed(texts):
        r = client.models.embed_content(model=MODEL, contents=texts, config=cfg)
        return [e.values for e in r.embeddings]

    def write_shard(num, buf):
        path = f"{OUT}/shard_{num:05d}.parquet"
        if os.path.exists(path):  # ripresa dopo interruzione
            return
        texts = [f'{r["title"]}\n{r["text"]}' for r in buf]
        batches = [texts[i:i + BATCH] for i in range(0, len(texts), BATCH)]
        with ThreadPoolExecutor(THREADS) as ex:
            vecs = [v for b in ex.map(embed, batches) for v in b]
        cols = {k: [r[k] for r in buf] for k in buf[0]}
        cols["embedding"] = pa.array(vecs, type=pa.list_(pa.float32()))
        pq.write_table(pa.table(cols), path + ".tmp")
        os.rename(path + ".tmp", path)
        print("ok", path, flush=True)

    os.makedirs(OUT, exist_ok=True)
    buf, num = [], 0
    for r in rows():
        buf.append(r)
        if len(buf) == SHARD_SIZE:
            write_shard(num, buf)
            buf, num = [], num + 1
            if num >= int(os.environ.get("MAX_SHARDS", "999")): return
    if buf:
        write_shard(num, buf)


if __name__ == "__main__":
    stima() if "--stima" in sys.argv else main()
