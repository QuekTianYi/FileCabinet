import json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from handlers.doc_handler import DocHandler
from time import time

start_time = time()
last_print = start_time

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

handlers = [
    # TO ADD IMAGE
    DocHandler()
]


'''
Take 2 vectors, normalize to unit vector, dot product to get the cosine angle between them.
1.0 -> same meaning, 0.0 -> unrelated, <0 -> opposite meaning
'''
def cosine(a, b):
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b))

'''
Converts list of strings to list of normalized NumPy array
'''
def embed_texts(model, texts, batch_size=64):
    return model.encode(texts, batch_size=batch_size, normalize_embeddings=True)

'''

'''
def make_folder_centroids(model, folder_to_prototypes: dict[str, list[str]]):
    folder_centroids = {}
    for folder, protos in folder_to_prototypes.items():
        vecs = embed_texts(model, protos)
        centroid = np.mean(vecs, axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
        folder_centroids[folder] = centroid
    return folder_centroids

'''
Compare file vector to each folder centroid
'''
def propose_destination(model, file_text, folder_centroids):
    # file_text: string representing a file (filename + snippet)
    # returns (best_folder, best_score, second_score, margin)
    fvec = model.encode([file_text], normalize_embeddings=True)[0]
    scored = [(folder, cosine(fvec, cvec)) for folder, cvec in folder_centroids.items()]
    scored.sort(key=lambda x: x[1], reverse=True)
    best_folder, best_score = scored[0]
    second_score = scored[1][1] if len(scored) > 1 else -1.0
    margin = best_score - second_score
    return best_folder, best_score, second_score, margin, scored[:5]

def file_to_text(p: Path) -> str:
    # MVP: filename only. Later: add extracted content snippet.
    return p.name.replace("_", " ").replace("-", " ")

def is_safe_path(p:Path) -> bool:
    return not any(part.startswith(".") for part in p.parts)

def process(path: Path):
    for handler in handlers:
        if handler.can_handle(path):
            text = handler.extract(path)
            return text
    
    return ''

def run(WATCH_DIR: Path, 
        LOG_PATH: Path, 
        AUTO_THRESHOLD: float, 
        MARGIN_THRESHOLD: float, 
        FOLDER_TO_PROTOTYPE: dict):

    count = 0   # For counting number of files processed

    print("Building folder centroids...")
    folder_centroids = make_folder_centroids(model, FOLDER_TO_PROTOTYPE)

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("Starting file scan...")
    with LOG_PATH.open("a", encoding="utf-8") as logf:
        for p in WATCH_DIR.rglob("*"):
            if (not p.is_file()) or (not is_safe_path(p)):
                continue
            text = process(p)
            if not text.strip():
                # fallback to filename at least
                text = file_to_text(p)

            # Heartbeat every N files
            count += 1
            if count % 50 == 0:
                elapsed = time () - start_time
                print(f"[{count}] files processed ({elapsed:.1f}s elapsed)")
      
            best_folder, best_score, second_score, margin, top5 = propose_destination(model, text, folder_centroids)

            decision = "REVIEW"
            if best_score >= AUTO_THRESHOLD and margin >= MARGIN_THRESHOLD:
                decision = "AUTO_SUGGEST"

            record = {
                "path": str(p),
                "text_used": text,
                "best_folder": best_folder,
                "best_score": best_score,
                "margin": margin,
                "decision": decision,
                "top5": top5,
            }
            logf.write(json.dumps(record, ensure_ascii=False) + "\n")

            print(f"{decision:12s}  {p.name}  ->  {best_folder}  (score={best_score:.3f}, margin={margin:.3f})")

    print(f"{count} file(s) processed.")