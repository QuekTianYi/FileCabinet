# Scoring & Similarity

Before any scoring based on the content of the file or folder, there will be a round of scan on the file format and metadata.

## Embedding Model
- sentence-transformers/all-MiniLM-L6-v2
Using a pretrained embedding model that turns text into vectors.

## Category Representation
Each category is represented by a centroid formed from multiple prototype
phrases.

## Similarity Metric
Cosine similarity between normalized embeddings.

## Decision Thresholds
- AUTO_THRESHOLD: minimum confidence
- MARGIN_THRESHOLD: separation from second-best category

## Failure Modes
- Thin text → low confidence
- Overlapping categories → small margins