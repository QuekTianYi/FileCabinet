# Architecture

Based on the digital file formats, there should be a first layer of checking.

First Layer: File Format
Second Layer: Metadata
Third Layer: Content

There will be a default category pack, which will be customizable to the user's preferences.

## High-Level Flow
1. Scan top-level items in WATCH_DIR
2. Extract semantic text using handlers
3. Embed text with SentenceTransformer
4. Compare against category centroids
5. Write proposals to JSONL
6. Optional execution step applies moves

## Core Components
- Runner
- Handlers
  - DocHandler
  - FolderHandler
  - ZipHandler
- Embedding + Scoring Engine
- Proposal Log

[ Files / Folders ]
        |
    [ Handlers ]
        |
   [ Text Builder ]
        |
   [ Embeddings ]
        |
[ Similarity Scoring ]
        |
 [ Proposal JSONL ]
