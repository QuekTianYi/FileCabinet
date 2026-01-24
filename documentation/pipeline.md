# Sorting Pipeline

## Step 1: Discovery
- Only top-level folders, zip files, and loose files are scanned
- Hidden paths are ignored

## Step 2: Text Extraction
- FolderHandler: samples file contents
- ZipHandler: samples internal filenames and text
- DocHandler: extracts file content

## Step 3: Text Normalization
Each item is converted into a structured semantic string:
- name
- extension
- parent folder
- sampled content

## Step 4: Scoring
Cosine similarity is computed against category centroids.

## Step 5: Decision
- AUTO_SUGGEST if score ≥ threshold and margin ≥ threshold
- REVIEW otherwise