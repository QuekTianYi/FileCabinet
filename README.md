# File Cabinet

**File Cabinet** is a local, privacy-first personal file organizer that understands files by *meaning* and helps you keep your system clean — carefully.

It is designed to start conservatively and become more automated only when it is safe to do so.

---

## What File Cabinet is

- A **local-first** file organization assistant
- Understands files using semantic similarity
- Supports *progressive automation*, from suggestions to actions
- Built with explicit safety gates and human oversight
- Designed to learn your preferences over time

---

## Automation philosophy

File Cabinet is intentionally designed around **levels of automation**.

It does not jump straight to destructive actions.

Instead, automation is enabled gradually as confidence increases.

Typical progression:

1. **Suggest only**  
   - File Cabinet proposes actions
   - You review and approve

2. **Auto-move with safeguards**  
   - High-confidence files are moved automatically
   - Low-confidence cases require review
   - All actions are logged and reversible

3. **Auto-clean with protection**  
   - Duplicate detection
   - Archive-before-delete
   - Grace periods and undo windows

4. **Auto-delete (explicitly enabled)**  
   - Only after:
     - user opt-in
     - strict confidence thresholds
     - clear rules
     - recovery options

Destructive actions are *never* the default.

---

## What File Cabinet is NOT

- ❌ Not a cloud service
- ❌ Not an opaque “AI cleaner”
- ❌ Not a one-click destructive tool
- ❌ Not something that acts without clear rules

Automation is intentional, reviewable, and reversible.

---

## How it works (high level)

1. Files are scanned **locally**
2. Filenames and content are converted into semantic representations
3. Files are compared against known categories and history
4. Actions are scored by confidence
5. Safety rules determine whether to:
   - suggest
   - auto-move
   - archive
   - or require review
6. All actions are logged

---

## Setup

1. Copy `example_config.json` to `config.json`
2. Edit `config.json` to match your folders and preferences
3. Run the program

`config.json` is ignored by Git and will not be committed.

---

## Safety & privacy guarantees

- All processing happens on your machine
- No file contents are uploaded
- No telemetry or tracking
- Automation requires explicit opt-in
- All actions are logged
- Reversibility is a core requirement

File Cabinet treats your files as irreplaceable.

---

## Current status

- Similarity-based sorting using pretrained embeddings
- Recursive scanning
- Dry-run and suggestion mode
- Decision logging implemented

---

## Roadmap

- Confidence-based auto-move
- Archive-first cleanup workflows
- Duplicate detection
- Review queue UI
- User-defined automation rules
- Time-delayed deletion with recovery

---

File Cabinet is built to be powerful — but never careless.