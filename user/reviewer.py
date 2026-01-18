# from pathlib import Path
# import hashlib, json

# categories = list(FOLDER_TO_PROTOTYPE.keys())

# def review_decision(path: Path, proposed: str, top5, categories) -> str | None:
#     print("\n---")
#     print(f"File: {path}")
#     print(f"Proposed: {proposed}")
#     print("Top matches:")
#     for i, (cat, score) in enumerate(top5, 1):
#         print(f"  {i}. {cat:15s}  {score:.3f}")

#     print("\nChoose:")
#     print("  [Enter] accept proposal")
#     print("  [1-5]   choose from top matches")
#     print("  [name]  type a category name exactly")
#     print("  s       skip")

#     ans = input("> ").strip()

#     if ans == "":
#         return proposed
#     if ans.lower() == "s":
#         return None
#     if ans.isdigit():
#         idx = int(ans) - 1
#         if 0 <= idx < len(top5):
#             return top5[idx][0]
#         print("Invalid number, skipping.")
#         return None

#     # user typed a category name
#     if ans in categories:
#         return ans

#     print("Unknown category, skipping.")
#     return None

# def file_hash(path: Path) -> str:
#     # quick stable ID: path + size + mtime (fast)
#     st = path.stat()
#     s = f"{path}|{st.st_size}|{st.st_mtime}".encode("utf-8", errors="ignore")
#     return hashlib.sha256(s).hexdigest()[:16]

# TO ADD IN RUNNER.PY
# final_folder = review_decision(p, best_folder, top5, categories)
# if final_folder is None:
#     continue

# feedback_record = {
#     "file_id": file_hash(p),
#     "path": str(p),
#     "proposed_folder": best_folder,
#     "final_folder": final_folder,
#     "best_score": best_score,
#     "margin": margin,
# }
# with Path("feedback.jsonl").open("a", encoding="utf-8") as ff:
#     ff.write(json.dumps(feedback_record, ensure_ascii=False) + "\n")
