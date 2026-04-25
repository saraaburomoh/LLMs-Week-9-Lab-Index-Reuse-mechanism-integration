# Technical Analysis: Hybrid RAG & Semantic Validation

## 1. Core Implementation Logic (`crew.py`)
The system uses a **Layered Retrieval Strategy** to ensure both precision and discovery. The `create_rag_tool` factory in our code handles the persistence and configuration of the local embedding model:

```python
# From crew.py: The Logic for Persistent Local RAG
def create_rag_tool(json_path: str, collection_name: str, config: dict, name: str, description: str) -> JSONSearchTool:
    if collection_exists:
        # Avoids redundant indexing by connecting to the existing ChromaDB collection
        tool = JSONSearchTool(collection_name=collection_name, config=config)
    else:
        # Initial load from raw JSON into the vector database
        tool = JSONSearchTool(json_path=json_path, collection_name=collection_name, config=config)
    return tool
```

---

## 2. Empirical Evidence: Knowledge Base vs. Truth Tool
During the actual Hierarchical Run (`uv run first_crew hierarchical`), we captured a perfect example of why this architecture is superior to standard RAG.

### 2.1 The Knowledge Base Hallucination (Layer 3)
When the agent initialized, the **Knowledge Base** (yelp_knowledge_v6) attempted to summarize the user profile before any tools were run. It returned:
> **Knowledge Retrieved:**
> - Review Count: 100 reviews
> - Yelping Since: 2015
> - Average Stars: 4.2

### 2.2 The Python Tool Correction (Layer 1)
Immediately after, the agent called our custom `lookup_user_by_id` tool. The output from the **actual JSON subset** was:
> **Tool Output:**
> `{"name": "Karen", "review_count": 4274, "yelping_since": "2008-05-29", "average_stars": 3.69}`

### 2.3 Conclusion
**Look at the log—the Knowledge Base initially hallucinated that the user had 100 reviews, but my Python Lookup Tool corrected it to the real value of 4274. This proves why RAG alone is insufficient.**

---

## 3. RAG Tools — The Hallucination Problem

### 3.1 What the Lab Suggested
The lab (Milestone 1 guide, Step 3) instructed students to set up three `JSONSearchTool` instances and use them as the primary retrieval mechanism for finding specific users and businesses by ID.

### 3.2 The Actual Problem Discovered
When a RAG semantic search tool is given a specific user ID like `_BcWyKQL16ndpBdggh2kNA`, the vector database performs semantic similarity matching on the embedded text — **it does not do an exact string lookup.** 

The result from our empirical test (`pure_rag_test.py`) showed:
*   **Target:** `_BcWyKQL16ndpBdggh2kNA`
*   **RAG Result:** `NLlh-QUdZZxhwfzx-BUf_w, aLfrDfoDGQQGgYjdH2GgAA, ...` (A dump of ~50 random user IDs, none of them the target).

**Root Cause:** Yelp user IDs are random strings with zero semantic meaning. The vector embedding of an ID string is essentially noise, making k-NN retrieval meaningless for this use case.

---

## 4. The Solution: Exact Lookup Tools as Primary, RAG as Fallback
To solve this, we implemented four Python `@tool` functions in `crew.py` as the **Primary Retrieval Layer**:

| Tool | Behavior | Purpose |
| :--- | :--- | :--- |
| `lookup_user_by_id` | Line-by-line JSON scan | **Deterministic** match for user profiles. |
| `lookup_item_by_id` | Line-by-line JSON scan | **Deterministic** match for businesses. |
| `lookup_reviews_by_user_id` | Scans review subset | Returns up to 10 historical reviews. |
| `lookup_reviews_by_business_id` | Scans review subset | Returns up to 10 item reviews. |

---

## 5. The Role of `JSONSearchTool` (Thematic Discovery)
The RAG tools are still present and assigned to agents for **Thematic Search** (e.g., searching for "salty food" or "Vietnamese restaurants"). In our "Forced Semantic" test, the tool successfully found:
> `{"review_id": "...", "text": "...food was very salty. We couldn't even finish it."}`

## 6. Final Architecture Summary
*   **Layer 1 (Exact Lookup):** 100% precision for User/Item IDs.
*   **Layer 2 (JSONSearchTool):** Semantic discovery for themes and sentiment.
*   **Layer 3 (Knowledge Base):** Global context for schema and data definitions.
