# MP2 Reflection — Mini-RAG: Ask Your Documents

## What worked

The end-to-end RAG pipeline successfully loaded all five Sherlock Holmes story summaries, split them into 86 chunks, generated embeddings with `text-embedding-3-small`, and stored them in the Qdrant `mp2_sherlock` collection. The application retrieved relevant excerpts and generated answers using `gpt-4o-mini`, including citations with story and section metadata.

The validation run achieved the expected source match for both predefined questions and all three learner questions: 5/5 source matches overall.

## What didn't work

The first validation run matched 12 of the 20 expected fact phrases. One example was the question about Vincent Spaulding's real identity: the initial answer identified Spaulding but did not identify John Clay. Increasing the minimum retrieval context from three to five chunks improved the overall expected-fact phrase matches to 15/20.

The fact-match score is based on literal phrase matching, so it does not fully measure semantic correctness. The remaining unmatched phrases should be reviewed rather than automatically treated as incorrect answers.

## What I'd change

I would evaluate retrieval quality systematically using different chunk sizes, overlap values, and top-k settings. I would also improve the validation harness to assess semantic correctness in addition to literal phrase matching, and add automated unit tests for chunking, metadata, and citation formatting.

For a larger corpus, I would add retry/backoff for transient API failures and monitor API usage and latency.

## One surprise

Retrieving the correct story did not always guarantee a complete answer. The assistant's identity was present in the source summary, but the initial response omitted it. Increasing the retrieval context improved fact coverage, illustrating the importance of both retrieval quality and answer grounding.

## Validation record

- Ingestion: successful; 5 documents and 86 chunks processed.
- Predefined questions: 2/2 expected-source matches.
- Learner questions: 3/3 expected-source matches.
- Total expected-source matches: 5/5.
- Expected-fact phrase matches after the retrieval adjustment: 15/20.
- Average observed answer latency during the final validation run: approximately 2.75 seconds.

Validation command: `python mp2_rag.py validate`

Note: The source-match and fact-match results measure different things. A source match indicates that the expected story was cited; a fact match indicates that an expected phrase appeared in the generated answer.
