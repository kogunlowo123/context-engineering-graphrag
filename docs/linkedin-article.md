# Context is the product: what we learned building agentic GraphRAG

An agent asked which supplier its analyzer depends on can answer confidently from a stale, poisoned or unauthorized document. The model is rarely the weak link. The context is.

**The problem.** Plain vector RAG retrieves similar text, not facts that were true at a given time, and it cannot say which sentence supports which claim.

**Why it matters.** Regulated teams must show where an answer came from, who could see it and whether it was grounded before anyone acted on it.

**The flow.** Source -> ingestion (chunking, digests, injection quarantine, secret redaction) -> context (BM25 plus vectors with rank fusion, reranking, a temporal knowledge graph, community summaries, token budgeting) -> agent (planner, retrieval, cited answer) -> tools over MCP and A2A -> governance (identity fabric authorization, groundedness gate) -> output -> observability.

**How we built it.**
1. Every chunk keeps offsets and a SHA-256 digest.
2. Graph relations carry validity intervals, so we can ask what was true last March.
3. Louvain communities feed global questions; personalised PageRank feeds entity questions.
4. Each answer sentence is checked against its cited source and a policy decides release.

**Practices we hold to.** Treat retrieved text as data, never instructions. Fail closed when identity or embeddings are unavailable. Audit digests, not questions. Make every replica rebuild from the database so erasure is real.

**Tradeoffs.** Reranking and evaluation add latency. Model-backed extraction adds cost. A graph adds moving parts. Compliance needs human approval paths. Provider interfaces limit lock-in but do not remove it.

**Value.** Answers that cite versioned sources, withhold themselves when unsupported, and can be erased on request.

**What next.** As agents call each other, context packs become a contract between systems, not a prompt detail.

Question for you: should an agent ever see an answer that failed its groundedness check, if a human is reviewing it?

Code: https://github.com/kogunlowo123/context-engineering-graphrag

#AI #ArtificialIntelligence #GenerativeAI #AgenticAI #LLMOps #EnterpriseAI #AISecurity #RAG #MachineLearning #DataScience #CloudComputing #Kubernetes #MLOps #DigitalTransformation #Innovation #GraphRAG #ContextEngineering #KnowledgeGraph #MCP
