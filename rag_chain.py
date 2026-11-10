"""
DocuMind AI - RAG Chain Module
Orchestrates the retrieval-augmented generation workflow:
Query Reformulation -> Semantic Retrieval -> Context Assembly ->
Grounded Generation -> Citation Verification & Metadata Mapping.
"""

import re
from typing import List, Dict, Any, Tuple, Optional
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from config import (
    DEFAULT_GROQ_MODEL,
    DEFAULT_TEMPERATURE,
    MAX_RESPONSE_TOKENS,
    DEFAULT_TOP_K,
    DEFAULT_SIMILARITY_THRESHOLD,
    REFUSAL_RESPONSE,
    get_groq_api_key
)
from vector_store import VectorStoreManager
from prompts import (
    QA_PROMPT,
    CONDENSE_QUESTION_PROMPT,
    format_context_passage
)


class RAGPipeline:
    """
    Complete end-to-end RAG pipeline with citation verification
    and conversational query contextualization.
    """

    def __init__(
        self,
        vector_store_manager: VectorStoreManager,
        groq_api_key: Optional[str] = None,
        model_name: str = DEFAULT_GROQ_MODEL,
        temperature: float = DEFAULT_TEMPERATURE
    ):
        self.vector_store_manager = vector_store_manager
        self.groq_api_key = groq_api_key or get_groq_api_key()
        self.model_name = model_name
        self.temperature = temperature
        self._llm = None

    def _get_llm(self):
        """Lazy initialization of ChatGroq client."""
        if self._llm is None:
            if not self.groq_api_key:
                raise ValueError(
                    "Groq API Key is not set. Please provide your Groq API key in the sidebar or via the GROQ_API_KEY environment variable."
                )
            from langchain_groq import ChatGroq
            self._llm = ChatGroq(
                groq_api_key=self.groq_api_key,
                model_name=self.model_name,
                temperature=self.temperature,
                max_tokens=MAX_RESPONSE_TOKENS
            )
        return self._llm

    def update_config(self, groq_api_key: Optional[str] = None, model_name: Optional[str] = None, temperature: Optional[float] = None):
        """Update runtime configurations."""
        if groq_api_key is not None and groq_api_key != self.groq_api_key:
            self.groq_api_key = groq_api_key
            self._llm = None
        if model_name is not None and model_name != self.model_name:
            self.model_name = model_name
            self._llm = None
        if temperature is not None and temperature != self.temperature:
            self.temperature = temperature
            self._llm = None

    def reformulate_query(self, question: str, chat_history: List[Dict[str, str]]) -> str:
        """
        Reformulate follow-up questions into standalone search queries
        using conversation context.
        """
        if not chat_history:
            return question

        # If only 1 turn or simple query without pronouns, return directly
        history_text = "\n".join(
            f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
            for msg in chat_history[-4:]  # Last 2 conversation rounds
        )

        try:
            llm = self._get_llm()
            prompt_val = CONDENSE_QUESTION_PROMPT.format_messages(
                chat_history=history_text,
                question=question
            )
            response = llm.invoke(prompt_val)
            reformulated = response.content.strip()
            return reformulated if reformulated else question
        except Exception:
            # Fallback to raw question if reformulation fails or no key
            return question

    def extract_and_validate_citations(
        self,
        llm_answer: str,
        retrieved_chunks: List[Tuple[Document, float]]
    ) -> List[Dict[str, Any]]:
        """
        Extract citation markers from LLM output and cross-validate against
        retrieved chunk metadata. Ensures no fabricated pages/sources are surfaced.
        
        Matches formats like:
        - [Source: sample.pdf, Page: 2]
        - [source: sample.pdf, page: 2]
        - (sample.pdf, page 2)
        """
        # Create lookup of verified sources and pages from retrieved documents
        verified_chunks_lookup: Dict[Tuple[str, int], Dict[str, Any]] = {}
        for doc, score in retrieved_chunks:
            source = doc.metadata.get("source", "")
            page = int(doc.metadata.get("page", 0))
            key = (source.lower(), page)
            if key not in verified_chunks_lookup:
                verified_chunks_lookup[key] = {
                    "source": source,
                    "page": page,
                    "score": score,
                    "excerpt": doc.page_content.strip()[:300] + ("..." if len(doc.page_content) > 300 else ""),
                    "chunk_id": doc.metadata.get("chunk_id", "")
                }

        # Regex to capture [Source: <filename>, Page: <number>]
        citation_pattern = re.compile(
            r'\[(?:Source:\s*)?([^,\]]+),\s*(?:Page:\s*)?(\d+)\]',
            re.IGNORECASE
        )

        found_citations: List[Dict[str, Any]] = []
        seen_keys = set()

        for match in citation_pattern.finditer(llm_answer):
            raw_source = match.group(1).strip()
            try:
                page_num = int(match.group(2).strip())
            except ValueError:
                continue

            lookup_key = (raw_source.lower(), page_num)

            # Check exact match or basename match
            matched_entry = None
            if lookup_key in verified_chunks_lookup:
                matched_entry = verified_chunks_lookup[lookup_key]
            else:
                # Try matching by filename ending
                for (v_src, v_pg), v_data in verified_chunks_lookup.items():
                    if (v_src.endswith(raw_source.lower()) or raw_source.lower().endswith(v_src)) and v_pg == page_num:
                        matched_entry = v_data
                        lookup_key = (v_src, v_pg)
                        break

            if matched_entry and lookup_key not in seen_keys:
                seen_keys.add(lookup_key)
                found_citations.append(matched_entry)

        # Fallback: if LLM didn't insert explicit markers but answered from context,
        # return top retrieved chunks as reference sources
        if not found_citations and retrieved_chunks and REFUSAL_RESPONSE.lower() not in llm_answer.lower():
            for doc, score in retrieved_chunks[:2]:
                source = doc.metadata.get("source", "")
                page = int(doc.metadata.get("page", 0))
                key = (source.lower(), page)
                if key not in seen_keys:
                    seen_keys.add(key)
                    found_citations.append({
                        "source": source,
                        "page": page,
                        "score": score,
                        "excerpt": doc.page_content.strip()[:300] + ("..." if len(doc.page_content) > 300 else ""),
                        "chunk_id": doc.metadata.get("chunk_id", "")
                    })

        return found_citations

    def answer_question(
        self,
        question: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        top_k: int = DEFAULT_TOP_K,
        score_threshold: Optional[float] = DEFAULT_SIMILARITY_THRESHOLD,
        filter_source: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute the full RAG pipeline for a user question.
        
        Returns:
            Dict with keys:
            - 'answer': Generated markdown answer string.
            - 'citations': List of verified citation objects.
            - 'retrieved_chunks': List of all retrieved chunk summaries with scores.
            - 'standalone_query': The reformulated query used for vector search.
            - 'error': Optional error string if an issue occurred.
        """
        chat_history = chat_history or []

        # 1. Verify vector store has indexed documents
        if not self.vector_store_manager.is_indexed:
            return {
                "answer": "No documents have been uploaded or indexed yet. Please upload one or more PDF files to begin.",
                "citations": [],
                "retrieved_chunks": [],
                "standalone_query": question,
                "error": "Vector store is empty."
            }

        # 2. Query contextualization for multi-turn chats
        standalone_query = self.reformulate_query(question, chat_history)

        # 3. Retrieve relevant chunks with calibrated similarity scores
        retrieved_results = self.vector_store_manager.similarity_search_with_scores(
            query=standalone_query,
            k=top_k,
            score_threshold=score_threshold,
            filter_source=filter_source
        )

        # 4. Strict handling if no relevant chunks meet threshold
        if not retrieved_results:
            return {
                "answer": REFUSAL_RESPONSE,
                "citations": [],
                "retrieved_chunks": [],
                "standalone_query": standalone_query,
                "error": None
            }

        # 5. Format retrieved context into numbered passages with metadata
        context_blocks = [
            format_context_passage(doc, idx + 1)
            for idx, (doc, _) in enumerate(retrieved_results)
        ]
        context_str = "\n".join(context_blocks)

        # Format history string
        history_str = "\n".join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in chat_history[-6:]
        ) if chat_history else "No prior conversation."

        # 6. Call LLM for grounded generation
        try:
            llm = self._get_llm()
            prompt_val = QA_PROMPT.format_messages(
                context=context_str,
                chat_history=history_str,
                question=question,
                refusal_message=REFUSAL_RESPONSE
            )
            response = llm.invoke(prompt_val)
            raw_answer = response.content.strip()
        except ValueError as ve:
            return {
                "answer": f"**API Key Required**: {str(ve)}",
                "citations": [],
                "retrieved_chunks": [
                    {
                        "source": doc.metadata.get("source"),
                        "page": doc.metadata.get("page"),
                        "score": score,
                        "content": doc.page_content
                    }
                    for doc, score in retrieved_results
                ],
                "standalone_query": standalone_query,
                "error": str(ve)
            }
        except Exception as e:
            return {
                "answer": f"An error occurred while communicating with the LLM provider: {str(e)}",
                "citations": [],
                "retrieved_chunks": [],
                "standalone_query": standalone_query,
                "error": str(e)
            }

        # 7. Extract and validate source citations
        citations = self.extract_and_validate_citations(raw_answer, retrieved_results)

        # Format retrieved chunks summary for UI inspection
        chunk_summaries = [
            {
                "chunk_id": doc.metadata.get("chunk_id", f"chunk_{i}"),
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", 1),
                "score": score,
                "content": doc.page_content
            }
            for i, (doc, score) in enumerate(retrieved_results)
        ]

        return {
            "answer": raw_answer,
            "citations": citations,
            "retrieved_chunks": chunk_summaries,
            "standalone_query": standalone_query,
            "error": None
        }
