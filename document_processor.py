"""
DocuMind AI - Document Processing Module
Extracts text from PDF documents using PyPDF, validates inputs,
handles edge cases (corrupted, password-protected, empty files),
and chunks text using RecursiveCharacterTextSplitter while preserving
fine-grained page-level metadata for citations.
"""

import io
from pathlib import Path
from typing import List, Tuple, Dict, Any, BinaryIO, Union
import pypdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    CHUNK_SEPARATORS,
    MAX_FILE_SIZE_MB,
    ALLOWED_EXTENSIONS
)


class PDFProcessingError(Exception):
    """Custom exception raised during PDF validation or extraction."""
    pass


class DocumentProcessor:
    """
    Handles PDF validation, text extraction, page-level metadata enrichment,
    and semantic chunking.
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=CHUNK_SEPARATORS,
            length_function=len,
            is_separator_regex=False
        )

    @staticmethod
    def validate_file(file_obj: Union[BinaryIO, bytes, io.BytesIO], filename: str) -> None:
        """
        Validate file format and size limits.
        
        Args:
            file_obj: The file-like stream or bytes.
            filename: The original filename with extension.
        
        Raises:
            PDFProcessingError: If validation fails.
        """
        # 1. Extension validation
        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            raise PDFProcessingError(
                f"Unsupported file type '{suffix}'. Only PDF files ({', '.join(ALLOWED_EXTENSIONS)}) are supported."
            )

        # 2. File size validation
        if hasattr(file_obj, "getbuffer"):
            size_bytes = file_obj.getbuffer().nbytes
        elif hasattr(file_obj, "seek") and hasattr(file_obj, "tell"):
            file_obj.seek(0, io.SEEK_END)
            size_bytes = file_obj.tell()
            file_obj.seek(0)
        elif isinstance(file_obj, (bytes, bytearray)):
            size_bytes = len(file_obj)
        else:
            size_bytes = 0

        max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
        if size_bytes > max_bytes:
            raise PDFProcessingError(
                f"File '{filename}' ({size_bytes / (1024*1024):.1f} MB) exceeds maximum allowed size of {MAX_FILE_SIZE_MB} MB."
            )
        if size_bytes == 0:
            raise PDFProcessingError(f"File '{filename}' is empty (0 bytes).")

    def extract_pages(
        self,
        file_obj: Union[BinaryIO, bytes, io.BytesIO],
        filename: str
    ) -> List[Document]:
        """
        Extract text from each page of a PDF, maintaining exact page numbering.
        
        Args:
            file_obj: File-like object containing PDF bytes.
            filename: Original name of the document.
            
        Returns:
            List of LangChain Document objects with page-level metadata.
        """
        self.validate_file(file_obj, filename)

        # Ensure stream is a seekable file-like object
        if isinstance(file_obj, (bytes, bytearray)):
            stream = io.BytesIO(file_obj)
        else:
            stream = file_obj

        try:
            reader = pypdf.PdfReader(stream)
        except pypdf.errors.PdfReadError as e:
            raise PDFProcessingError(f"Could not parse '{filename}'. The file may be corrupted: {str(e)}")
        except Exception as e:
            raise PDFProcessingError(f"Unexpected error opening '{filename}': {str(e)}")

        # Check for password encryption
        if reader.is_encrypted:
            try:
                # Attempt empty password decryption
                decrypted = reader.decrypt("")
                if decrypted == 0:
                    raise PDFProcessingError(f"File '{filename}' is password-protected. Please provide an unencrypted PDF.")
            except Exception:
                raise PDFProcessingError(f"File '{filename}' is password-protected and cannot be read.")

        total_pages = len(reader.pages)
        if total_pages == 0:
            raise PDFProcessingError(f"File '{filename}' contains 0 pages.")

        page_documents: List[Document] = []
        total_extracted_chars = 0

        for page_idx, page in enumerate(reader.pages):
            page_num = page_idx + 1  # 1-indexed page numbering for human citations
            try:
                text = page.extract_text() or ""
            except Exception as e:
                # If a single page fails extraction, log and continue with remaining pages
                text = ""

            clean_text = text.strip()
            total_extracted_chars += len(clean_text)

            if clean_text:
                page_documents.append(
                    Document(
                        page_content=clean_text,
                        metadata={
                            "source": filename,
                            "page": page_num,
                            "total_pages": total_pages
                        }
                    )
                )

        if total_extracted_chars == 0:
            raise PDFProcessingError(
                f"No extractable text found in '{filename}'. "
                "The PDF might contain only scanned images or bitmaps without an OCR text layer."
            )

        return page_documents

    def process_and_chunk(
        self,
        file_obj: Union[BinaryIO, bytes, io.BytesIO],
        filename: str
    ) -> Tuple[List[Document], Dict[str, Any]]:
        """
        Extract and split a PDF into citation-ready chunk Documents.
        
        Args:
            file_obj: File stream or bytes of the PDF.
            filename: Name of the PDF file.
            
        Returns:
            Tuple of (chunk_documents, extraction_stats_dict)
        """
        page_docs = self.extract_pages(file_obj, filename)
        chunk_docs: List[Document] = []

        chunk_counter = 0
        for doc in page_docs:
            source = doc.metadata["source"]
            page = doc.metadata["page"]
            total_pages = doc.metadata["total_pages"]

            # Split the single page content into smaller semantic chunks
            chunks = self.text_splitter.split_text(doc.page_content)
            for idx, chunk_text in enumerate(chunks):
                chunk_id = f"{source}_p{page}_c{idx}"
                chunk_doc = Document(
                    page_content=chunk_text,
                    metadata={
                        "source": source,
                        "page": page,
                        "total_pages": total_pages,
                        "chunk_index": idx,
                        "chunk_id": chunk_id,
                        "char_count": len(chunk_text)
                    }
                )
                chunk_docs.append(chunk_doc)
                chunk_counter += 1

        stats = {
            "filename": filename,
            "total_pages": page_docs[0].metadata["total_pages"] if page_docs else 0,
            "pages_with_text": len(page_docs),
            "total_chunks": len(chunk_docs),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap
        }

        return chunk_docs, stats
