"""
DocuMind AI - Test PDF Generator
Generates a multi-page test PDF with structured facts across pages
using pure Python (zero external dependencies).
"""

from pathlib import Path


def generate_test_pdf(output_path: str = "sample_company_report.pdf") -> str:
    """
    Generate a clean 4-page PDF with verifiable facts on each page.
    Uses valid raw PDF 1.4 syntax.
    """
    pages_content = [
        # Page 1: Overview
        [
            "DOCUMIND CORP - ANNUAL REPORT 2024",
            "Page 1: Company Overview and Executive Summary",
            "",
            "DocuMind Inc. was founded in 2024 by Dr. Elena Vance in Zurich, Switzerland.",
            "The mission of DocuMind is to eliminate enterprise hallucinations in knowledge retrieval.",
            "The flagship software product is the DocuMind Platform 2.0.",
            "By Q3 2024, DocuMind reached 50,000 active enterprise users across 12 countries.",
            "The company currently employs 42 full-time researchers and software engineers."
        ],
        # Page 2: Financials
        [
            "DOCUMIND CORP - ANNUAL REPORT 2024",
            "Page 2: Financial Performance and Key Metrics",
            "",
            "In fiscal year 2024, total revenue reached 12.4 million USD.",
            "This represented a 45 percent year-over-year increase compared to 2023.",
            "Research and Development expenditure was 3.8 million USD.",
            "DocuMind achieved a net operating profit of 2.1 million USD.",
            "The enterprise software gross margin remained healthy at 78 percent.",
            "The projected revenue for fiscal year 2025 is 19.5 million USD."
        ],
        # Page 3: Technology
        [
            "DOCUMIND CORP - ANNUAL REPORT 2024",
            "Page 3: Technical Architecture and Infrastructure",
            "",
            "The core indexing engine leverages FAISS with cosine similarity over 384-dimensional dense vectors.",
            "The default embedding model deployed is sentence-transformers all-MiniLM-L6-v2.",
            "The retrieval latency SLA is strictly under 150 milliseconds for 99 percent of requests.",
            "All enterprise data is encrypted using AES-256 at rest and TLS 1.3 in transit.",
            "The underlying vector index supports up to 10 million vector records per node."
        ],
        # Page 4: Security & Compliance
        [
            "DOCUMIND CORP - ANNUAL REPORT 2024",
            "Page 4: Security Policies and Compliance",
            "",
            "DocuMind strictly treats all retrieved document passages as untrusted user data.",
            "The system adheres to SOC2 Type II compliance and ISO 27001 certification standards.",
            "All session states are strictly isolated to avoid cross-tenant data leakage.",
            "System instruction injection attempts are blocked by prompt isolation boundaries.",
            "Compliance audits are conducted bi-annually by an independent security advisory board."
        ]
    ]

    objects = []
    
    # Object 1: Catalog
    objects.append("1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
    
    # Object 3: Font
    font_obj_idx = 3
    
    # Page object IDs will be 4, 6, 8, 10
    # Content stream IDs will be 5, 7, 9, 11
    page_ids = []
    stream_objs = []
    
    current_id = 4
    for lines in pages_content:
        page_id = current_id
        stream_id = current_id + 1
        page_ids.append(page_id)
        
        # Build text stream
        stream_text = "BT\n/F1 12 Tf\n50 720 Td\n18 TL\n"
        for line in lines:
            safe_line = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            if "ANNUAL REPORT" in safe_line:
                stream_text += f"/F1 16 Tf ({safe_line}) Tj T* /F1 12 Tf\n"
            else:
                stream_text += f"({safe_line}) Tj T*\n"
        stream_text += "ET\n"
        
        stream_len = len(stream_text.encode("latin-1"))
        
        page_obj = (
            f"{page_id} 0 obj\n"
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
            f"/Contents {stream_id} 0 R /Resources << /Font << /F1 {font_obj_idx} 0 R >> >> >>\n"
            f"endobj\n"
        )
        stream_obj = (
            f"{stream_id} 0 obj\n"
            f"<< /Length {stream_len} >>\n"
            f"stream\n{stream_text}endstream\n"
            f"endobj\n"
        )
        
        stream_objs.append((page_obj, stream_obj))
        current_id += 2
        
    # Object 2: Pages
    kids_str = " ".join(f"{pid} 0 R" for pid in page_ids)
    objects.append(f"2 0 obj\n<< /Type /Pages /Kids [{kids_str}] /Count {len(page_ids)} >>\nendobj\n")
    
    # Object 3: Font Helvetica
    objects.append("3 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n")
    
    # Add page and stream objects
    for p_obj, s_obj in stream_objs:
        objects.append(p_obj)
        objects.append(s_obj)

    # Assemble PDF with cross-reference table
    pdf_bytes = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
    offsets = []
    
    for obj_str in objects:
        offsets.append(len(pdf_bytes))
        pdf_bytes += obj_str.encode("latin-1")
        
    xref_offset = len(pdf_bytes)
    total_objs = len(objects) + 1
    
    xref_table = f"xref\n0 {total_objs}\n0000000000 65535 f \n"
    for off in offsets:
        xref_table += f"{off:010d} 00000 n \n"
        
    trailer = (
        f"trailer\n<< /Size {total_objs} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    )
    
    pdf_bytes += xref_table.encode("latin-1")
    pdf_bytes += trailer.encode("latin-1")
    
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "wb") as f:
        f.write(pdf_bytes)
        
    return str(out_file.resolve())


if __name__ == "__main__":
    path = generate_test_pdf()
    print(f"Generated test PDF: {path}")
