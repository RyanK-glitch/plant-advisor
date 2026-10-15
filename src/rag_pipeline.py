import os
import re
import shutil
from pathlib import Path
import pandas as pd
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

PROJECT_DIR = Path(__file__).resolve().parent.parent


class RegulatoryRAG:
    # Chemical names and doses come only from the verified PPQS/CIBRC table, so the
    # other knowledge-base files contribute symptoms, conditions and prevention only
    TABLE_SHEETS = {
        "crop_disease_rag_advisory_india.xlsx": {
            # Row 1 is a title, so the column names are on row 2
            "Advisory Data": (1, [
                "record_id", "crop", "condition", "causal_agent", "agent_type",
                "key_symptoms", "favouring_conditions", "integrated_management"
            ])
        },
        "india_ppqs_cibrc_verified_treatment_table.xlsx": {
            "Regulatory_Treatments": (0, None)
        }
    }

    # Chemical table of each advisory PDF, up to the prevention section
    PDF_CHEMICAL_SECTION = re.compile(
        r"Chemical treatment\s*-\s*only after the disease is confirmed.*?"
        r"(?=Prevention and integrated management)",
        re.S | re.I
    )
    PDF_PAGE_FURNITURE = re.compile(
        r"^(Plant Disease Safe-RAG Knowledge Base.*|"
        r"Verify the current product label and local registration before field use.*)$",
        re.M
    )

    def __init__(self, persist_directory=PROJECT_DIR / "chroma_db", load_existing=True):
        self.persist_directory = str(persist_directory)

        # BGE embeddings are normalized for reliable similarity comparison
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-en-v1.5",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

        self.vector_store = None
        # Skip opening the old database when a clean rebuild is requested
        if load_existing and os.path.exists(self.persist_directory):
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )

    @staticmethod
    def _clean(value):
        return str(value).strip() if pd.notna(value) else ""

    def _load_pdf_file(self, file_path):
        # One document per advisory PDF, with its chemical table removed
        text = "\n".join(page.extract_text() or "" for page in PdfReader(str(file_path)).pages)
        text = self.PDF_PAGE_FURNITURE.sub("", text)
        text = self.PDF_CHEMICAL_SECTION.sub(
            "Chemical treatment: use only the verified PPQS/CIBRC safety rule for this disease.\n",
            text
        )

        record_id = re.search(r"Record ID\s+(ADV-\d+)", text)
        record_key = re.search(r"canonical key:\s*([a-z0-9_]+)", text)
        return Document(page_content=text.strip(), metadata={
            "source": str(file_path),
            "record_id": record_id.group(1) if record_id else "",
            "record_key": record_key.group(1) if record_key else "",
            "crop_key": record_key.group(1).split("_")[0] if record_key else ""
        })

    def _read_tables(self, file_path):
        if file_path.suffix.lower() == ".csv":
            return {"CSV": pd.read_csv(file_path)}

        sheet_rules = self.TABLE_SHEETS.get(file_path.name.lower())
        if sheet_rules is None:
            return pd.read_excel(file_path, sheet_name=None, engine="openpyxl")

        # Known workbooks: only the data sheets, and only the allowed columns
        tables = {}
        for sheet_name, (header_row, columns) in sheet_rules.items():
            dataframe = pd.read_excel(
                file_path, sheet_name=sheet_name, header=header_row, engine="openpyxl"
            )
            dataframe = dataframe.loc[:, ~dataframe.columns.astype(str).str.startswith("Unnamed")]
            if columns:
                dataframe = dataframe[[column for column in columns if column in dataframe.columns]]
            tables[sheet_name] = dataframe
        return tables

    def _load_table_file(self, file_path):
        # Store each spreadsheet row as one structured RAG document
        file_path = Path(file_path)
        tables = self._read_tables(file_path)
        documents = []

        for sheet_name, dataframe in tables.items():
            for row_index, row in dataframe.fillna("").iterrows():
                values = {column: self._clean(value) for column, value in row.items()}
                content = " | ".join(f"{key}: {value}" for key, value in values.items() if value)
                if not content:
                    continue

                documents.append(Document(page_content=content, metadata={
                    "source": str(file_path), "sheet": sheet_name,
                    "row": int(row_index) + 2,
                    "record_id": values.get("record_id", ""),
                    "crop_key": values.get("crop_key", values.get("crop", "")).lower(),
                    "disease_key": values.get("disease_key", "").lower(),
                    "regulatory_status": values.get("regulatory_status", values.get("status", "")),
                    "source_url": values.get("source_url", "")
                }))

        return documents

    def ingest_documents(self, docs_path=PROJECT_DIR / "knowledge_base", rebuild=True):
        # Rebuild to prevent outdated metadata and duplicated chunks
        if rebuild and os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)

        all_documents = [
            self._load_pdf_file(pdf_file)
            for pdf_file in sorted(Path(docs_path).rglob("*.pdf"))
        ]

        for file_path in Path(docs_path).rglob("*"):
            if file_path.suffix.lower() in (".csv", ".xlsx"):
                all_documents.extend(self._load_table_file(file_path))

        if not all_documents:
            print("No supported knowledge-base documents were found.")
            return

        chunks = RecursiveCharacterTextSplitter(
            chunk_size=900, chunk_overlap=120,
            separators=["\n\n", "\n", ". ", " ", ""]
        ).split_documents(all_documents)

        self.vector_store = Chroma.from_documents(
            chunks, self.embeddings,
            persist_directory=self.persist_directory
        )
        print(f"✅ Ingested {len(chunks)} RAG chunks successfully.")

    def query_knowledge_base(self, query, crop_key="", disease_key="", k=4):
        if self.vector_store is None:
            return {"context": "", "records": [], "has_verified": False}

        # Retrieve exact disease records first, then supporting semantic evidence
        exact = self.vector_store.similarity_search(
            query, k=2, filter={"disease_key": disease_key}
        ) if disease_key else []
        semantic = self.vector_store.similarity_search(query, k=k)

        documents = []
        seen = set()
        for document in exact + semantic:
            identity = (document.page_content, document.metadata.get("source", ""))
            if identity not in seen:
                seen.add(identity)
                documents.append(document)

        # Keep only evidence that matches the detected disease or crop
        disease_phrase = disease_key.removeprefix(f"{crop_key}_").replace("_", " ")
        filtered = [
            doc for doc in documents
            if doc.metadata.get("disease_key") == disease_key
            or crop_key in doc.page_content.lower()
            and disease_phrase in doc.page_content.lower()
        ]
        filtered = (filtered or exact)[:k]

        records = [
            {"text": doc.page_content, "metadata": doc.metadata}
            for doc in filtered
        ]
        context = "\n\n".join(record["text"] for record in records)
        has_verified = any(
            str(record["metadata"].get("regulatory_status", "")).upper().startswith("VERIFIED")
            for record in records
        )

        return {"context": context, "records": records, "has_verified": has_verified}


if __name__ == "__main__":
    # Run this file whenever the knowledge base changes
    RegulatoryRAG(load_existing=False).ingest_documents(rebuild=True)
