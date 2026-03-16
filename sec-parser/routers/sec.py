import tempfile
import shutil
import requests as http_client

from fastapi import APIRouter, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import Iterator, List, Optional

from config.settings import INGEST_URL

router = APIRouter(prefix="/sec", tags=["sec"])


class SECProcessRequest(BaseModel):
    directory_path: str
    pdf_table_method: str = "img2table"
    file_extensions: Optional[List[str]] = None
    recursive: bool = True


def _stream(request: SECProcessRequest) -> Iterator[str]:
    yield f"Reading files from {request.directory_path}, it may take a few minutes...\n"

    try:
        from sec.sec_filing_processor import SecFilingProcessor
        processor = SecFilingProcessor(
            directory_path=request.directory_path,
            pdf_table_method=request.pdf_table_method,
        )
        results = processor.process_directory(
            file_extensions=request.file_extensions,
            recursive=request.recursive,
        )
    except Exception as e:
        yield f"Failed: could not parse files — {e}\n"
        return

    try:
        resp = http_client.post(INGEST_URL, json=results, timeout=120)
        resp.raise_for_status()
    except Exception as e:
        yield f"Failed: parsed {len(results)} file(s) but could not send to {INGEST_URL} — {e}\n"
        return

    yield f"Success: {len(results)} file(s) parsed and sent to {INGEST_URL}\n"


@router.post("/process")
def process_sec_filings(request: SECProcessRequest) -> StreamingResponse:
    """
    Parse SEC filings in a directory and forward results to the ingest service.

    Streams a status message immediately, then reports success or failure
    once parsing and forwarding complete.
    """
    return StreamingResponse(_stream(request), media_type="text/plain")


@router.post("/parse")
def parse_uploaded_files(
    files: List[UploadFile] = File(...),
    pdf_table_method: str = "img2table",
):
    """
    Receive uploaded files from build-service, parse them, and return
    structured documents as JSON. Used in the Cloud Run pipeline where
    build-service downloads from GCS and forwards file contents here.
    """
    tmp_dir = tempfile.mkdtemp(prefix="sec-parse-")
    try:
        for upload in files:
            dest = f"{tmp_dir}/{upload.filename}"
            with open(dest, "wb") as f:
                shutil.copyfileobj(upload.file, f)

        from sec.sec_filing_processor import SecFilingProcessor
        processor = SecFilingProcessor(
            directory_path=tmp_dir,
            pdf_table_method=pdf_table_method,
        )
        documents = processor.process_directory()
        return JSONResponse(content={"documents": documents})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
