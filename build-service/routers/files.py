"""
File management router.

POST /files/upload        — upload one or more source files
GET  /files               — list files in the files directory
DELETE /files/{filename}  — delete a file
"""

import os
from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse

from config.settings import FILES_DIR

router = APIRouter(prefix="/files")


def _files_dir() -> Path:
    p = Path(FILES_DIR)
    p.mkdir(parents=True, exist_ok=True)
    return p


@router.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """Upload one or more source documents to the files directory."""
    saved = []
    for file in files:
        dest = _files_dir() / file.filename
        content = await file.read()
        dest.write_bytes(content)
        saved.append({"filename": file.filename, "size": len(content)})
    return {"uploaded": saved}


@router.get("")
async def list_files():
    """List all files in the files directory."""
    d = _files_dir()
    result = []
    for p in sorted(d.iterdir()):
        if p.is_file():
            stat = p.stat()
            result.append({
                "filename": p.name,
                "size": stat.st_size,
                "last_modified": stat.st_mtime,
            })
    return {"files": result, "count": len(result)}


@router.delete("/{filename}")
async def delete_file(filename: str):
    """Delete a file from the files directory."""
    # Prevent path traversal
    safe_name = Path(filename).name
    target = _files_dir() / safe_name
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"File '{safe_name}' not found")
    target.unlink()
    return {"deleted": safe_name}
