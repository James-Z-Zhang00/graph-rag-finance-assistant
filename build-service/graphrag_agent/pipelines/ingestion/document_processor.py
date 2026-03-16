import os
from typing import List, Dict, Optional, Any

from graphrag_agent.pipelines.ingestion.file_reader import FileReader
from graphrag_agent.pipelines.ingestion.text_chunker import TextChunker
from graphrag_agent.config.settings import FILES_DIR, CHUNK_SIZE, OVERLAP


class DocumentProcessor:
    """
    Document processor that integrates file reading, text chunking, and vector operations.
    """
    
    def __init__(self, directory_path: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP):
        """
        Initialize the document processor.
        
        Args:
            directory_path: File directory path
            chunk_size: Chunk size
            overlap: Chunk overlap size
        """
        self.directory_path = directory_path
        self.file_reader = FileReader(directory_path)
        self.chunker = TextChunker(chunk_size, overlap)
        
    def process_directory(self, file_extensions: Optional[List[str]] = None, recursive: bool = True) -> List[Dict[str, Any]]:
        """
        Process all supported files in a directory.
        
        Args:
            file_extensions: File extensions to process; if None, process all supported types
            recursive: Whether to process subdirectories recursively (default True)
            
        Returns:
            List[Dict]: Results per file including name, content, and chunks
        """
        # Read files
        file_contents = self.file_reader.read_files(file_extensions, recursive=recursive)
        
        # Print debug info
        print(f"DocumentProcessor found {len(file_contents)} files")
        if len(file_contents) > 0:
            print(f"File types: {[os.path.splitext(f[0])[1] for f in file_contents]}")
        
        # Process each file
        results = []
        for filepath, content in file_contents:
            file_ext = os.path.splitext(filepath)[1].lower()
            
            # Build per-file result data
            file_result = {
                "filepath": filepath,  # Relative path
                "filename": os.path.basename(filepath),  # File name only
                "extension": file_ext,
                "content": content,
                "content_length": len(content),
                "chunks": None
            }
            
            # Chunk the text content
            try:
                chunks = self.chunker.chunk_text(content)
                file_result["chunks"] = chunks
                file_result["chunk_count"] = len(chunks)
                
                # Compute each chunk length
                chunk_lengths = [len(''.join(chunk)) for chunk in chunks]
                file_result["chunk_lengths"] = chunk_lengths
                file_result["average_chunk_length"] = sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0
                
            except Exception as e:
                file_result["chunk_error"] = str(e)
                print(f"Chunking error ({filepath}): {str(e)}")
                
            results.append(file_result)
            
        return results
        
    def get_file_stats(self, file_extensions: Optional[List[str]] = None, recursive: bool = True) -> Dict[str, Any]:
        """
        Get file statistics for a directory.
        
        Args:
            file_extensions: File extensions to count; if None, include all supported types
            recursive: Whether to include subdirectories (default True)
            
        Returns:
            Dict: File statistics
        """
        # Read files
        file_contents = self.file_reader.read_files(file_extensions, recursive=recursive)
        
        # Count files by extension
        extension_counts = {}
        total_content_length = 0
        
        # Count subdirectories
        directories = set()
        
        for filepath, content in file_contents:
            ext = os.path.splitext(filepath)[1].lower()
            extension_counts[ext] = extension_counts.get(ext, 0) + 1
            
            # Track the file's subdirectory
            dirpath = os.path.dirname(filepath)
            if dirpath:  # Non-empty means the file is in a subdirectory
                directories.add(dirpath)
                
            if content is not None:
                total_content_length += len(content)
            else:
                print(f"Warning: file {filepath} has None content")
            
        return {
            "total_files": len(file_contents),
            "extension_counts": extension_counts,
            "total_content_length": total_content_length,
            "average_file_length": total_content_length / len(file_contents) if file_contents else 0,
            "directories": list(directories),
            "directory_count": len(directories)
        }
        
    def get_extension_type(self, extension: str) -> str:
        """
        Get a document type description for a file extension.
        
        Args:
            extension: File extension including '.' (e.g., '.pdf')
            
        Returns:
            str: Document type description
        """
        extension_types = {
            '.txt': 'Text file',
            '.pdf': 'PDF document',
            '.md': 'Markdown document',
            '.doc': 'Word document',
            '.docx': 'Word document',
            '.csv': 'CSV data file',
            '.json': 'JSON data file',
            '.yaml': 'YAML config file',
            '.yml': 'YAML config file',
            '.html': 'HTML document',
            '.htm': 'HTML document',
            '.xbrl': 'XBRL document',
        }
        
        return extension_types.get(extension.lower(), 'Unknown type')
        
        
if __name__ == "__main__":
    # Create document processor
    processor = DocumentProcessor(FILES_DIR)
    
    # List all files in the directory
    print(f"All files in {FILES_DIR} and its subdirectories:")
    all_files = processor.file_reader.list_all_files(recursive=True)
    for filepath in all_files:
        print(f"  {filepath}")
    
    # Get file statistics
    stats = processor.get_file_stats(recursive=True)
    print("Directory file stats:")
    print(f"Total files: {stats['total_files']}")
    print(f"Subdirectories: {stats['directory_count']}")
    if stats['directory_count'] > 0:
        print("Subdirectory list:")
        for directory in stats['directories']:
            print(f"  {directory}")
    
    print("File type distribution:")
    for ext, count in stats["extension_counts"].items():
        print(f"  {ext} ({processor.get_extension_type(ext)}): {count} files")
    print(f"Total text length: {stats['total_content_length']} characters")
    print(f"Average file length: {stats['average_file_length']:.2f} characters")
    
    # Process all files
    print("\nStarting to process all files...")
    results = processor.process_directory(recursive=True)
    
    # Print a summary of processing results
    for result in results:
        print(f"\nFile: {result['filepath']}")
        print(f"Type: {processor.get_extension_type(result['extension'])}")
        print(f"Content length: {result['content_length']} characters")
        
        if result.get("chunks"):
            print(f"Chunk count: {result['chunk_count']}")
            print(f"Average chunk length: {result['average_chunk_length']:.2f} characters")
        else:
            print(f"Chunking failed: {result.get('chunk_error', 'Unknown error')}")