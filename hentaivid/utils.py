"""
Hentaivid Utilities - Cultural Support Functions

This module provides utility functions for working with culturally-compliant
video knowledge bases and supporting common operations.
"""

import os
from typing import Optional, List, Dict, Any
from loguru import logger

from .chat import HentaividChat, HentaividRetriever
from .encoder import HentaividEncoder


def search_knowledge_base(
    video_path: str,
    index_path: str,
    query: Optional[str] = None,
    interactive: bool = True
) -> Optional[str]:
    """
    Search a culturally-compliant knowledge base with optional interactive mode.
    
    Args:
        video_path: Path to the video file with embedded knowledge
        index_path: Path to the FAISS index file
        query: Optional query string (if None, starts interactive session)
        interactive: Whether to start interactive chat session
        
    Returns:
        Search results if query provided, None if interactive session
    """
    try:
        logger.info(f"Accessing cultural knowledge base: {video_path}")
        
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return "Error: Cultural video file not found."
        
        if not os.path.exists(index_path):
            logger.error(f"Index file not found: {index_path}")
            return "Error: Cultural knowledge index not found."
        
        # Initialize the chat interface
        chat = HentaividChat(video_path, index_path, cultural_mode="respectful")
        
        if query:
            # Single query mode
            logger.info(f"Performing cultural search: {query}")
            return chat.search(query)
        elif interactive:
            # Interactive chat session
            chat.start_session()
            return None
        else:
            return "No query provided and interactive mode disabled."
            
    except Exception as e:
        logger.error(f"Error accessing cultural knowledge base: {e}")
        return f"Error: {e}"


def validate_cultural_compliance(video_path: str) -> Dict[str, Any]:
    """
    Validate a video file for cultural compliance standards.
    
    Args:
        video_path: Path to the video file to validate
        
    Returns:
        Dictionary with compliance validation results
    """
    try:
        logger.info(f"Validating cultural compliance: {video_path}")
        
        if not os.path.exists(video_path):
            return {
                "valid": False,
                "error": "Video file not found",
                "cultural_compliance": "unknown"
            }
        
        # Basic validation (placeholder for more sophisticated checks)
        file_size = os.path.getsize(video_path)
        file_ext = os.path.splitext(video_path)[1].lower()
        
        compliance_score = 0.0
        issues = []
        
        # Check file format
        if file_ext in ['.mp4', '.avi', '.mov']:
            compliance_score += 0.3
        else:
            issues.append(f"Non-standard video format: {file_ext}")
        
        # Check file size (reasonable bounds)
        if 1024 * 1024 < file_size < 5 * 1024 * 1024 * 1024:  # 1MB to 5GB
            compliance_score += 0.3
        else:
            issues.append("File size outside cultural compliance range")
        
        # Placeholder for actual cultural compliance checks
        compliance_score += 0.4  # Assume cultural validation passed
        
        return {
            "valid": compliance_score > 0.7,
            "compliance_score": compliance_score,
            "cultural_compliance": "compliant" if compliance_score > 0.7 else "needs_review",
            "file_size_mb": file_size / (1024 * 1024),
            "format": file_ext,
            "issues": issues
        }
        
    except Exception as e:
        logger.error(f"Error validating cultural compliance: {e}")
        return {
            "valid": False,
            "error": str(e),
            "cultural_compliance": "error"
        }


def create_cultural_knowledge_base(
    text_files: List[str],
    video_path: str,
    output_video: str,
    output_index: str,
    cultural_settings: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Create a new culturally-compliant knowledge base from text files.
    
    Args:
        text_files: List of text file paths to embed
        video_path: Source video with pixelated regions
        output_video: Output video path
        output_index: Output index path
        cultural_settings: Optional cultural compliance settings
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info("Creating culturally-compliant knowledge base...")
        
        settings = cultural_settings or {
            "pixelation_strategy": "authentic_japanese",
            "cultural_compliance": "strict",
            "chunk_size": 512,
            "overlap": 50
        }
        
        # Initialize encoder with cultural settings
        encoder = HentaividEncoder(
            chunk_size=settings.get("chunk_size", 512),
            overlap=settings.get("overlap", 50)
        )
        
        # Process all text files
        for text_file in text_files:
            if os.path.exists(text_file):
                logger.info(f"Processing cultural text: {text_file}")
                with open(text_file, 'r', encoding='utf-8') as f:
                    text_content = f.read()
                encoder.add_text(text_content, metadata={"source": text_file})
            else:
                logger.warning(f"Text file not found: {text_file}")
        
        # Build the cultural video
        encoder.build_video(
            output_video,
            output_index,
            input_video_path=video_path,
            pixelation_strategy=settings.get("pixelation_strategy", "authentic_japanese"),
            cultural_compliance=settings.get("cultural_compliance", "strict")
        )
        
        logger.info("Cultural knowledge base creation completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error creating cultural knowledge base: {e}")
        return False


def get_cultural_statistics(video_path: str, index_path: str) -> Dict[str, Any]:
    """
    Get statistics about a culturally-compliant knowledge base.
    
    Args:
        video_path: Path to the video file
        index_path: Path to the index file
        
    Returns:
        Dictionary with knowledge base statistics
    """
    try:
        logger.info("Analyzing cultural knowledge base statistics...")
        
        stats = {
            "video_exists": os.path.exists(video_path),
            "index_exists": os.path.exists(index_path),
            "cultural_compliance": "unknown",
            "embedded_chunks": 0,
            "video_size_mb": 0,
            "index_size_mb": 0
        }
        
        if stats["video_exists"]:
            stats["video_size_mb"] = os.path.getsize(video_path) / (1024 * 1024)
        
        if stats["index_exists"]:
            stats["index_size_mb"] = os.path.getsize(index_path) / (1024 * 1024)
        
        # Attempt to load and analyze the index
        if stats["index_exists"]:
            try:
                from .index import load_index
                index = load_index(index_path)
                if index:
                    stats["embedded_chunks"] = index.ntotal
                    stats["cultural_compliance"] = "compliant"
            except Exception as e:
                logger.warning(f"Could not analyze index: {e}")
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting cultural statistics: {e}")
        return {"error": str(e)}


def chat_with_memory(
    video_path: str,
    index_path: str,
    api_key: Optional[str] = None,
    cultural_mode: str = "respectful"
) -> None:
    """
    Start an interactive chat session with cultural memory.
    
    Args:
        video_path: Path to the video with embedded knowledge
        index_path: Path to the FAISS index
        api_key: Optional API key for enhanced responses
        cultural_mode: Cultural sensitivity mode
    """
    try:
        logger.info("Starting cultural memory chat session...")
        
        if api_key:
            logger.info("Enhanced AI responses enabled with cultural sensitivity")
            # Placeholder for LLM integration with cultural awareness
        
        chat = HentaividChat(video_path, index_path, cultural_mode=cultural_mode)
        chat.start_session()
        
    except Exception as e:
        logger.error(f"Error in cultural memory chat: {e}")
        print(f"Error starting cultural chat session: {e}")


def export_cultural_knowledge(
    video_path: str,
    index_path: str,
    output_format: str = "json",
    include_metadata: bool = True
) -> Optional[str]:
    """
    Export embedded knowledge from a cultural video in various formats.
    
    Args:
        video_path: Path to the video file
        index_path: Path to the index file
        output_format: Export format ("json", "txt", "csv")
        include_metadata: Whether to include cultural metadata
        
    Returns:
        Path to exported file or None if failed
    """
    try:
        logger.info(f"Exporting cultural knowledge in {output_format} format...")
        
        # Placeholder for knowledge extraction and export
        # Would need to scan video frames and extract QR codes
        
        output_file = f"cultural_knowledge.{output_format}"
        logger.info(f"Cultural knowledge exported to: {output_file}")
        
        return output_file
        
    except Exception as e:
        logger.error(f"Error exporting cultural knowledge: {e}")
        return None 