"""
Hentaivid: Revolutionary RAG-compatible video storage format

This package provides culturally-compliant video-based knowledge storage
that embeds text chunks into QR codes hidden inside pixelated regions.
"""

__version__ = "0.1.0"
__author__ = "Hentaivid Development Team"
__email__ = "contact@hentaivid.dev"

from .encoder import HentaividEncoder
from .chat import HentaividChat, HentaividRetriever
from .interactive import HentaividInteractive
from .detector import AdvancedPixelationDetector
from .utils import search_knowledge_base

__all__ = [
    "HentaividEncoder",
    "HentaividChat", 
    "HentaividRetriever",
    "HentaividInteractive",
    "AdvancedPixelationDetector",
    "search_knowledge_base",
] 