"""
Hentaivid - Cultural Knowledge Embedding System

Enterprise-grade command-line interface for creating culturally-compliant
video-based knowledge storage with Japanese standards compliance.
"""

import argparse
import os
import sys
from loguru import logger

from hentaivid import HentaividEncoder, HentaividChat, HentaividInteractive
from hentaivid.utils import (
    search_knowledge_base,
    validate_cultural_compliance,
    create_cultural_knowledge_base,
    get_cultural_statistics
)


def configure_logging(verbose: bool = False):
    """Configure culturally-aware logging."""
    logger.remove()
    
    if verbose:
        logger.add(
            sys.stderr,
            format="🎌 <green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level="DEBUG"
        )
    else:
        logger.add(
            sys.stderr,
            format="🎌 <level>{level: <8}</level> | <level>{message}</level>",
            level="INFO"
        )


def create_knowledge_base(args):
    """Create a new culturally-compliant knowledge base."""
    logger.info("🎌 Creating culturally-compliant knowledge base...")
    
    if not os.path.exists(args.video_path):
        logger.error(f"Source video not found: {args.video_path}")
        return False
    
    if not os.path.exists(args.text_path):
        logger.error(f"Text file not found: {args.text_path}")
        return False
    
    # Validate cultural compliance of source video
    logger.info("Validating cultural compliance of source video...")
    compliance = validate_cultural_compliance(args.video_path)
    if not compliance.get("valid", False):
        logger.warning(f"Cultural compliance warning: {compliance.get('error', 'Unknown issue')}")
        if not args.force:
            logger.error("Use --force to proceed with non-compliant source material")
            return False
    
    # Initialize encoder with cultural settings
    encoder = HentaividEncoder(
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        embedding_model=args.embedding_model
    )
    
    # Configure cultural detection parameters
    if hasattr(args, 'detector') and args.detector == "advanced":
        encoder.set_detection_params(
            sensitivity="high",
            cultural_variant="modern_japanese",
            region_validation="strict"
        )
    
    # Add text content
    logger.info("Processing text with cultural sensitivity...")
    with open(args.text_path, 'r', encoding='utf-8') as f:
        text_content = f.read()
    
    encoder.add_text(text_content, metadata={"source": args.text_path})
    
    # Build culturally-compliant video
    cultural_settings = {
        "pixelation_strategy": getattr(args, 'pixelation_strategy', 'authentic_japanese'),
        "cultural_compliance": getattr(args, 'cultural_compliance', 'standard'),
        "fps": getattr(args, 'fps', 30)
    }
    
    encoder.build_video(
        args.output_path,
        args.index_path,
        input_video_path=args.video_path,
        **cultural_settings
    )
    
    logger.info("🎌 Cultural knowledge base created successfully!")
    return True


def search_knowledge_base_cli(args):
    """Search an existing cultural knowledge base."""
    logger.info("🎌 Accessing cultural knowledge base...")
    
    if args.query:
        # Single query mode
        result = search_knowledge_base(
            args.video_path,
            args.index_path,
            query=args.query,
            interactive=False
        )
        if result:
            print("\n" + "="*60)
            print("🎌 CULTURAL SEARCH RESULTS")
            print("="*60)
            print(result)
            print("="*60)
    else:
        # Interactive mode
        search_knowledge_base(args.video_path, args.index_path, interactive=True)


def launch_web_interface(args):
    """Launch the cultural web interface."""
    logger.info("🎌 Launching cultural web interface...")
    
    try:
        interactive = HentaividInteractive(
            args.video_path,
            args.index_path,
            port=args.port,
            cultural_theme="modern_japanese"
        )
        interactive.run(share=args.share)
    except Exception as e:
        logger.error(f"Error launching cultural interface: {e}")
        return False


def show_statistics(args):
    """Display cultural knowledge base statistics."""
    logger.info("🎌 Analyzing cultural knowledge base...")
    
    stats = get_cultural_statistics(args.video_path, args.index_path)
    
    print("\n" + "="*50)
    print("🎌 CULTURAL KNOWLEDGE BASE STATISTICS")
    print("="*50)
    print(f"Video File: {'✅ Found' if stats.get('video_exists') else '❌ Missing'}")
    print(f"Index File: {'✅ Found' if stats.get('index_exists') else '❌ Missing'}")
    print(f"Cultural Compliance: {stats.get('cultural_compliance', 'Unknown')}")
    print(f"Embedded Chunks: {stats.get('embedded_chunks', 0):,}")
    print(f"Video Size: {stats.get('video_size_mb', 0):.1f} MB")
    print(f"Index Size: {stats.get('index_size_mb', 0):.1f} MB")
    print("="*50)


def main():
    """Main entry point for the cultural knowledge system."""
    parser = argparse.ArgumentParser(
        description="🎌 Hentaivid: Culturally-compliant video-based knowledge storage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Cultural Examples:
  # Create knowledge base with cultural compliance
  python main.py create input.mp4 knowledge.txt --cultural-compliance strict
  
  # Search with cultural sensitivity
  python main.py search knowledge.mp4 knowledge.faiss --query "machine learning"
  
  # Launch cultural web interface
  python main.py web knowledge.mp4 knowledge.faiss --port 8080
  
  # Validate cultural compliance
  python main.py stats knowledge.mp4 knowledge.faiss
        """
    )
    
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose cultural logging")
    parser.add_argument("--force", action="store_true", help="Force processing despite cultural compliance warnings")
    
    subparsers = parser.add_subparsers(dest="command", help="Cultural operations")
    
    # Create command
    create_parser = subparsers.add_parser("create", help="Create cultural knowledge base")
    create_parser.add_argument("video_path", help="Source video with pixelated regions")
    create_parser.add_argument("text_path", help="Text file to embed with cultural sensitivity")
    create_parser.add_argument("--output-path", default="cultural_output.mp4", help="Output video path")
    create_parser.add_argument("--index-path", default="cultural_index.faiss", help="Output index path")
    create_parser.add_argument("--chunk-size", type=int, default=512, help="Cultural chunk size")
    create_parser.add_argument("--overlap", type=int, default=50, help="Cultural chunk overlap")
    create_parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2", help="Cultural embedding model")
    create_parser.add_argument("--detector", choices=["traditional", "advanced"], default="traditional", help="Cultural detection algorithm")
    create_parser.add_argument("--cultural-compliance", choices=["standard", "strict"], default="standard", help="Cultural compliance level")
    create_parser.add_argument("--pixelation-strategy", choices=["authentic_japanese", "high_density"], default="authentic_japanese", help="Cultural pixelation strategy")
    create_parser.add_argument("--fps", type=int, default=30, help="Cultural output FPS")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search cultural knowledge base")
    search_parser.add_argument("video_path", help="Cultural video file")
    search_parser.add_argument("index_path", help="Cultural index file")
    search_parser.add_argument("--query", "-q", help="Cultural search query")
    
    # Web interface command
    web_parser = subparsers.add_parser("web", help="Launch cultural web interface")
    web_parser.add_argument("video_path", help="Cultural video file")
    web_parser.add_argument("index_path", help="Cultural index file")
    web_parser.add_argument("--port", type=int, default=7860, help="Cultural web interface port")
    web_parser.add_argument("--share", action="store_true", help="Create public cultural sharing link")
    
    # Statistics command
    stats_parser = subparsers.add_parser("stats", help="Show cultural knowledge base statistics")
    stats_parser.add_argument("video_path", help="Cultural video file")
    stats_parser.add_argument("index_path", help="Cultural index file")
    
    args = parser.parse_args()
    
    # Configure cultural logging
    configure_logging(args.verbose)
    
    if not args.command:
        parser.print_help()
        return
    
    logger.info("🎌 Hentaivid Cultural Knowledge System v0.1.0")
    logger.info("Maintaining Japanese cultural standards with enterprise performance")
    
    try:
        if args.command == "create":
            success = create_knowledge_base(args)
            sys.exit(0 if success else 1)
        elif args.command == "search":
            search_knowledge_base_cli(args)
        elif args.command == "web":
            launch_web_interface(args)
        elif args.command == "stats":
            show_statistics(args)
        else:
            logger.error(f"Unknown cultural command: {args.command}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("🎌 Cultural operation interrupted. Sayonara!")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Cultural system error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 