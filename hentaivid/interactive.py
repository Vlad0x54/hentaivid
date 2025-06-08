"""
Hentaivid Interactive Interface - Cultural Web Application

This module provides a web-based interface for interacting with culturally-compliant
video knowledge bases through a user-friendly Gradio interface.
"""

from typing import Optional
from loguru import logger

from .chat import HentaividRetriever


class HentaividInteractive:
    """
    Web-based interactive interface for cultural knowledge exploration.
    
    Provides a Gradio-powered web interface for searching and exploring
    embedded knowledge with cultural sensitivity and Japanese standards.
    """
    
    def __init__(
        self,
        video_path: str,
        index_path: str,
        port: int = 7860,
        cultural_theme: str = "modern_japanese"
    ):
        """
        Initialize the interactive cultural interface.
        
        Args:
            video_path: Path to the video with embedded knowledge
            index_path: Path to the FAISS index
            port: Port for the web interface
            cultural_theme: UI theme reflecting cultural standards
        """
        self.video_path = video_path
        self.index_path = index_path
        self.port = port
        self.cultural_theme = cultural_theme
        
        # Initialize the retriever
        self.retriever = HentaividRetriever(video_path, index_path)
        
        logger.info(f"HentaividInteractive initialized with {cultural_theme} theme")
    
    def run(self, share: bool = False):
        """
        Launch the culturally-aware web interface.
        
        Args:
            share: Whether to create a public sharing link
        """
        try:
            import gradio as gr
            
            logger.info("Starting cultural web interface...")
            
            def search_interface(query: str, num_results: int = 5) -> str:
                """Interface function for web-based search."""
                if not query.strip():
                    return "Please enter a culturally-appropriate query."
                
                results = self.retriever.search(query, top_k=num_results)
                
                if not results:
                    return "No culturally-relevant results found."
                
                formatted_output = f"🎌 Cultural Search Results for: '{query}'\n\n"
                
                for i, (chunk_text, score, timestamp) in enumerate(results, 1):
                    formatted_output += f"Result {i} [Relevance: {score:.3f}]\n"
                    if timestamp:
                        formatted_output += f"Cultural Context at: {timestamp:.1f}s\n"
                    formatted_output += f"{chunk_text}\n\n"
                    formatted_output += "─" * 50 + "\n\n"
                
                return formatted_output
            
            def get_context_interface(query: str, max_tokens: int = 2000) -> str:
                """Interface function for context retrieval."""
                if not query.strip():
                    return "Please enter a query for cultural context."
                
                context = self.retriever.get_context(query, max_tokens=max_tokens)
                return f"🎌 Cultural Context for: '{query}'\n\n{context}"
            
            # Create the Gradio interface with cultural styling
            with gr.Blocks(
                title="🎌 Hentaivid Cultural Knowledge Explorer",
                theme=gr.themes.Soft(),
                css=".gradio-container {background: linear-gradient(45deg, #ff9a9e, #fecfef);}"
            ) as interface:
                
                gr.HTML("""
                <div style="text-align: center; padding: 20px;">
                    <h1>🎌 Hentaivid Cultural Knowledge Explorer</h1>
                    <p><em>Culturally-compliant semantic search for embedded video knowledge</em></p>
                    <p>Respectfully exploring knowledge embedded within Japanese cultural standards</p>
                </div>
                """)
                
                with gr.Tab("🔍 Cultural Search"):
                    with gr.Row():
                        with gr.Column():
                            search_input = gr.Textbox(
                                label="Cultural Query",
                                placeholder="Enter your culturally-sensitive search query...",
                                lines=2
                            )
                            num_results = gr.Slider(
                                minimum=1,
                                maximum=10,
                                value=5,
                                step=1,
                                label="Number of Results"
                            )
                            search_btn = gr.Button("🎌 Search Cultural Knowledge", variant="primary")
                        
                        with gr.Column():
                            search_output = gr.Textbox(
                                label="Cultural Search Results",
                                lines=20,
                                max_lines=30
                            )
                
                with gr.Tab("📚 Cultural Context"):
                    with gr.Row():
                        with gr.Column():
                            context_input = gr.Textbox(
                                label="Context Query",
                                placeholder="Enter query for cultural context...",
                                lines=2
                            )
                            max_tokens = gr.Slider(
                                minimum=500,
                                maximum=5000,
                                value=2000,
                                step=100,
                                label="Maximum Context Tokens"
                            )
                            context_btn = gr.Button("🎌 Get Cultural Context", variant="primary")
                        
                        with gr.Column():
                            context_output = gr.Textbox(
                                label="Cultural Context",
                                lines=20,
                                max_lines=30
                            )
                
                with gr.Tab("ℹ️ Cultural Information"):
                    gr.HTML("""
                    <div style="padding: 20px;">
                        <h2>🎌 About Hentaivid Cultural Standards</h2>
                        <p>This system respects Japanese cultural standards while providing:</p>
                        <ul>
                            <li>✅ Culturally-compliant video processing</li>
                            <li>✅ Respectful knowledge embedding techniques</li>
                            <li>✅ Semantic search with cultural sensitivity</li>
                            <li>✅ Zero-database architecture for privacy</li>
                        </ul>
                        
                        <h3>📋 Cultural Guidelines</h3>
                        <p>All searches and interactions maintain the highest standards of:</p>
                        <ul>
                            <li>Cultural sensitivity and respect</li>
                            <li>Japanese censorship compliance</li>
                            <li>Appropriate content handling</li>
                            <li>Professional knowledge management</li>
                        </ul>
                    </div>
                    """)
                
                # Connect the interfaces
                search_btn.click(
                    search_interface,
                    inputs=[search_input, num_results],
                    outputs=search_output
                )
                
                context_btn.click(
                    get_context_interface,
                    inputs=[context_input, max_tokens],
                    outputs=context_output
                )
            
            # Launch the interface
            interface.launch(
                server_port=self.port,
                share=share,
                server_name="0.0.0.0" if share else "127.0.0.1"
            )
            
        except ImportError:
            logger.error("Gradio not installed. Install with: pip install gradio")
            print("🎌 Cultural Web Interface requires Gradio")
            print("Install with: pip install gradio")
        except Exception as e:
            logger.error(f"Error launching cultural interface: {e}")
            print(f"Error launching cultural web interface: {e}") 