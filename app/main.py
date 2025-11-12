from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from datetime import datetime
import numpy as np
import asyncio
from typing import Dict, Optional
import json
import base64

from app.nova import DeepgramNovaSTT
from app.llm import RemoteGPTLLM
from app.tts import ElevenLabsTTS
from app.productsearch import ProductSearch
from app.database import customer_db, CustomerDatabase

# ============================================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================================

# Load .env file
load_dotenv()

# Debug: Check if API keys are loaded
deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")
elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")

print(f"🔑 Deepgram API Key loaded: {'Yes' if deepgram_api_key else 'No'}")
print(f"🔑 ElevenLabs API Key loaded: {'Yes' if elevenlabs_api_key else 'No'}")

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class TextToSpeechRequest(BaseModel):
    text: str

class ChatRequest(BaseModel):
    message: str
    include_tts: bool = False

class ProductSearchRequest(BaseModel):
    query: str
    top_k: int = 3

# ============================================================================
# FASTAPI APP INITIALIZATION
# ============================================================================

app = FastAPI(
    title="Real-Time Voice Assistant API",
    description="Deepgram Nova 3 Streaming STT + gpt oss-20b + ElevenLabs TTS + Product Search",
    version="4.1.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create uploads directory
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ============================================================================
# GLOBAL MODEL INITIALIZATION
# ============================================================================

print("\n" + "="*80)
print("🚀 INITIALIZING VOICE ASSISTANT BACKEND")
print("="*80 + "\n")

# Global instances
nova_stt = None
tts_engine = None
RemoteGPT = None
product_search = None
customer_db = None

async def initialize_services():
    """Initialize all services"""
    global nova_stt, tts_engine, RemoteGPT, product_search, customer_db
    
    try:
        print("🗄️ Initializing Database...")
        customer_db = CustomerDatabase()
        db_success = await customer_db.initialize()
        if db_success:
            print("✅ Database initialized successfully")
        else:
            print("⚠️ Database initialization failed - continuing without database")
        # Initialize Product Search
        print("🔍 Initializing Product Search...")
        product_search = ProductSearch(product_file="data/product.json")
        await product_search.initialize()
        print("✅ Product Search initialized successfully")
        
        # Initialize Deepgram Nova STT
        print("📝 Initializing Deepgram Nova...")
        nova_stt = DeepgramNovaSTT(
            api_key=os.getenv("DEEPGRAM_API_KEY"),
            sample_rate=16000,
            language=None,  # Auto-detect
            model="nova-3",
            verbose=False,
            enable_multilingual=True, 
        )
        print("✅ Deepgram Nova initialized successfully")
        
        # Initialize ElevenLabs TTS
        print("🔊 Initializing ElevenLabs TTS...")
        try:
            tts_engine = ElevenLabsTTS(
                voice_id="iP95p4xoKVk53GoZ742B",
                model_id="eleven_multilingual_v2",
                output_format="mp3_44100_128"
            )
            print("✅ ElevenLabs TTS initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize ElevenLabs TTS: {e}")
            tts_engine = None
        
        # Initialize Remote GPT LLM with Product Search
        print("\n🧠 Initializing OpenAI LLM with Product Search...")
        try:
            RemoteGPT = RemoteGPTLLM(
                api_key=os.getenv("OPENAI_API_KEY"),
                model="gpt-4o-mini",  # or "gpt-4" if you have access
                product_search=product_search,
                customer_db=customer_db
            )
            print("✅ OpenAI LLM with Product Search initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize OpenAI LLM: {e}")
            
            # Fallback to a simple LLM
            class FallbackLLM:
                async def generate_response(self, text):
                    return {
                        'text': f"I received: '{text}'. The AI system is being configured.",
                        'timestamp': datetime.now().isoformat(),
                        'model': 'fallback',
                        'success': False,
                    }
                def clear_conversation_history(self): pass
                def get_conversation_stats(self): return {}
            RemoteGPT = FallbackLLM()
            
        print("\n🎉 All services initialized successfully!")
        
    except Exception as e:
        print(f"❌ Service initialization failed: {e}")
        raise

# ============================================================================
# CONNECTION MANAGER FOR WEBSOCKETS
# ============================================================================

class ConnectionManager:
    """Manage WebSocket connections with Deepgram streaming"""
    
    def __init__(self):
        self.active_connections: Dict[WebSocket, Dict] = {}
    
    async def connect(self, websocket: WebSocket):
        """Accept new WebSocket connection"""
        await websocket.accept()
        
        self.active_connections[websocket] = {
            'connected_at': datetime.now(),
            'chunks_received': 0,
            'utterances_processed': 0,
            'deepgram_connection': None
        }
        
        print(f"✓ New WebSocket connection (total: {len(self.active_connections)})")
    
    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        if websocket in self.active_connections:
            connection_info = self.active_connections[websocket]
            print(f"✗ WebSocket disconnected:")
            print(f"  - Chunks received: {connection_info['chunks_received']}")
            print(f"  - Utterances processed: {connection_info['utterances_processed']}")
            print(f"  - Duration: {datetime.now() - connection_info['connected_at']}")
            del self.active_connections[websocket]
    
    def increment_chunks(self, websocket: WebSocket):
        """Increment chunk counter"""
        if websocket in self.active_connections:
            self.active_connections[websocket]['chunks_received'] += 1
    
    def increment_utterances(self, websocket: WebSocket):
        """Increment utterance counter"""
        if websocket in self.active_connections:
            self.active_connections[websocket]['utterances_processed'] += 1

manager = ConnectionManager()

# ============================================================================
# CALLBACK FUNCTIONS FOR WEBSOCKETS
# ============================================================================

async def handle_assistant_transcription_callback(transcription_data: Dict, client_websocket: WebSocket):
    """Handle transcriptions and generate LLM responses with TTS - ROBUST INTERRUPTION"""
    try:
        # Check if websocket is still open before sending
        if client_websocket.client_state.name != "CONNECTED":
            print("⚠️ WebSocket not connected, skipping transcription send")
            return
        
        # Send transcription to client
        try:
            await client_websocket.send_json(transcription_data)
        except Exception as e:
            print(f"⚠️ Failed to send transcription: {e}")
            return
        
        if transcription_data.get('is_final', False):
            manager.increment_utterances(client_websocket)
            final_text = transcription_data['text']
            print(f"🎯 Final transcription: '{final_text}'")
            
            # Only generate LLM response for meaningful inputs
            if len(final_text.strip()) > 1:
                print("🧠 Generating AI response with product search...")
                
                # ROBUST TTS INTERRUPTION - Stop immediately and wait for completion
                if tts_engine:
                    print("🛑 FORCE STOPPING ALL TTS PLAYBACK FOR NEW QUESTION")
                    # Use synchronous stop with timeout
                    try:
                        await asyncio.wait_for(
                            asyncio.get_event_loop().run_in_executor(
                                None, 
                                tts_engine.stop_current_audio
                            ),
                            timeout=2.0
                        )
                        print("✅ TTS playback stopped successfully")
                    except asyncio.TimeoutError:
                        print("⚠️ TTS stop timed out, continuing anyway")
                    except Exception as e:
                        print(f"⚠️ Error stopping TTS: {e}")
                
                try:
                    # Generate LLM response with timeout
                    llm_response = await asyncio.wait_for(
                        RemoteGPT.generate_response(final_text),
                        timeout=15.0
                    )
                    
                    # Check websocket state before sending response
                    if client_websocket.client_state.name != "CONNECTED":
                        print("⚠️ WebSocket disconnected during LLM generation")
                        return
                    
                    # Send LLM response to client
                    response_data = {
                        "type": "llm_response",
                        "text": llm_response['text'],
                        "timestamp": llm_response['timestamp'],
                        "model": llm_response['model'],
                        "success": llm_response['success'],
                        "products_used": llm_response.get('products_used_in_context', 0)
                    }
                    
                    try:
                        await client_websocket.send_json(response_data)
                        print(f"🤖 AI Response: '{llm_response['text']}'")
                        print(f"📦 Products used in context: {llm_response.get('products_used_in_context', 0)}")
                    except Exception as e:
                        print(f"⚠️ Failed to send LLM response: {e}")
                        return
                    
                    # Generate TTS audio if TTS is available and response is successful
                    if tts_engine and llm_response.get('success'):
                        print("🔊 Generating TTS audio with interruption...")
                        
                        # Run TTS generation in background
                        asyncio.create_task(
                            generate_and_send_tts_with_interrupt(
                                llm_response['text'], 
                                client_websocket
                            )
                        )
                    
                except asyncio.TimeoutError:
                    print("❌ LLM response timeout")
                    if client_websocket.client_state.name == "CONNECTED":
                        try:
                            timeout_response = {
                                "type": "llm_response",
                                "text": "I'm taking too long to respond. Please try again.",
                                "timestamp": datetime.now().isoformat(),
                                "model": "timeout",
                                "success": False
                            }
                            await client_websocket.send_json(timeout_response)
                        except:
                            pass
                    
                except Exception as e:
                    print(f"❌ LLM generation error: {e}")
                    if client_websocket.client_state.name == "CONNECTED":
                        try:
                            error_response = {
                                "type": "llm_response",
                                "text": "I encountered an error while generating a response. Please try again.",
                                "timestamp": datetime.now().isoformat(),
                                "model": "error",
                                "success": False
                            }
                            await client_websocket.send_json(error_response)
                        except:
                            pass
    
    except Exception as e:
        print(f"❌ Error in assistant callback: {e}")

async def generate_and_send_tts_with_interrupt(text: str, client_websocket: WebSocket):
    """
    Generate TTS audio with proper interruption handling
    """
    try:
        # Check websocket is still connected
        if client_websocket.client_state.name != "CONNECTED":
            print("⚠️ WebSocket not connected, skipping TTS generation")
            return
        
        print(f"🔊 Converting text to speech: '{text[:50]}...'")
        
        # Generate TTS with force interruption
        try:
            audio_data = await asyncio.wait_for(
                tts_engine.text_to_speech(text, play_audio=False, interrupt_current=True),
                timeout=10.0
            )
        except asyncio.TimeoutError:
            print("⏰ TTS generation timed out")
            return
        
        if audio_data:
            # Double-check websocket is still connected before sending
            if client_websocket.client_state.name != "CONNECTED":
                print("⚠️ WebSocket disconnected during TTS generation, discarding audio")
                return
            
            # Convert audio to base64
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            
            # Send TTS audio to client with interrupt flag
            try:
                await client_websocket.send_json({
                    "type": "tts_audio",
                    "audio_data": audio_base64,
                    "text": text,
                    "timestamp": datetime.now().isoformat(),
                    "interrupt_previous": True
                })
                print(f"✅ TTS audio sent to client ({len(audio_data)} bytes)")
                
            except Exception as send_error:
                print(f"⚠️ Failed to send TTS audio: {send_error}")
        else:
            print("⚠️ TTS audio generation failed - no data returned")
            
    except Exception as e:
        print(f"❌ TTS generation error: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# IMPROVED: WebSocket Endpoint with Better Error Handling
# ============================================================================

@app.websocket("/ws/voice-assistant")
async def websocket_voice_assistant(websocket: WebSocket):
    """
    WebSocket endpoint for full voice assistant with STT + LLM + TTS + Product Search
    IMPROVED: Better connection management and error handling
    """
    await manager.connect(websocket)
    
    deepgram_connection = None
    receive_task = None
    
    try:
        print("\n" + "="*80)
        print("🎯 VOICE ASSISTANT SESSION STARTED")
        print("="*80)
        
        # Connect to Deepgram
        deepgram_connection = await nova_stt.create_streaming_connection(
            websocket,
            handle_assistant_transcription_callback
        )
        
        manager.active_connections[websocket]['deepgram_connection'] = deepgram_connection
        
        # Send welcome message
        await websocket.send_json({
            "type": "assistant_connected",
            "message": "Voice assistant ready with TTS and Product Search",
            "config": {
                "stt": "deepgram-nova-3",
                "llm": "gpt-4o-mini",
                "tts": "elevenlabs",
                "product_search": "sentence-transformers",
                "features": ["live_transcription", "llm_responses", "text_to_speech", "product_search"]
            }
        })
        
        print("✅ Voice assistant connection established")
        
        # Start receiving transcriptions from Deepgram
        receive_task = asyncio.create_task(
            deepgram_connection['receive_transcriptions']()
        )
        
        # Main loop to receive audio from client and send to Deepgram
        while True:
            try:
                # Check if connection is still alive
                if websocket.client_state.name != "CONNECTED":
                    print("⚠️ WebSocket no longer connected, breaking loop")
                    break
                
                # Receive audio with timeout to allow periodic connection checks
                data = await asyncio.wait_for(
                    websocket.receive_bytes(),
                    timeout=30.0  # 30 second timeout
                )
                
                manager.increment_chunks(websocket)
                
                # Send audio to Deepgram
                await deepgram_connection['send_audio'](data)
                
                # Small delay to prevent overwhelming the connection
                await asyncio.sleep(0.01)
                
            except asyncio.TimeoutError:
                # Timeout is normal - just means no data was received
                # Send a ping to keep connection alive
                try:
                    await websocket.send_json({
                        "type": "ping",
                        "timestamp": datetime.now().isoformat()
                    })
                except:
                    print("⚠️ Failed to send ping, connection may be dead")
                    break
                continue
                
            except WebSocketDisconnect:
                print("🔌 Client disconnected")
                break
                
            except Exception as e:
                print(f"❌ Error in main loop: {e}")
                break
    
    except WebSocketDisconnect:
        print("🔌 Voice assistant client disconnected")
    
    except Exception as e:
        print(f"❌ Voice assistant error: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to send error to client if still connected
        try:
            if websocket.client_state.name == "CONNECTED":
                await websocket.send_json({
                    "type": "error",
                    "message": str(e)
                })
        except:
            pass
    
    finally:
        # Cleanup
        print("🧹 Cleaning up voice assistant connection...")
        
        # Cancel receive task if running
        if receive_task and not receive_task.done():
            receive_task.cancel()
            try:
                await asyncio.wait_for(receive_task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        
        # Close Deepgram connection
        if deepgram_connection:
            try:
                await asyncio.wait_for(
                    deepgram_connection['close'](),
                    timeout=5.0
                )
            except asyncio.TimeoutError:
                print("⏰ Deepgram close timed out")
            except Exception as e:
                print(f"⚠️ Error closing Deepgram: {e}")
        
        # Disconnect from manager
        manager.disconnect(websocket)
        
        print("="*80)
        print("🎯 VOICE ASSISTANT SESSION ENDED")
        print("="*80 + "\n")
        
# ============================================================================
# REST API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with system status"""
    return {
        "message": "Deepgram Nova + OpenAI LLM + ElevenLabs TTS + Product Search Voice Assistant",
        "status": "online",
        "version": "4.1.0",
        "features": {
            "deepgram_nova": True,
            "RemoteGPT": True,
            "elevenlabs_tts": tts_engine is not None,
            "product_search": product_search is not None,
            "live_streaming": True,
            "text_to_speech": True
        },
        "models": {
            "stt_model": "nova-3",
            "llm_model": "gpt-4o-mini",
            "tts_model": "eleven_multilingual_v2",
            "embedding_model": "all-MiniLM-L6-v2"
        }
    }

@app.post("/text-to-speech")
async def text_to_speech_endpoint(request: TextToSpeechRequest):
    """Convert text to speech using ElevenLabs"""
    try:
        text = request.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        if not tts_engine:
            raise HTTPException(status_code=500, detail="TTS engine not available")
        
        print(f"🔊 TTS request: '{text}'")
        
        # Generate TTS audio
        audio_data = await tts_engine.text_to_speech(text, play_audio=False, interrupt_current=True)
        
        if audio_data:
            # Convert to base64 for JSON response
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            
            return {
                "status": "success",
                "text": text,
                "audio_data": audio_base64,
                "audio_format": "mp3_44100_128",
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=500, detail="TTS conversion failed")
            
    except Exception as e:
        print(f"❌ TTS endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"TTS error: {str(e)}")

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """Chat endpoint with optional TTS"""
    try:
        user_message = request.message.strip()
        include_tts = request.include_tts
        
        if not user_message:
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        print(f"💬 Chat request: '{user_message}'")
        
        # Generate LLM response
        llm_response = await asyncio.wait_for(
            RemoteGPT.generate_response(user_message),
            timeout=15.0
        )
        
        response_data = {
            "status": "success",
            "user_message": user_message,
            "assistant_response": llm_response['text'],
            "timestamp": llm_response['timestamp'],
            "model": llm_response['model'],
            "success": llm_response['success'],
            "products_used": llm_response.get('products_used_in_context', 0)
        }
        
        # Include TTS if requested and available
        if include_tts and tts_engine and llm_response.get('success'):
            audio_data = await tts_engine.text_to_speech(
                llm_response['text'], 
                play_audio=False,
                interrupt_current=True
            )
            if audio_data:
                response_data["audio_data"] = base64.b64encode(audio_data).decode('utf-8')
        
        return response_data
        
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Response timeout")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@app.post("/search-products")
async def search_products_endpoint(request: ProductSearchRequest):
    """Search products using semantic similarity"""
    try:
        if not product_search:
            raise HTTPException(status_code=500, detail="Product search not available")
        
        query = request.query.strip()
        top_k = request.top_k
        
        if not query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        print(f"🔍 Product search: '{query}' (top_k: {top_k})")
        
        # Search for products
        results = await product_search.search_products(query, top_k=top_k)
        
        return {
            "status": "success",
            "query": query,
            "top_k": top_k,
            "results_found": len(results),
            "products": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

@app.get("/products")
async def get_all_products():
    """Get all products"""
    try:
        if not product_search:
            raise HTTPException(status_code=500, detail="Product search not available")
        
        return {
            "status": "success",
            "products": product_search.products,
            "total_products": len(product_search.products),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting products: {str(e)}")

@app.get("/product/{product_id}")
async def get_product_by_id(product_id: str):
    """Get product by ID"""
    try:
        if not product_search:
            raise HTTPException(status_code=500, detail="Product search not available")
        
        product = await product_search.get_product_by_id(product_id)
        
        if product:
            return {
                "status": "success",
                "product": product
            }
        else:
            raise HTTPException(status_code=404, detail=f"Product {product_id} not found")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting product: {str(e)}")

@app.get("/tts-voices")
async def get_tts_voices():
    """Get available TTS voices and configuration"""
    if not tts_engine:
        return {"status": "error", "message": "TTS not available"}
    
    voice_info = tts_engine.get_voice_info()
    return {
        "status": "success",
        "tts_available": True,
        "voice_config": voice_info
    }

@app.get("/search-stats")
async def get_search_stats():
    """Get product search statistics"""
    if not product_search:
        return {"status": "error", "message": "Product search not available"}
    
    return {
        "status": "success",
        "search_stats": product_search.get_stats()
    }

@app.get("/llm-stats")
async def get_llm_stats():
    """Get LLM statistics"""
    if not RemoteGPT:
        return {"status": "error", "message": "LLM not available"}
    
    return {
        "status": "success",
        "llm_stats": RemoteGPT.get_system_stats()
    }

@app.get("/debug/database")
async def debug_database():
    """Debug database connection"""
    if not customer_db:
        return {"status": "error", "message": "Customer DB not initialized"}
    
    return {
        "database_initialized": customer_db.is_connected,
        "database_url": customer_db.database_url if customer_db.database_url else "Not set",
        "llm_has_database": hasattr(RemoteGPT, 'customer_db') and RemoteGPT.customer_db is not None
    }

@app.get("/debug/queries")
async def debug_queries():
    """Get recent queries from database"""
    if not customer_db or not customer_db.is_connected:
        return {"status": "error", "message": "Database not connected"}
    
    try:
        async with customer_db.connection_pool.acquire() as conn:
            rows = await conn.fetch('''
                SELECT query_id, customer_name, customer_email, product_requested, 
                       full_query_text, timestamp 
                FROM customer_queries 
                ORDER BY timestamp DESC 
                LIMIT 10
            ''')
            
            queries = []
            for row in rows:
                queries.append(dict(row))
            
            return {
                "status": "success",
                "total_queries": len(queries),
                "queries": queries
            }
            
    except Exception as e:
        return {"status": "error", "message": f"Database error: {str(e)}"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "deepgram_configured": bool(nova_stt and nova_stt.api_key),
        "ollama_configured": hasattr(RemoteGPT, 'generate_response'),
        "tts_configured": tts_engine is not None,
        "product_search_configured": product_search is not None,
        "active_connections": len(manager.active_connections),
        "timestamp": datetime.now().isoformat()
    }

# ============================================================================
# STARTUP EVENT
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    await initialize_services()
    
    print("\n" + "="*80)
    print("🚀 VOICE ASSISTANT SERVER STARTED SUCCESSFULLY")
    print("="*80)
    print(f"📡 Server running at: http://localhost:8004")
    print(f"📚 API Documentation: http://localhost:8004/docs")
    print(f"🔊 TTS Status: {'✅ Enabled' if tts_engine else '❌ Disabled'}")
    print(f"🔍 Product Search: {'✅ Enabled' if product_search else '❌ Disabled'}")
    print("="*80 + "\n")

# shudown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown"""
    print("🧹 Shutting down services...")
    
    if tts_engine:
        # Synchronous cleanup for TTS
        try:
            await asyncio.get_event_loop().run_in_executor(None, tts_engine.cleanup)
        except Exception as e:
            print(f"⚠️ Error during TTS cleanup: {e}")
    
    print("✅ Services shut down successfully")