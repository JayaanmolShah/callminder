from fastapi import FastAPI, HTTPException,WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import spacy
from sentence_transformers import SentenceTransformer
from datetime import datetime
import google.generativeai as genai
from pydantic import BaseModel
from config import GEMINI_API_KEY, SUPABASE_URL, SUPABASE_KEY
import sounddevice as sd
import speech_recognition as sr
import queue
import time
from threading import Thread, Event
import numpy as np
import pyttsx4
import asyncio
from typing import List, Optional,Dict, Any
import re

END_CALL_PHRASES = {
    "end call", "end the call", "goodbye", "bye", "quit", "stop", "hang up", 
    "end conversation", "that's all", "thank you bye", "thanks bye","stop the call","leave me alone"
}
# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Modify for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

genai.configure(api_key=GEMINI_API_KEY)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # Sentence-transformer model
faiss_index = None
audio_queue = queue.Queue()  # Queue for audio frames
transcriptions = []          # List to store transcription results
recognizer = sr.Recognizer()
stop_event = Event()  # Add an event to control the recording thread
recording_thread = None
tts_engine = pyttsx4.init()

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.speaking_event = asyncio.Event()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                print(f"Error broadcasting message: {e}")

# Initialize connection manager
manager = ConnectionManager()

# Pydantic models
class UserInput(BaseModel):
    user_input: str

class ConversationStatus(BaseModel):
    is_active: bool
    lead_id: Optional[str]
    timestamp: datetime

def datetime_serializer(obj):
    """Custom serializer for datetime objects"""
    if isinstance(obj, datetime):
        return obj.isoformat()  # Convert datetime to ISO 8601 format
    raise TypeError("Type not serializable")

class AIAgent:
    def __init__(self):
        # Initialize spaCy
        self.nlp = spacy.load("en_core_web_sm")

        # States and intents
        self.current_state = "greeting"  # Possible states: greeting, info_collection, qualification, pitching, closing
        self.last_question_asked = None
        self.pending_entities = ["name", "company","requirements", "budget", "timeline"]
        self.current_intent = None  # Detected user intent

        # Lead data
        self.current_lead = {
            "name": None,
            "company": None,
            "phone": None,
            "email": None,
            "requirements": None,
            "budget": None,
            "timeline": None,
            "qualified": False,
            "meeting_scheduled": False,
            "meeting_date": None,
            "meeting_time": None
        }

        # Conversation history
        self.conversation_history = {
            "lead_id": None,
            "start_time": None,
            "messages": [],
            "lead_info": self.current_lead
        }

        # Services offered
        self.services = {
            "digital marketing": {
                "description": "Complete digital marketing solutions including SEO, PPC, and social media management",
                "starting_price": 1500,
                "minimum_contract": "3 months"
            },
            "web development": {
                "description": "Custom website development and e-commerce solutions",
                "starting_price": 5000,
                "minimum_contract": "1 month"
            },
            "consulting": {
                "description": "Business growth and digital transformation consulting",
                "starting_price": 2000,
                "minimum_contract": "1 month"
            }
        }

        # Initialize Gemini
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel("gemini-pro")

        self.system_prompt = """You are an AI sales agent for Toshal Infotech. 
- Progress naturally through conversation states.
- Request only one piece of information per response.
- If users don’t provide info, move on politely without badgering them.
- Seamlessly switch between states as appropriate based on the user's responses.
- Always update records when new information is provided.
- Respond in under 500 characters with a professional and cheerful tone.
- Refer to past interactions when relevant."""

    def update_conversation(self, speaker: str, message: str):
        """Add a message to the conversation history."""
        self.conversation_history["messages"].append({
            "speaker": speaker,
            "message": message,
            "timestamp": datetime.now().isoformat()
        })

    def detect_intent(self, user_input: str) -> str:
        """Detect user intent from input."""
        user_input = user_input.lower()
        if any(keyword in user_input for keyword in ["price", "cost", "services"]):
            return "ask_about_services"
        if any(keyword in user_input for keyword in ["schedule", "meeting", "call"]):
            return "schedule_meeting"
        if any(keyword in user_input for keyword in ["requirement", "need", "looking for"]):
            return "provide_info"
        if any(keyword in user_input for keyword in ["explain", "clarify", "help"]):
            return "clarify_doubts"
        if any(keyword in user_input for keyword in ["not comfortable", "prefer not"]):
            return "decline_info"
        return "unknown"

    def extract_entities(self, user_input: str):
        """Extract entities from user input."""
        doc = self.nlp(user_input)
        extracted = {}

        # Extract name
        if not self.current_lead["name"] and "name is" in user_input:
            extracted["name"] = user_input.split("name is")[-1].strip().title()

        # Extract company
        if not self.current_lead["company"] and "company is" in user_input:
            extracted["company"] = user_input.split("company is")[-1].strip().title()

        # Extract email
        for token in doc:
            if "@" in token.text and "." in token.text:
                extracted["email"] = token.text

        # Extract phone number
        phone_numbers = re.findall(r"\b\d{10}\b|\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", user_input)
        if phone_numbers:
            extracted["phone"] = phone_numbers[0]

        # Extract requirements
        if "need" in user_input or "looking for" in user_input:
            extracted["requirements"] = user_input

        # Update lead information and manage pending entities
        for key, value in extracted.items():
            if key in self.current_lead and not self.current_lead[key]:
                self.current_lead[key] = value
                if key in self.pending_entities:
                    self.pending_entities.remove(key)

        # Ensure that if an entity is provided, it is not asked again
        for key in self.pending_entities.copy():
            if self.current_lead[key] is not None:  # Check if the value is not None
                self.pending_entities.remove(key)

        return extracted


    def generate_gemini_prompt(self, user_input: str) -> str:
        """Generate a context-aware prompt for Gemini."""
        intent = self.current_intent or "general"
        state_context = f"Current State: {self.current_state}. Intent: {intent}."
        unresolved_entities = ", ".join(self.pending_entities) if self.pending_entities else "None"
        recent_messages = "\n".join([
            f"{msg['speaker']}: {msg['message']}" for msg in self.conversation_history["messages"][-5:]
        ])

        # Determine the next entity to request
        next_entity = None
        for entity in self.pending_entities:
            if not self.current_lead[entity]:
                next_entity = entity
                break

        return (
            f"{self.system_prompt}\n\n"
            f"{state_context}\n"
            f"Next Required Information: {next_entity}\n"
            f"Unresolved Entities: {unresolved_entities}\n\n"
            f"Conversation History:\n{recent_messages}\n\n"
            f"User: {user_input}\n\n"
            f"Assistant:"
        )

    def process_input(self, user_input: str) -> str:
        """Process user input, detect intent, and interact via Gemini."""
        try:
            # Update conversation and detect intent
            self.update_conversation("User", user_input)
            self.current_intent = self.detect_intent(user_input)

            # Handle user declining to provide information
            if self.current_intent == "decline_info":
                response = "That's perfectly fine. Let me know how else I can assist you!"
                self.update_conversation("Assistant", response)
                return response

            # Extract entities from input
            extracted_entities = self.extract_entities(user_input)

            # Generate a Gemini prompt based on the current state and intent
            prompt = self.generate_gemini_prompt(user_input)

            # Generate response using Gemini
            chat = self.model.start_chat(history=[])
            response = chat.send_message(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.8,
                    max_output_tokens=200,
                    top_p=0.9,
                    top_k=50
                )
            )
            ai_response = response.text.strip()

            # Update conversation
            self.update_conversation("Assistant", ai_response)

            # Transition logic: Move to the next state if appropriate
            if self.current_state == "info_collection" and not self.pending_entities:
                self.current_state = "qualification"
            elif self.current_state == "qualification":
                self.current_state = "pitching"
            elif self.current_state == "pitching":
                self.current_state = "closing"
            print(self.current_state," ",self.pending_entities)
            return ai_response
        except Exception as e:
            print(f"Error in process_input: {e}")
            return "I'm having some trouble right now. Could you try again?"


ai_agent = AIAgent()

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.speaking_event = asyncio.Event()
        self.transcription_task = None  # Store reference to transcription task

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def disconnect_all(self):
        """Disconnect all active WebSocket connections"""
        for connection in self.active_connections.copy():
            await connection.close()
        self.active_connections.clear()

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                print(f"Error broadcasting message: {e}")

    async def stop_transcription(self):
        """Stop the transcription task if it's running"""
        if self.transcription_task and not self.transcription_task.done():
            self.transcription_task.cancel()
            try:
                await self.transcription_task
            except asyncio.CancelledError:
                pass
            self.transcription_task = None


async def stop_conversation():
    try:
        # Set stop event
        stop_event.set()
        
        # Stop the transcription task and disconnect WebSocket connections
        await manager.stop_transcription()
        await manager.disconnect_all()
        
        # Wait for any pending audio processing
        await asyncio.sleep(1)
        
        # Save conversation
        conversation_id = await ai_agent.save_conversation()
        
        # Clean up resources
        transcriptions.clear()
        while not audio_queue.empty():
            try:
                audio_queue.get_nowait()
            except queue.Empty:
                break
        
        # Reset AI agent conversation history
        ai_agent.conversation_history = {
            "lead_id": None,
            "start_time": None,
            "messages": [],
            "lead_info": ai_agent.current_lead.copy()
        }
        
        # Reset stop event for future conversations
        stop_event.clear()
        
        return {
            "status": "success",
            "conversation_id": conversation_id,
            "message": "Conversation ended and saved"
        }
    except Exception as e:
        print(f"Error stopping conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def speak_text_async(text: str):
    try:
        manager.speaking_event.set()
        await manager.broadcast({
            "type": "speaking_status",
            "data": {"is_speaking": True}
        })
        
        tts_engine.say(text)
        tts_engine.runAndWait()
        
    finally:
        manager.speaking_event.clear()
        await manager.broadcast({
            "type": "speaking_status",
            "data": {"is_speaking": False}
        })

def audio_callback(indata, frames, time, status):
    if status:
        print(f"Audio status: {status}")
    audio_queue.put(indata.copy())

async def transcribe_audio():
    try:
        with sd.InputStream(channels=1, samplerate=16000, callback=audio_callback):
            while not stop_event.is_set():
                if manager.speaking_event.is_set():
                    await asyncio.sleep(0.1)
                    continue
                    
                audio_data = []
                start_time = time.time()
                
                while time.time() - start_time < 3 and not stop_event.is_set():
                    try:
                        chunk = audio_queue.get(timeout=0.1)
                        audio_data.append(chunk)
                    except queue.Empty:
                        continue

                if audio_data and not stop_event.is_set():
                    await process_audio_chunks(audio_data)
                    
    except asyncio.CancelledError:
        print("Transcription task cancelled")
    except Exception as e:
        print(f"Audio stream error: {e}")
    finally:
        print("Audio transcription ended")

async def process_audio_chunks(audio_data):
    try:
        if not audio_data:
            return
            
        audio = np.concatenate(audio_data)
        if np.max(np.abs(audio)) < 0.01:  # Check if audio is too quiet
            return
            
        audio_bytes = (audio * 32767).astype(np.int16).tobytes()
        
        audio_source = sr.AudioData(audio_bytes, 16000, 2)
        text = recognizer.recognize_google(audio_source)
        
        if text.strip():
            # Check if the user wants to end the call
            if any(phrase in text.lower() for phrase in END_CALL_PHRASES):
                # Send farewell message before ending
                farewell = "Thank you for speaking with Toshal Infotech. Have a great day! Goodbye."
                await manager.broadcast({
                    "type": "ai_response",
                    "data": {"response": farewell}
                })
                await speak_text_async(farewell)
                
                # Call stop_conversation endpoint
                await stop_conversation()
                return

            transcription = {
                "timestamp": time.time(),
                "text": text,
                "response": None
            }
            transcriptions.append(transcription)
            
            await manager.broadcast({
                "type": "user_input",
                "data": {"text": text}
            })
            
            ai_response = ai_agent.process_input(text)
            transcription["response"] = ai_response
            
            await manager.broadcast({
                "type": "ai_response",
                "data": {"response": ai_response}
            })
            
            await speak_text_async(ai_response)
            
    except sr.UnknownValueError:
        pass
    except Exception as e:
        print(f"Error processing audio: {e}")
        await manager.broadcast({
            "type": "error",
            "data": {"message": "Error processing audio"}
        })
# Request Model for API Input
class UserInput(BaseModel):
    user_input: str

manager = ConnectionManager()


# FastAPI Endpoints
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            if data["type"] == "user_message":
                ai_response = ai_agent.process_input(data["content"])
                await manager.broadcast({
                    "type": "ai_response",
                    "data": {"response": ai_response}
                })
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket)



# FastAPI Endpoints
@app.on_event("startup")
async def startup_event():
    try:
        await ai_agent.connect_db()
    except Exception as e:
        print(f"Startup error: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    await ai_agent.close_db()


@app.post("/start_conversation")
async def start_conversation():
    try:
        # Clean up previous conversation
        transcriptions.clear()
        stop_event.clear()
        
        # Initialize new conversation
        lead_id = datetime.now().strftime("%Y%m%d%H%M%S")
        ai_agent.conversation_history["lead_id"] = lead_id
        ai_agent.conversation_history["start_time"] = datetime.now()
        ai_agent.current_lead["lead_id"] = lead_id
        
        # Initial greeting
        greeting = "Hello! I'm your AI sales agent from Toshal Infotech. How can I help you today?"
        await speak_text_async(greeting)
        
        # Start audio transcription
        manager.transcription_task = asyncio.create_task(transcribe_audio())
        
        return {
            "status": "success",
            "lead_id": lead_id,
            "message": "Conversation started"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stop_conversation")
async def stop_conversation_endpoint():
    return await stop_conversation()


@app.get("/conversation_status")
async def get_conversation_status():
    return {
        "is_active": not stop_event.is_set(),
        "transcriptions_count": len(transcriptions),
        "last_transcription": transcriptions[-1] if transcriptions else None
    }

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("index.html") as f:
        return f.read()
# from fastapi import FastAPI, HTTPException,WebSocket, WebSocketDisconnect
# # from fastapi.responses import JSONResponse
# from fastapi.middleware.cors import CORSMiddleware
# import spacy
# from sentence_transformers import SentenceTransformer
# from datetime import datetime
# import asyncpg
# import google.generativeai as genai
# import os
# import random
# from pydantic import BaseModel
# from config import GEMINI_API_KEY, SUPABASE_URL, SUPABASE_KEY
# import json
# import sounddevice as sd
# import speech_recognition as sr
# import queue
# import time
# from threading import Event #,thread
# import numpy as np
# import pyttsx3
# import asyncio
# from typing import List, Optional,Dict, Any
# import re
# from contextlib import asynccontextmanager
# import logging

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Initialize FastAPI app
# app = FastAPI()

# # Add CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Modify for production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# genai.configure(api_key=GEMINI_API_KEY)
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # Sentence-transformer model
# faiss_index = None
# audio_queue = queue.Queue()  # Queue for audio frames
# transcriptions = []          # List to store transcription results
# recognizer = sr.Recognizer()
# stop_event = Event()  # Add an event to control the recording thread
# recording_thread = None
# tts_engine = pyttsx3.init()



# class ConnectionManager:
#     def __init__(self):
#         self.active_connections: List[WebSocket] = []
#         self.speaking_event = asyncio.Event()

#     async def connect(self, websocket: WebSocket):
#         await websocket.accept()
#         self.active_connections.append(websocket)

#     def disconnect(self, websocket: WebSocket):
#         self.active_connections.remove(websocket)

#     async def broadcast(self, message: dict):
#         for connection in self.active_connections:
#             try:
#                 await connection.send_json(message)
#             except Exception as e:
#                 print(f"Error broadcasting message: {e}")

# # Initialize connection manager
# manager = ConnectionManager()

# @asynccontextmanager
# async def get_db_connection():
#     try:
#         async with ai_agent.db_pool.acquire() as connection:
#             yield connection
#     except Exception as e:
#         logger.error(f"Database connection error: {e}")
#         raise

# # Pydantic models
# class UserInput(BaseModel):
#     user_input: str

# class ConversationStatus(BaseModel):
#     is_active: bool
#     lead_id: Optional[str]
#     timestamp: datetime

# class TTSSettings(BaseModel):
#     rate: Optional[int] = None
#     volume: Optional[float] = None
#     voice: Optional[str] = None

# def datetime_serializer(obj):
#     """Custom serializer for datetime objects"""
#     if isinstance(obj, datetime):
#         return obj.isoformat()  # Convert datetime to ISO 8601 format
#     raise TypeError("Type not serializable")  

# # AI Agent Class
# class AIAgent:
#     def __init__(self):
#         # Initialize spaCy
#         self.nlp = spacy.load("en_core_web_sm")
        
#         self.current_lead = {
#             "name": None,
#             "company": None,
#             "phone": None,
#             "email": None,
#             "requirements": None,
#             "budget": None,
#             "timeline": None,
#             "qualified": False,
#             "meeting_scheduled": False,
#             "meeting_date": None,
#             "meeting_time": None
#         }
        
#         # Services offered (for reference)
#         self.services = {
#             "digital marketing": {
#                 "description": "Complete digital marketing solutions including SEO, PPC, and social media management",
#                 "starting_price": 1500,
#                 "minimum_contract": "3 months"
#             },
#             "web development": {
#                 "description": "Custom website development and e-commerce solutions",
#                 "starting_price": 5000,
#                 "minimum_contract": "1 month"
#             },
#             "consulting": {
#                 "description": "Business growth and digital transformation consulting",
#                 "starting_price": 2000,
#                 "minimum_contract": "1 month"
#             }
#         }
        
#         # Initialize Supabase connection (PostgreSQL)
#         self.db_pool = None
#         self.conversation_history = {
#             "lead_id": None,
#             "start_time": None,
#             "messages": [],
#             "lead_info": self.current_lead
#         }
        
#         # Replace OpenAI initialization with Gemini
#         genai.configure(api_key=GEMINI_API_KEY)
#         self.model = genai.GenerativeModel('gemini-pro')
        
#         # Update system prompt for Gemini
#         self.system_prompt = """You are an AI assistant engaging in natural conversation. While you represent Toshal Infotech, 
# your primary goal is to have genuine, helpful discussions with people. You should:

# 1. Be conversational and natural in your responses
# 2. Avoid repetitive or templated answers
# 3. Show personality and empathy in your interactions
# 4. Adapt your tone to match the user's style
# 5. Be helpful first, sales-focused second

# Company Context:
# - You represent Toshal Infotech
# - Services: Digital Marketing, Web Development, and Consulting
# - You can discuss pricing and services when relevant
# - Focus on understanding needs before suggesting solutions

# Remember to:
# - Keep responses varied and natural
# - Use conversational language
# - Ask follow-up questions naturally
# - Share relevant information when appropriate
# - Be genuine and helpful
# """
        
#     async def connect_db(self):
#         try:
#             if not all([SUPABASE_URL, SUPABASE_KEY]):
#                 raise ValueError("Missing database configuration")
                
#             self.db_pool = await asyncpg.create_pool(
#                 user="postgres.xcayvvljlfsfzgaealev",
#                 password="H@rshu@123",
#                 database="postgres",
#                 host="aws-0-ap-south-1.pooler.supabase.com",
#                 port=5432,
#                 ssl="require"
#             )
#             logger.info("Database connected")
#         except Exception as e:
#             logger.error(f"Database connection error: {e}")
#             raise
        
#     async def close_db(self):
#         if self.db_pool:
#             await self.db_pool.close()
#             logger.info("Database connection closed")
    
#     async def save_conversation(self):
#         try:
#             async with get_db_connection() as conn:
#                 async with conn.transaction():  # Add transaction
#                     conversation_id = await conn.fetchval("""
#                         INSERT INTO conversations (lead_id, start_time, end_time, messages)
#                         VALUES ($1, $2, $3, $4)
#                         RETURNING id
#                     """, 
#                     self.conversation_history["lead_id"],
#                     self.conversation_history["start_time"],
#                     datetime.now(),
#                     json.dumps(self.conversation_history["messages"])
#                     )
                    
#                     # Save lead information within same transaction
#                     await conn.execute("""
#                         INSERT INTO leads (
#                             lead_id, name, company, phone, email,
#                             requirements, budget, timeline, qualified,
#                             meeting_scheduled, meeting_date, meeting_time,
#                             conversation_id
#                         ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
#                     """,
#                     self.current_lead["lead_id"],
#                     self.current_lead["name"],
#                     self.current_lead["company"],
#                     self.current_lead["phone"],
#                     self.current_lead["email"],
#                     self.current_lead["requirements"],
#                     self.current_lead["budget"],
#                     self.current_lead["timeline"],
#                     self.current_lead["qualified"],
#                     self.current_lead["meeting_scheduled"],
#                     self.current_lead["meeting_date"],
#                     self.current_lead["meeting_time"],
#                     conversation_id
#                     )
                    
#                     return conversation_id
#         except Exception as e:
#             logger.error(f"Error saving conversation: {e}")
#             raise

#     def update_conversation(self, speaker: str, message: str):
#         """Add message to conversation history"""
#         self.conversation_history["messages"].append({
#             "speaker": speaker,
#             "message": message,
#             "timestamp": datetime.now().isoformat()
#         })

#     def process_input(self, user_input: str) -> str:
#         """Process user input using Gemini AI for dynamic responses"""
#         try:
#             # Create context from conversation history
#             context = self.system_prompt + "\n\nConversation history:\n"
#             for msg in self.conversation_history["messages"][-5:]:
#                 context += f"{msg['speaker']}: {msg['message']}\n"
            
#             # Generate response
#             chat = self.model.start_chat(history=[])
#             response = chat.send_message(
#                 f"{context}\n\nUser: {user_input}\n\nAssistant:",
#                 generation_config=genai.types.GenerationConfig(
#                     temperature=0.8,
#                     max_output_tokens=200,
#                     top_p=0.8,
#                     top_k=40
#                 )
#             )
            
#             ai_response = response.text.strip()
#             self.update_lead_info(user_input, ai_response)
#             return ai_response
            
#         except Exception as e:
#             print(f"Gemini AI Error: {e}")
#             error_responses = [
#                 "I didn't quite catch that. Could you please rephrase?",
#                 "I'm having a bit of trouble understanding. Could you say that again?",
#                 "Sorry, I missed that. One more time, please?",
#                 "Could you please repeat that in a different way?",
#                 "I'm not sure I understood correctly. Could you elaborate?"
#             ]
#             return random.choice(error_responses)

#     def update_lead_info(self, user_input: str, ai_response: str):
#         """Update lead information based on user input and AI response"""
#         # Extract name if not already set
#         if not self.current_lead["name"] and "name is" in user_input.lower():
#             name = user_input.lower().split("name is")[-1].strip().title()
#             self.current_lead["name"] = name

#         # Update company
#         if not self.current_lead["company"] and "company" in user_input.lower():
#             company = user_input.lower().split("company")[-1].strip().title()
#             self.current_lead["company"] = company

#         # Update email if found in input
#         email_doc = self.nlp(user_input)
#         for token in email_doc:
#             if "@" in token.text and "." in token.text:
#                 self.current_lead["email"] = token.text

#         # Update phone if found in input
#         phone_numbers = re.findall(r'\b\d{10}\b|\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', user_input)
#         if phone_numbers:
#             self.current_lead["phone"] = phone_numbers[0]

#         # Update meeting information
#         if "meeting" in ai_response.lower() and "scheduled" in ai_response.lower():
#             self.current_lead["meeting_scheduled"] = True
#             if "on" in ai_response and "at" in ai_response:
#                 try:
#                     date_time = ai_response.split("on")[-1].split("at")
#                     self.current_lead["meeting_date"] = date_time[0].strip()
#                     self.current_lead["meeting_time"] = date_time[1].split(".")[0].strip()
#                 except:
#                     pass


# # Instantiate AI Agent
# ai_agent = AIAgent()

# async def speak_text_async(text: str):
#     try:
#         if not text:
#             return
            
#         manager.speaking_event.set()
#         await manager.broadcast({
#             "type": "speaking_status",
#             "data": {"is_speaking": True}
#         })
        
#         tts_engine.say(text)
#         tts_engine.runAndWait()
        
#     except Exception as e:
#         logger.error(f"TTS error: {e}")
#         await manager.broadcast({
#             "type": "error",
#             "data": {"message": "Text-to-speech error"}
#         })
#     finally:
#         manager.speaking_event.clear()
#         await manager.broadcast({
#             "type": "speaking_status",
#             "data": {"is_speaking": False}
#         })

# def audio_callback(indata, frames, time, status):
#     if status:
#         print(f"Audio status: {status}")
#     audio_queue.put(indata.copy())

# async def transcribe_audio():
#     try:
#         with sd.InputStream(channels=1, samplerate=16000, callback=audio_callback):
#             while not stop_event.is_set():
#                 if manager.speaking_event.is_set():
#                     await asyncio.sleep(0.1)
#                     continue
                    
#                 audio_data = []
#                 start_time = time.time()
                
#                 while time.time() - start_time < 3 and not stop_event.is_set():
#                     try:
#                         chunk = audio_queue.get(timeout=0.1)
#                         audio_data.append(chunk)
#                     except queue.Empty:
#                         continue

#                 if audio_data:
#                     await process_audio_chunks(audio_data)
                    
#     except Exception as e:
#         print(f"Audio stream error: {e}")
#     finally:
#         print("Audio transcription ended")

# async def process_audio_chunks(audio_data):
#     try:
#         if not audio_data:
#             return
            
#         audio = np.concatenate(audio_data)
#         if np.max(np.abs(audio)) < 0.01:  # Check if audio is too quiet
#             return
            
#         audio_bytes = (audio * 32767).astype(np.int16).tobytes()
        
#         audio_source = sr.AudioData(audio_bytes, 16000, 2)
#         text = recognizer.recognize_google(audio_source)
        
#         if text.strip():
#             transcription = {
#                 "timestamp": time.time(),
#                 "text": text,
#                 "response": None
#             }
#             transcriptions.append(transcription)
            
#             await manager.broadcast({
#                 "type": "user_input",
#                 "data": {"text": text}
#             })
            
#             ai_response = ai_agent.process_input(text)
#             transcription["response"] = ai_response
            
#             await manager.broadcast({
#                 "type": "ai_response",
#                 "data": {"response": ai_response}
#             })
            
#             await speak_text_async(ai_response)
            
#     except sr.UnknownValueError:
#         pass
#     except Exception as e:
#         print(f"Error processing audio: {e}")
#         await manager.broadcast({
#             "type": "error",
#             "data": {"message": "Error processing audio"}
#         })
# # Request Model for API Input
# class UserInput(BaseModel):
#     user_input: str

# # FastAPI Endpoints
# @app.websocket("/ws/{client_id}")
# async def websocket_endpoint(websocket: WebSocket, client_id: str):
#     await manager.connect(websocket)
#     try:
#         while True:
#             data = await websocket.receive_json()
#             if data["type"] == "user_message":
#                 ai_response = ai_agent.process_input(data["content"])
#                 await manager.broadcast({
#                     "type": "ai_response",
#                     "data": {"response": ai_response}
#                 })
#     except WebSocketDisconnect:
#         manager.disconnect(websocket)
#     except Exception as e:
#         print(f"WebSocket error: {e}")
#         manager.disconnect(websocket)


# # FastAPI Endpoints
# @app.on_event("startup")
# async def startup_event():
#     await ai_agent.connect_db()

# @app.on_event("shutdown")
# async def shutdown_event():
#     await ai_agent.close_db()


# @app.post("/start_conversation")
# async def start_conversation():
#     try:
#         # Clean up previous conversation
#         transcriptions.clear()
#         stop_event.clear()
        
#         # Initialize new conversation
#         lead_id = datetime.now().strftime("%Y%m%d%H%M%S")
#         ai_agent.conversation_history["lead_id"] = lead_id
#         ai_agent.conversation_history["start_time"] = datetime.now()
#         ai_agent.current_lead["lead_id"] = lead_id
        
#         # Initial greeting
#         greeting = "Hello! I'm your AI assistant from Toshal Infotech. How can I help you today?"
#         await speak_text_async(greeting)
        
#         # Start audio transcription
#         asyncio.create_task(transcribe_audio())
        
#         return {
#             "status": "success",
#             "lead_id": lead_id,
#             "message": "Conversation started"
#         }
#     except Exception as e:
#         logger.error(f"Error starting conversation: {e}")
#         raise HTTPException(status_code=500, detail=str(e))

# @app.post("/stop_conversation")
# async def stop_conversation():
#     try:
#         # Stop audio processing
#         stop_event.set()
        
#         # Wait for any pending audio processing
#         await asyncio.sleep(1)
        
#         # Save conversation
#         conversation_id = await ai_agent.save_conversation()
        
#         # Clean up resources
#         transcriptions.clear()
#         audio_queue.queue.clear()
        
#         # Reset AI agent conversation history
#         ai_agent.conversation_history = {
#             "lead_id": None,
#             "start_time": None,
#             "messages": [],
#             "lead_info": ai_agent.current_lead.copy()
#         }
        
#         return {
#             "status": "success",
#             "conversation_id": conversation_id,
#             "message": "Conversation ended and saved"
#         }
#     except Exception as e:
#         logger.error(f"Error stopping conversation: {e}")
#         raise HTTPException(status_code=500, detail=str(e))


# @app.get("/conversation_status")
# async def get_conversation_status():
#     return {
#         "is_active": not stop_event.is_set(),
#         "transcriptions_count": len(transcriptions),
#         "last_transcription": transcriptions[-1] if transcriptions else None
#     }

# class TTSSettings(BaseModel):
#     rate: Optional[int] = None  # Words per minute
#     volume: Optional[float] = None  # 0.0 to 1.0
#     voice: Optional[str] = None  # Name of the voice to use

# @app.post("/tts_settings/")
# async def update_tts_settings(settings: TTSSettings):
#     try:
#         if settings.rate is not None:
#             tts_engine.setProperty('rate', settings.rate)
#         if settings.volume is not None:
#             tts_engine.setProperty('volume', settings.volume)
#         if settings.voice is not None:
#             voices = tts_engine.getProperty('voices')
#             for voice in voices:
#                 if settings.voice.lower() in voice.name.lower():
#                     tts_engine.setProperty('voice', voice.id)
#                     break
        
#         return {
#             "message": "TTS settings updated successfully",
#             "current_settings": {
#                 "rate": tts_engine.getProperty('rate'),
#                 "volume": tts_engine.getProperty('volume'),
#                 "voice": tts_engine.getProperty('voice')
#             }
#         }
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Failed to update TTS settings: {str(e)}")

# # Add endpoint to get available voices
# @app.get("/tts_voices/")
# async def get_tts_voices():
#     voices = tts_engine.getProperty('voices')
#     return {
#         "voices": [
#             {
#                 "name": voice.name,
#                 "id": voice.id,
#                 "languages": voice.languages
#             }
#             for voice in voices
#         ]
#     }
