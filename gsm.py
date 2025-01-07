from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import asyncio
import websockets
import spacy
import requests
import google.generativeai as genai
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv
import random
import json
import logging
from datetime import datetime
from typing import Dict, Optional
import aiohttp
import asyncpg
from dataclasses import dataclass
import wave
import io
import os
from datetime import datetime
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

def datetime_serializer(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()  # Convert datetime to ISO 8601 format
    raise TypeError("Type not serializable")
    
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

app = FastAPI()

# Configuration
GSM_GATEWAY_URL = "http://gsm_gateway_ip/api/start_call"
TRANSCRIPTION_WS_URL = "wss://transcription_service_url"
TTS_SERVICE_URL = "http://tts_service_url/api/generate_speech"  # Text-to-Speech service

@dataclass
class CallSession:
    call_id: str
    from_number: str
    to_number: str
    start_time: datetime
    status: str
    audio_buffer: io.BytesIO
    transcription_buffer: list
    websocket: Optional[WebSocket] = None

#AGENT CLASS

class AIAgent:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
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
            "meeting_time": None,
        }
        self.db_pool = None
        self.conversation_history = {
            "lead_id": None,
            "start_time": None,
            "messages": [],
            "lead_info": self.current_lead,
        }
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel("gemini-pro")
        self.system_prompt = """You are an AI assistant engaging in natural conversation. While you represent Toshal Infotech, 
your primary goal is to have genuine, helpful discussions with people. You should:

1. Be conversational and natural in your responses
2. Avoid repetitive or templated answers
3. Show personality and empathy in your interactions
4. Adapt your tone to match the user's style
5. Be helpful first, sales-focused second

Company Context:
- You represent Toshal Infotech
- Services: Digital Marketing, Web Development, and Consulting
- You can discuss pricing and services when relevant
- Focus on understanding needs before suggesting solutions

Remember to:
- Keep responses varied and natural
- Use conversational language
- Ask follow-up questions naturally
- Share relevant information when appropriate
- Be genuine and helpful
"""

    async def connect_db(self):
        """Connect to the Supabase database (PostgreSQL)"""
        try:
            self.db_pool = await asyncpg.create_pool(
                user="postgres.xcayvvljlfsfzgaealev",
                password="H@rshu@123",
                database="postgres",
                host="aws-0-ap-south-1.pooler.supabase.com",
                port=5432,
            )
            print("Database connected")
        except Exception as e:
            print(f"Error connecting to database: {e}")

    async def close_db(self):
        """Close the database connection pool"""
        if self.db_pool:
            await self.db_pool.close()
            print("Database connection pool closed.")

    async def save_to_db(self):
        """Save conversation history to Supabase (PostgreSQL)"""
        try:
            async with self.db_pool.acquire() as connection:
                query = """
                INSERT INTO conversations (lead_id, start_time, end_time, messages, lead_info)
                VALUES ($1, $2, $3, $4, $5)
                """
                await connection.execute(
                    query,
                    self.conversation_history["lead_id"],
                    self.conversation_history["start_time"],
                    datetime.now(),
                    json.dumps(self.conversation_history["messages"], default=datetime_serializer),
                    json.dumps(self.conversation_history["lead_info"], default=datetime_serializer),
                )
                print("Conversation saved to database")
        except Exception as e:
            print(f"Error saving to database: {e}")

    def update_conversation(self, speaker, message):
        """Add message to conversation history"""
        self.conversation_history["messages"].append(
            {"speaker": speaker, "message": message, "timestamp": datetime.now()}
        )

    def process_input(self, user_input):
        """Process user input using Gemini AI for dynamic responses"""
        try:
            context = self.system_prompt
            if any(value for value in self.current_lead.values() if value):
                lead_context = "\nCurrent lead information:\n" + "\n".join(
                    [f"{k}: {v}" for k, v in self.current_lead.items() if v]
                )
                context += lead_context
            history = "\nRecent conversation:\n"
            for message in self.conversation_history["messages"][-5:]:
                history += f"{message['speaker']}: {message['message']}\n"
            context += history

            chat = self.model.start_chat(history=[])
            response = chat.send_message(
                f"{context}\n\nUser: {user_input}\n\nAssistant:",
                generation_config=genai.types.GenerationConfig(
                    temperature=0.8,
                    max_output_tokens=200,
                ),
            )
            ai_response = response.text.strip()
            self.update_lead_info(user_input, ai_response)
            return ai_response
        except Exception as e:
            print(f"Gemini AI Error: {e}")
            error_responses = [
                "I didn't quite catch that. Could you please rephrase?",
                "I'm having a bit of trouble understanding. Could you say that again?",
                "Sorry, I missed that. One more time, please?",
                "Could you please repeat that in a different way?",
                "I'm not sure I understood correctly. Could you elaborate?",
            ]
            return random.choice(error_responses)

    def update_lead_info(self, user_input, ai_response):
        """Update lead information based on user input and AI response"""
        if not self.current_lead["name"] and "name is" in user_input.lower():
            name = user_input.lower().split("name is")[-1].strip()
            self.current_lead["name"] = name.title()
        if not self.current_lead["company"] and "company" in user_input.lower():
            company = user_input.lower().split("company")[-1].strip()
            self.current_lead["company"] = company.title()
        if "meeting" in ai_response.lower() and "scheduled" in ai_response.lower():
            self.current_lead["meeting_scheduled"] = True
            if "on" in ai_response and "at" in ai_response:
                try:
                    date_time = ai_response.split("on")[-1].split("at")
                    self.current_lead["meeting_date"] = date_time[0].strip()
                    self.current_lead["meeting_time"] = date_time[1].split(".")[0].strip()
                except:
                    pass




class CallManager:
    def __init__(self):
        self.active_calls: Dict[str, CallSession] = {}
        self.db_pool = None

    async def setup_database(self):
        """Initialize database connection pool"""
        self.db_pool = await asyncpg.create_pool(
            user="postgres.xcayvvljlfsfzgaealev",
            password="H@rshu@123",
            database="postgres",
            host="aws-0-ap-south-1.pooler.supabase.com"
        )

    async def store_transcription(self, call_id: str, transcription: str):
        """Store transcription in database"""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO call_transcriptions (call_id, transcription, timestamp)
                VALUES ($1, $2, $3)
            """, call_id, transcription, datetime.now())

    async def initiate_call(self, from_number: str, to_number: str) -> str:
        """Initiate a call and return call_id"""
        try:
            # Call GSM gateway
            async with aiohttp.ClientSession() as session:
                async with session.post(GSM_GATEWAY_URL, json={
                    "from": from_number,
                    "to": to_number
                }) as response:
                    if response.status != 200:
                        raise HTTPException(status_code=500, detail="Failed to initiate call")
                    
                    call_data = await response.json()
                    call_id = call_data.get("call_id")

            # Create call session
            self.active_calls[call_id] = CallSession(
                call_id=call_id,
                from_number=from_number,
                to_number=to_number,
                start_time=datetime.now(),
                status="initiating",
                audio_buffer=io.BytesIO(),
                transcription_buffer=[]
            )

            return call_id

        except Exception as e:
            logger.error(f"Error initiating call: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def process_audio(self, call_id: str, audio_data: bytes):
        """Process incoming audio data"""
        call_session = self.active_calls.get(call_id)
        if not call_session:
            raise HTTPException(status_code=404, detail="Call not found")

        # Store audio data
        call_session.audio_buffer.write(audio_data)

        # Convert to proper format for transcription service
        audio_chunk = self.prepare_audio_chunk(audio_data)
        return audio_chunk

    def prepare_audio_chunk(self, audio_data: bytes) -> bytes:
        """Prepare audio data for transcription service"""
        # Convert audio to proper format (example: PCM 16-bit, 8kHz)
        with wave.open(io.BytesIO(audio_data), 'rb') as wave_file:
            # Implement audio format conversion here
            return audio_data  # Placeholder return

    async def get_ai_response(self, transcription: str) -> str:
        """Send transcription to AI agent and get response"""
        ai_response = await ai_agent.process_input(transcription)
        return ai_response

    async def send_tts_response(self, ai_response: str, call_id: str):
        """Send AI response to TTS service and play it back on the call"""
        async with aiohttp.ClientSession() as session:
            async with session.post(TTS_SERVICE_URL, json={
                "text": ai_response
            }) as response:
                if response.status == 200:
                    audio_data = await response.read()
                    # Play the audio data back on the call
                    call_session = self.active_calls.get(call_id)
                    if call_session:
                        await self.play_audio_to_call(call_session, audio_data)
                else:
                    logger.error("TTS Service failed to generate speech.")

    async def play_audio_to_call(self, call_session: CallSession, audio_data: bytes):
        """Play the audio to the call"""
        # Implement playing the audio data back into the call here
        pass

call_manager = CallManager()
ai_agent = AIAgent()

# Request Model for API Input
class UserInput(BaseModel):
    user_input: str


@app.on_event("startup")
async def startup():
    await ai_agent.connect_db()
    
@app.on_event("shutdown")
async def shutdown():
    await ai_agent.close_db()

@app.post("/start_conversation/")
async def start_conversation():
    ai_agent.conversation_history["lead_id"] = datetime.now().strftime("%Y%m%d%H%M%S")
    ai_agent.conversation_history["start_time"] = datetime.now()
    return {"message": "Conversation started"}

@app.post("/send_message/")
async def send_message(data: UserInput):
    user_input = data.user_input
    ai_response = ai_agent.process_input(user_input)
    ai_agent.update_conversation("User", user_input)
    ai_agent.update_conversation("AI", ai_response)
    return {"response": ai_response}

@app.post("/stop_conversation/")
async def stop_conversation():
    try:
        await ai_agent.save_to_db()
        return {"message": "Conversation saved successfully"}
    except Exception as e:
        return {"error": str(e)}

@app.get("/status/")
async def get_status():
    return {"status": "AI Sales Assistant is running"}

@app.websocket("/call_transcription")
async def call_transcription(websocket: WebSocket):
    await websocket.accept()
    call_id = None

    try:
        # Receive call initiation details
        init_data = await websocket.receive_json()
        from_number = init_data.get("from_number")
        to_number = init_data.get("to_number")

        if not from_number or not to_number:
            await websocket.close(code=1000, reason="Missing call details")
            return

        # Initiate call
        call_id = await call_manager.initiate_call(from_number, to_number)
        call_session = call_manager.active_calls[call_id]
        call_session.websocket = websocket

        await websocket.send_json({"status": "call_initiated", "call_id": call_id})

        # Connect to transcription service
        async with websockets.connect(TRANSCRIPTION_WS_URL) as transcription_ws:
            await websocket.send_json({"status": "transcription_connected"})

            async def handle_audio():
                try:
                    while True:
                        # Receive audio data
                        audio_data = await websocket.receive_bytes()
                        
                        # Process audio
                        processed_audio = await call_manager.process_audio(call_id, audio_data)
                        
                        # Send to transcription service
                        await transcription_ws.send(processed_audio)
                except Exception as e:
                    logger.error(f"Error in audio handling: {e}")
                    raise

            async def handle_transcription():
                try:
                    while True:
                        # Receive transcription
                        transcription = await transcription_ws.recv()
                        transcription_data = json.loads(transcription)
                        text = transcription_data.get("text")

                        # Store transcription
                        await call_manager.store_transcription(call_id, text)

                        # Send transcription to AI agent and get response
                        ai_response = await call_manager.get_ai_response(text)

                        # Send AI response to TTS and play on call
                        await call_manager.send_tts_response(ai_response, call_id)

                        # Send to client
                        await websocket.send_json({
                            "type": "transcription",
                            "data": transcription_data
                        })
                except Exception as e:
                    logger.error(f"Error in transcription handling: {e}")
                    raise

            # Handle audio and transcription concurrently
            await asyncio.gather(
                handle_audio(),
                handle_transcription()
            )

    except Exception as e:
        logger.error(f"Error in call transcription: {e}")
        await websocket.send_json({
            "type": "error",
            "message": str(e)
        })
    finally:
        if call_id and call_id in call_manager.active_calls:
            del call_manager.active_calls[call_id]
        await websocket.close()

@app.get("/calls/{call_id}/transcription")
async def get_call_transcription(call_id: str):
    """Get stored transcriptions for a call"""
    async with call_manager.db_pool.acquire() as conn:
        transcriptions = await conn.fetch("""
            SELECT * FROM call_transcriptions 
            WHERE call_id = $1 
            ORDER BY timestamp ASC
        """, call_id)
        return {"transcriptions": [dict(t) for t in transcriptions]}