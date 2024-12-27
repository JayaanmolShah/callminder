from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import spacy
import speech_recognition as sr
import pyttsx3
from datetime import datetime
import asyncpg
import google.generativeai as genai
import os
import random
from pydantic import BaseModel
from config import GEMINI_API_KEY, SUPABASE_URL, SUPABASE_KEY
import json
from datetime import datetime

# Initialize FastAPI app
app = FastAPI()
def datetime_serializer(obj):
    """Custom serializer for datetime objects"""
    if isinstance(obj, datetime):
        return obj.isoformat()  # Convert datetime to ISO 8601 format
    raise TypeError("Type not serializable")

# AI Agent Class
class AIAgent:
    def __init__(self):
        # Initialize spaCy
        self.nlp = spacy.load("en_core_web_sm")
        
        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        
        # Initialize text-to-speech engine
        self.engine = pyttsx3.init()
        
        # Lead information storage
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
        
        # Services offered (for reference)
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
        
        # Initialize Supabase connection (PostgreSQL)
        self.db_pool = None
        self.conversation_history = {
            "lead_id": None,
            "start_time": None,
            "messages": [],
            "lead_info": self.current_lead
        }
        
        # Replace OpenAI initialization with Gemini
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel('gemini-pro')
        
        # Update system prompt for Gemini
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
                port=5432
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
                # Use json.dumps() with custom datetime serializer
                await connection.execute(query, 
                    self.conversation_history["lead_id"],
                    self.conversation_history["start_time"],
                    datetime.now(),
                    json.dumps(self.conversation_history["messages"], default=datetime_serializer),  # Handle datetime serialization
                    json.dumps(self.conversation_history["lead_info"], default=datetime_serializer)  # Handle datetime serialization
                )
                print("Conversation saved to database")
        except Exception as e:
            print(f"Error saving to database: {e}")

    def update_conversation(self, speaker, message):
        """Add message to conversation history"""
        self.conversation_history["messages"].append({
            "speaker": speaker,
            "message": message,
            "timestamp": datetime.now()
        })

    def process_input(self, user_input):
        """Process user input using Gemini AI for dynamic responses"""
        try:
            # Create conversation context
            context = self.system_prompt
            
            # Add lead information context if available
            if any(value for value in self.current_lead.values() if value):
                lead_context = "\nCurrent lead information:\n" + \
                    "\n".join([f"{k}: {v}" for k, v in self.current_lead.items() if v])
                context += lead_context
            
            # Add conversation history
            history = "\nRecent conversation:\n"
            for message in self.conversation_history["messages"][-5:]:
                history += f"{message['speaker']}: {message['message']}\n"
            context += history
            
            # Generate response using Gemini
            chat = self.model.start_chat(history=[])
            response = chat.send_message(
                f"{context}\n\nUser: {user_input}\n\nAssistant:",
                generation_config=genai.types.GenerationConfig(
                    temperature=0.8,
                    max_output_tokens=200,
                )
            )
            
            # Extract and process the response
            ai_response = response.text.strip()
            
            # Update lead information based on the response and user input
            self.update_lead_info(user_input, ai_response)
            
            return ai_response
            
        except Exception as e:
            print(f"Gemini AI Error: {e}")
            error_responses = [
                "I didn't quite catch that. Could you please rephrase?",
                "I'm having a bit of trouble understanding. Could you say that again?",
                "Sorry, I missed that. One more time, please?",
                "Could you please repeat that in a different way?",
                "I'm not sure I understood correctly. Could you elaborate?"
            ]
            return random.choice(error_responses)

    def update_lead_info(self, user_input, ai_response):
        """Update lead information based on user input and AI response"""
        # Extract name if not already set
        if not self.current_lead["name"] and "name is" in user_input.lower():
            name = user_input.lower().split("name is")[-1].strip()
            self.current_lead["name"] = name.title()
        
        # Extract company if not already set
        if not self.current_lead["company"] and "company" in user_input.lower():
            company = user_input.lower().split("company")[-1].strip()
            self.current_lead["company"] = company.title()
        
        # Update meeting information if present in AI response
        if "meeting" in ai_response.lower() and "scheduled" in ai_response.lower():
            self.current_lead["meeting_scheduled"] = True
            
            # Try to extract date and time from the response
            if "on" in ai_response and "at" in ai_response:
                try:
                    date_time = ai_response.split("on")[-1].split("at")
                    self.current_lead["meeting_date"] = date_time[0].strip()
                    self.current_lead["meeting_time"] = date_time[1].split(".")[0].strip()
                except:
                    pass


# Instantiate AI Agent
ai_agent = AIAgent()

# Request Model for API Input
class UserInput(BaseModel):
    user_input: str


# FastAPI Endpoints
@app.on_event("startup")
async def startup():
    await ai_agent.connect_db()
    print("Database connection pool initialized.")

@app.on_event("shutdown")
async def shutdown():
    await ai_agent.close_db()
    print("Database connection pool closed.")


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
