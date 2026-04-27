import asyncio
import base64
import json
import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from gemini_live import GeminiLive

# Load environment variables
load_dotenv()

# Configure logging - DEBUG for our modules, INFO for everything else
logging.basicConfig(level=logging.INFO)
logging.getLogger("gemini_live").setLevel(logging.DEBUG)
logging.getLogger(__name__).setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("MODEL", "gemini-3.1-flash-live-preview")

# ============ MOCK BACKEND DATA ============

VEHICLES = {
    "default": {
        "vehicle_number": "GJ05GT0903",
        "owner_name": "Dashrath Patel",
        "phone": "+919876543210",
        "model": "Maruti Suzuki Baleno",
        "year": 2024,
        "purchase_date": "2024-10-30",
        "warranty_expiry": "2026-10-30",
        "warranty_active": True,
        "current_km_system": 8604,
        "service_history": [
            {
                "service_number": 1,
                "date": "2025-02-15",
                "km": 1023,
                "workshop": "Kataria Automobiles, Ahmedabad",
                "type": "First Free Service",
                "cost": 0
            },
            {
                "service_number": 2,
                "date": "2025-08-20",
                "km": 5111,
                "workshop": "Karodhra Workshop",
                "type": "Second Service",
                "cost": 2200
            }
        ],
        "next_service": {
            "service_number": 3,
            "type": "Third Service",
            "due_km": 10000,
            "estimated_cost_min": 2500,
            "estimated_cost_max": 3000
        },
        "pickup_drop_free": True
    }
}

def handle_get_vehicle_info(**kwargs):
    return VEHICLES["default"]

def handle_schedule_pickup(**kwargs):
    return {
        "success": True,
        "booking_id": "BK-20260413-001",
        "vehicle_number": kwargs.get("vehicle_number", "GJ05GT0903"),
        "pickup_date": kwargs.get("date", "2026-04-13"),
        "pickup_time": kwargs.get("time", "9:30 AM"),
        "driver_name": "Rajesh Kumar",
        "driver_phone": "+919876500001",
        "workshop": "Kataria Automobiles, S.G. Highway, Ahmedabad",
        "special_instructions": kwargs.get("special_instructions", ""),
        "note": "Driver details will be sent via SMS on the morning of pickup."
    }

def handle_get_service_cost_estimate(**kwargs):
    estimates = {
        "Third Service": {"min": 2500, "max": 3000, "includes": "Oil change, filter replacement, brake inspection, general checkup"},
        "Second Service": {"min": 2000, "max": 2500, "includes": "Oil change, filter check, general inspection"},
        "First Free Service": {"min": 0, "max": 0, "includes": "General inspection, fluid top-up (free under warranty)"},
    }
    service_type = kwargs.get("service_type", "Third Service")
    return estimates.get(service_type, {"min": 2000, "max": 4000, "includes": "General service"})


# Initialize FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="frontend"), name="static")


@app.get("/")
async def root():
    return FileResponse("frontend/index.html")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for Gemini Live."""
    await websocket.accept()

    logger.info("WebSocket connection accepted")

    audio_input_queue = asyncio.Queue()
    video_input_queue = asyncio.Queue()
    text_input_queue = asyncio.Queue()

    async def audio_output_callback(data):
        await websocket.send_bytes(data)

    async def audio_interrupt_callback():
        # The event queue handles the JSON message, but we might want to do something else here
        pass

    gemini_client = GeminiLive(
        api_key=GEMINI_API_KEY, 
        model=MODEL, 
        input_sample_rate=16000,
        tool_mapping={
            "get_vehicle_info": handle_get_vehicle_info,
            "schedule_pickup": handle_schedule_pickup,
            "get_service_cost_estimate": handle_get_service_cost_estimate,
        }
    )

    async def receive_from_client():
        try:
            while True:
                message = await websocket.receive()

                if message.get("bytes"):
                    await audio_input_queue.put(message["bytes"])
                elif message.get("text"):
                    text = message["text"]
                    try:
                        payload = json.loads(text)
                        if isinstance(payload, dict) and payload.get("type") == "image":
                            logger.info(f"Received image chunk from client: {len(payload['data'])} base64 chars")
                            image_data = base64.b64decode(payload["data"])
                            await video_input_queue.put(image_data)
                            continue
                    except json.JSONDecodeError:
                        pass

                    await text_input_queue.put(text)
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected")
        except Exception as e:
            logger.error(f"Error receiving from client: {e}")

    receive_task = asyncio.create_task(receive_from_client())

    async def run_session():
        async for event in gemini_client.start_session(
            audio_input_queue=audio_input_queue,
            video_input_queue=video_input_queue,
            text_input_queue=text_input_queue,
            audio_output_callback=audio_output_callback,
            audio_interrupt_callback=audio_interrupt_callback,
        ):
            if event:
                # Forward events (transcriptions, etc) to client
                await websocket.send_json(event)

    try:
        await run_session()
    except Exception as e:
        import traceback
        logger.error(f"Error in Gemini session: {type(e).__name__}: {e}\n{traceback.format_exc()}")
    finally:
        receive_task.cancel()
        # Ensure websocket is closed if not already
        try:
            await websocket.close()
        except:
            pass


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
