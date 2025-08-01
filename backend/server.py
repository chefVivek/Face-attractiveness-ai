from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import uuid
from datetime import datetime
from face_analyzer import FaceAttractivenessAnalyzer

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Initialize face analyzer
face_analyzer = FaceAttractivenessAnalyzer()

# Define Models
class StatusCheck(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class StatusCheckCreate(BaseModel):
    client_name: str

class FaceAnalysisResult(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    overall_score: float
    symmetry_score: float
    golden_ratio_score: float
    feature_breakdown: Dict[str, float]
    analysis: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# Add your routes to the router instead of directly to app
@api_router.get("/")
async def root():
    return {"message": "Face Attractiveness API is running!"}

@api_router.post("/analyze-face")
async def analyze_face(file: UploadFile = File(...)):
    """Analyze uploaded face image for attractiveness scoring"""
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read file content
        file_content = await file.read()
        
        if len(file_content) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        # Analyze the face
        result = face_analyzer.analyze_face(file_content)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        # Create analysis result
        analysis_result = FaceAnalysisResult(**result)
        
        # Store in database (optional)
        try:
            await db.face_analyses.insert_one(analysis_result.dict())
        except Exception as e:
            logger.warning(f"Failed to store analysis in database: {e}")
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing face: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during analysis")

@api_router.get("/analysis-history")
async def get_analysis_history(limit: int = 10):
    """Get recent face analysis history"""
    try:
        analyses = await db.face_analyses.find().sort("timestamp", -1).limit(limit).to_list(limit)
        return analyses
    except Exception as e:
        logger.error(f"Error fetching analysis history: {e}")
        return []

@api_router.post("/status", response_model=StatusCheck)
async def create_status_check(input: StatusCheckCreate):
    status_dict = input.dict()
    status_obj = StatusCheck(**status_dict)
    _ = await db.status_checks.insert_one(status_obj.dict())
    return status_obj

@api_router.get("/status", response_model=List[StatusCheck])
async def get_status_checks():
    status_checks = await db.status_checks.find().to_list(1000)
    return [StatusCheck(**status_check) for status_check in status_checks]

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()