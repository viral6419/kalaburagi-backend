Click "Add file" → "Create new file"
Name it: server.py
Copy the code below in 3 parts (it's long):
PART 1 of 3 - Copy this first:
from fastapi import FastAPI, APIRouter, HTTPException, Depends, Query, File, UploadFile
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import FileResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field, BeforeValidator, ConfigDict
from typing import List, Optional, Annotated
import os, uuid, logging, bcrypt
from datetime import datetime, timezone
from pathlib import Path
from jose import jwt as jose_jwt
from emergentintegrations.llm.chat import LlmChat, UserMessage

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

UPLOADS_DIR = os.environ.get('UPLOADS_DIR', str(ROOT_DIR / 'uploads'))

mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

JWT_SECRET = os.environ.get('JWT_SECRET', 'kalaburagi-secret-2024')
EMERGENT_LLM_KEY = os.environ.get('EMERGENT_LLM_KEY', '')

app = FastAPI()
api_router = APIRouter(prefix="/api")
security = HTTPBearer(auto_error=False)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PyObjectId = Annotated[str, BeforeValidator(str)]

def hash_password(pwd: str) -> str:
    return bcrypt.hashpw(pwd.encode(), bcrypt.gensalt()).decode()

def verify_password(pwd: str, hashed: str) -> bool:
    return bcrypt.checkpw(pwd.encode(), hashed.encode())

def create_token(user_id: str, role: str) -> str:
    return jose_jwt.encode({"user_id": user_id, "role": role}, JWT_SECRET, algorithm="HS256")

class PropertyCreate(BaseModel):
    title: str
    type: str
    status: str
    price: float
    price_unit: str = "lakhs"
    area: float
    area_unit: str = "sqft"
    bedrooms: Optional[int] = None
    bathrooms: Optional[int] = None
    location: str
    pincode: str = "585101"
    address: str
    description: str
    images: List[str] = []
    amenities: List[str] = []
    is_featured: bool = False
    is_new_launch: bool = False
    ready_to_move: bool = True
    agent_name: str = "Kalaburagi Estates"
    agent_phone: str = "+91 9110278059"

class LeadCreate(BaseModel):
    name: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    intent: str = "buy"
    budget_min: Optional[float] = None
    budget_max: Optional[float] = None
    preferred_location: Optional[str] = None
    visit_date: Optional[str] = None
    contact_time: Optional[str] = None
    is_investor: bool = False
    property_id: Optional[str] = None
    source: str = "chatbot"
    session_id: Optional[str] = None
    notes: Optional[str] = None

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str
    lead_data: Optional[dict] = {}

class UserRegister(BaseModel):
    name: str
    email: str
    phone: Optional[str] = None
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

class LeadUpdate(BaseModel):
    status: Optional[str] = None
    score: Optional[str] = None
    notes: Optional[str] = None

class UserListingCreate(BaseModel):
    submitter_name: str
    submitter_phone: str
    submitter_email: Optional[str] = None
    title: str
    type: str
    status: str = "sale"
    price: float
    price_unit: str = "lakhs"
    area: float
    area_unit: str = "sqft"
    bedrooms: Optional[int] = None
    bathrooms: Optional[int] = None
    location: str
    pincode: str = "585101"
    address: str
    description: str
    images: List[str] = []
    amenities: List[str] = []
    utr_id: str

class UserListingReject(BaseModel):
    rejection_reason: Optional[str] = "Listing does not meet our criteria"
@api_router.get("/properties", response_model=List[dict])
async def get_properties(
    type: Optional[str] = None,
    status: Optional[str] = None,
    location: Optional[str] = None,
    pincode: Optional[str] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    bedrooms: Optional[int] = None,
    is_featured: Optional[bool] = None,
    is_new_launch: Optional[bool] = None,
    skip: int = 0,
    limit: int = 20
):
    query = {}
    if type: query["type"] = type
    if status: query["status"] = status
    if location: query["location"] = {"$regex": location, "$options": "i"}
    if pincode: query["pincode"] = pincode
    if bedrooms: query["bedrooms"] = bedrooms
    if is_featured is not None: query["is_featured"] = is_featured
    if is_new_launch is not None: query["is_new_launch"] = is_new_launch
    if min_price is not None or max_price is not None:
        query["price"] = {}
        if min_price is not None: query["price"]["$gte"] = min_price
        if max_price is not None: query["price"]["$lte"] = max_price
    cursor = db.properties.find(query, {"_id": 0}).skip(skip).limit(limit)
    return await cursor.to_list(limit)

@api_router.get("/properties/{property_id}", response_model=dict)
async def get_property(property_id: str):
    prop = await db.properties.find_one({"property_id": property_id}, {"_id": 0})
    if not prop:
        raise HTTPException(status_code=404, detail="Property not found")
    await db.properties.update_one({"property_id": property_id}, {"$inc": {"views": 1}})
    return prop

@api_router.post("/properties", response_model=dict)
async def create_property(prop: PropertyCreate):
    doc = {**prop.model_dump(), "property_id": str(uuid.uuid4()), "views": 0,
           "created_at": datetime.now(timezone.utc).isoformat()}
    await db.properties.insert_one(doc)
    doc.pop("_id", None)
    return doc

@api_router.put("/properties/{property_id}", response_model=dict)
async def update_property(property_id: str, prop: PropertyCreate):
    await db.properties.update_one({"property_id": property_id}, {"$set": prop.model_dump()})
    return await db.properties.find_one({"property_id": property_id}, {"_id": 0})

@api_router.delete("/properties/{property_id}")
async def delete_property(property_id: str):
    await db.properties.delete_one({"property_id": property_id})
    return {"message": "Deleted"}

@api_router.post("/leads", response_model=dict)
async def create_lead(lead: LeadCreate):
    score = "cold"
    if lead.budget_max and lead.budget_max >= 50: score = "hot"
    elif lead.visit_date: score = "warm"
    doc = {**lead.model_dump(), "lead_id": str(uuid.uuid4()), "score": score,
           "status": "new", "created_at": datetime.now(timezone.utc).isoformat()}
    await db.leads.insert_one(doc)
    doc.pop("_id", None)
    return doc

@api_router.get("/leads", response_model=List[dict])
async def get_leads(skip: int = 0, limit: int = 50):
    cursor = db.leads.find({}, {"_id": 0}).sort("created_at", -1).skip(skip).limit(limit)
    return await cursor.to_list(limit)

@api_router.put("/leads/{lead_id}", response_model=dict)
async def update_lead(lead_id: str, data: LeadUpdate):
    update = {k: v for k, v in data.model_dump().items() if v is not None}
    await db.leads.update_one({"lead_id": lead_id}, {"$set": update})
    return await db.leads.find_one({"lead_id": lead_id}, {"_id": 0})

@api_router.post("/chat", response_model=dict)
async def chat(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())
    session = await db.chat_sessions.find_one({"session_id": session_id})
    messages_history = session.get("messages", []) if session else []
    system_message = """You are Kavya, a friendly and professional real estate assistant for Kalaburagi Estates — a premium real estate platform in Gulbarga (Kalaburagi), Karnataka, India.

Your mission: Qualify leads naturally by collecting these details step by step:
1. Intent: Buy, Sell, or Rent?
2. Budget: Range in Lakhs or Crores?
3. Preferred area in Kalaburagi (Sedam Road, Super Market, Kapnoor, Station Area, Bidar Road, etc.)?
4. Timeline: When planning to visit?
5. Best time to contact?
6. Investor or end-user?

Rules:
- Keep responses SHORT and conversational (2-3 sentences max)
- Be warm, professional, and helpful
- After collecting key info, confirm inquiry is saved and agent will contact within 24 hours
- Represent a premium luxury brand"""
    chat_obj = LlmChat(
        api_key=EMERGENT_LLM_KEY,
        session_id=session_id,
        system_message=system_message
    ).with_model("openai", "gpt-4.1")
    user_msg = UserMessage(text=request.message)
    response_text = await chat_obj.send_message(user_msg)
    messages_history.append({"role": "user", "content": request.message})
    messages_history.append({"role": "assistant", "content": response_text})
    if session:
        await db.chat_sessions.update_one({"session_id": session_id}, {"$set": {"messages": messages_history}})
    else:
        await db.chat_sessions.insert_one({
            "session_id": session_id, "messages": messages_history,
            "lead_data": request.lead_data or {},
            "created_at": datetime.now(timezone.utc).isoformat()
        })
    return {"session_id": session_id, "message": response_text}

@api_router.get("/stats", response_model=dict)
async def get_stats():
    return {
        "total_properties": await db.properties.count_documents({}),
        "total_leads": await db.leads.count_documents({}),
        "hot_leads": await db.leads.count_documents({"score": "hot"}),
        "for_sale": await db.properties.count_documents({"status": "sale"}),
        "for_rent": await db.properties.count_documents({"status": "rent"}),
        "happy_clients": 1250,
        "years_of_trust": 8,
        "properties_sold": 850
    }

@api_router.post("/auth/register", response_model=dict)
async def register(data: UserRegister):
    if await db.users.find_one({"email": data.email}):
        raise HTTPException(status_code=400, detail="Email already registered")
    user_id = str(uuid.uuid4())
    doc = {"user_id": user_id, "name": data.name, "email": data.email,
           "phone": data.phone, "password_hash": hash_password(data.password),
           "role": "user", "favorites": [],
           "created_at": datetime.now(timezone.utc).isoformat()}
    await db.users.insert_one(doc)
    token = create_token(user_id, "user")
    return {"token": token, "user": {"id": user_id, "name": data.name, "email": data.email, "role": "user"}}

@api_router.post("/auth/login", response_model=dict)
async def login(data: UserLogin):
    user = await db.users.find_one({"email": data.email})
    if not user or not verify_password(data.password, user.get("password_hash", "")):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_token(user["user_id"], user["role"])
    return {"token": token, "user": {"id": user["user_id"], "name": user["name"], "email": user["email"], "role": user["role"]}}

@api_router.post("/favorites/{user_id}/{property_id}")
async def add_favorite(user_id: str, property_id: str):
    await db.users.update_one({"user_id": user_id}, {"$addToSet": {"favorites": property_id}})
    return {"message": "Added"}

@api_router.delete("/favorites/{user_id}/{property_id}")
async def remove_favorite(user_id: str, property_id: str):
    await db.users.update_one({"user_id": user_id}, {"$pull": {"favorites": property_id}})
    return {"message": "Removed"}

@api_router.get("/favorites/{user_id}", response_model=List[dict])
async def get_favorites(user_id: str):
    user = await db.users.find_one({"user_id": user_id})
    if not user or not user.get("favorites"): return []
    return await db.properties.find({"property_id": {"$in": user["favorites"]}}, {"_id": 0}).to_list(50)

@api_router.post("/upload/image", response_model=dict)
async def upload_image(file: UploadFile = File(...)):
    allowed = {"jpg", "jpeg", "png", "webp", "gif"}
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else "jpg"
    if ext not in allowed:
        raise HTTPException(400, "Only image files allowed (jpg, jpeg, png, webp)")
    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(400, "File too large (max 10MB)")
    filename = f"{uuid.uuid4()}.{ext}"
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    with open(f"{UPLOADS_DIR}/{filename}", "wb") as f:
        f.write(content)
    return {"url": f"/api/uploads/{filename}", "filename": filename}

@api_router.get("/uploads/{filename}")
async def serve_upload(filename: str):
    filepath = f"{UPLOADS_DIR}/{filename}"
    if not os.path.exists(filepath):
        raise HTTPException(404, "File not found")
    return FileResponse(filepath)

@api_router.post("/user-listings", response_model=dict)
async def create_user_listing(listing: UserListingCreate):
    doc = {
        **listing.model_dump(),
        "listing_id": str(uuid.uuid4()),
        "approval_status": "pending",
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    await db.user_listings.insert_one(doc)
    doc.pop("_id", None)
    return doc

@api_router.get("/user-listings", response_model=List[dict])
async def get_user_listings(status: Optional[str] = None):
    query = {}
    if status: query["approval_status"] = status
    cursor = db.user_listings.find(query, {"_id": 0}).sort("created_at", -1)
    return await cursor.to_list(100)

@api_router.put("/user-listings/{listing_id}/approve", response_model=dict)
async def approve_user_listing(listing_id: str):
    listing = await db.user_listings.find_one({"listing_id": listing_id})
    if not listing:
        raise HTTPException(404, "Listing not found")
    prop_doc = {
        "property_id": str(uuid.uuid4()),
        "title": listing["title"], "type": listing["type"],
        "status": listing["status"], "price": listing["price"],
        "price_unit": listing["price_unit"], "area": listing["area"],
        "area_unit": listing["area_unit"], "bedrooms": listing.get("bedrooms"),
        "bathrooms": listing.get("bathrooms"), "location": listing["location"],
        "pincode": listing.get("pincode", "585101"), "address": listing["address"],
        "description": listing["description"], "images": listing.get("images", []),
        "amenities": listing.get("amenities", []), "is_featured": False,
        "is_new_launch": False, "ready_to_move": True,
        "agent_name": listing.get("submitter_name", "Owner"),
        "agent_phone": listing.get("submitter_phone", "+91 9110278059"),
        "views": 0, "created_at": datetime.now(timezone.utc).isoformat()
    }
    await db.properties.insert_one(prop_doc)
    await db.user_listings.update_one({"listing_id": listing_id}, {"$set": {"approval_status": "approved"}})
    return {"message": "Approved and published as property"}

@api_router.put("/user-listings/{listing_id}/reject", response_model=dict)
async def reject_user_listing(listing_id: str, data: UserListingReject):
    await db.user_listings.update_one(
        {"listing_id": listing_id},
        {"$set": {"approval_status": "rejected", "rejection_reason": data.rejection_reason}}
    )
    return {"message": "Listing rejected"}

@api_router.get("/admin/dashboard", response_model=dict)
async def admin_dashboard():
    stats = await get_stats()
    recent_leads = await db.leads.find({}, {"_id": 0}).sort("created_at", -1).limit(10).to_list(10)
    return {"stats": stats, "recent_leads": recent_leads}
SAMPLE_PROPERTIES = [
    {"title": "Luxury 3 BHK Apartment - Sedam Road", "type": "residential", "status": "sale", "price": 45.0, "price_unit": "lakhs", "area": 1450.0, "area_unit": "sqft", "bedrooms": 3, "bathrooms": 2, "location": "Sedam Road", "pincode": "585101", "address": "Sedam Road, Near District Hospital, Kalaburagi", "description": "Spacious 3 BHK apartment with premium interiors, modular kitchen, and covered parking.", "images": ["https://images.unsplash.com/photo-1663672937496-f53fedcacf66?w=800&q=80"], "amenities": ["Power Backup", "Covered Parking", "Security", "Lift", "Gym"], "is_featured": True, "is_new_launch": False, "ready_to_move": True, "agent_name": "Kalaburagi Estates", "agent_phone": "+91 9110278059", "views": 125},
    {"title": "Prime Open Plot - Super Market Area", "type": "plot", "status": "sale", "price": 12.0, "price_unit": "lakhs", "area": 1200.0, "area_unit": "sqft", "bedrooms": None, "bathrooms": None, "location": "Super Market Area", "pincode": "585101", "address": "Super Market Area, Main Road, Kalaburagi", "description": "Excellent prime residential plot near Super Market. All government approvals done.", "images": ["https://images.unsplash.com/photo-1757924432508-d4e92411caeb?w=800&q=80"], "amenities": ["Road Facility", "Water Supply", "Electricity", "Drainage"], "is_featured": True, "is_new_launch": False, "ready_to_move": True, "agent_name": "Kalaburagi Estates", "agent_phone": "+91 9110278059", "views": 89},
    {"title": "Commercial Building - Main Road", "type": "commercial", "status": "sale", "price": 1.2, "price_unit": "crores", "area": 5000.0, "area_unit": "sqft", "bedrooms": None, "bathrooms": None, "location": "Gulbarga Main Road", "pincode": "585101", "address": "Main Road, Near Bus Stand, Kalaburagi", "description": "G+3 commercial building ideal for offices, showrooms or retail.", "images": ["https://images.unsplash.com/photo-1767950470198-c9cd97f8ed87?w=800&q=80"], "amenities": ["Lift", "Parking", "Security", "Power Backup", "Reception"], "is_featured": True, "is_new_launch": False, "ready_to_move": True, "agent_name": "Kalaburagi Estates", "agent_phone": "+91 9110278059", "views": 210},
    {"title": "2 BHK New Apartment - Kapnoor", "type": "residential", "status": "sale", "price": 28.0, "price_unit": "lakhs", "area": 950.0, "area_unit": "sqft", "bedrooms": 2, "bathrooms": 2, "location": "Kapnoor", "pincode": "585105", "address": "Kapnoor Layout, Near Park, Kalaburagi", "description": "Well-designed 2 BHK apartment in peaceful locality.", "images": ["https://images.pexels.com/photos/5644678/pexels-photo-5644678.jpeg?w=800"], "amenities": ["Security", "Children Play Area", "Garden", "Parking"], "is_featured": False, "is_new_launch": True, "ready_to_move": False, "agent_name": "Kalaburagi Estates", "agent_phone": "+91 9110278059", "views": 67},
    {"title": "Premium Villa - Ashraya Colony", "type": "residential", "status": "sale", "price": 85.0, "price_unit": "lakhs", "area": 2400.0, "area_unit": "sqft", "bedrooms": 4, "bathrooms": 3, "location": "Ashraya Colony", "pincode": "585101", "address": "Ashraya Colony, Kalaburagi", "description": "Stunning 4 BHK independent villa with premium finishes.", "images": ["https://images.unsplash.com/photo-1768463852120-9360d0e39912?w=800&q=80"], "amenities": ["Private Garden", "2 Parking", "Security", "Power Backup", "Modular Kitchen"], "is_featured": True, "is_new_launch": False, "ready_to_move": True, "agent_name": "Kalaburagi Estates", "agent_phone": "+91 9110278059", "views": 178},
    {"title": "Shop Space for Rent - Station Area", "type": "commercial", "status": "rent", "price": 25000.0, "price_unit": "month", "area": 800.0, "area_unit": "sqft", "bedrooms": None, "bathrooms": None, "location": "Station Area", "pincode": "585101", "address": "Railway Station Road, Kalaburagi", "description": "Prime commercial shop space on Station Road with high footfall.", "images": ["https://images.pexels.com/photos/35171247/pexels-photo-35171247.jpeg?w=800"], "amenities": ["Power Connection", "24x7 Security", "CCTV"], "is_featured": False, "is_new_launch": False, "ready_to_move": True, "agent_name": "Kalaburagi Estates", "agent_phone": "+91 9110278059", "views": 93},
    {"title": "New Gated Layout - Bidar Road", "type": "layout", "status": "upcoming", "price": 6.5, "price_unit": "lakhs", "area": 1200.0, "area_unit": "sqft", "bedrooms": None, "bathrooms": None, "location": "Bidar Road", "pincode": "585101", "address": "Bidar Road, 8km from City Center, Kalaburagi", "description": "Premium gated layout with 100+ plots. Wide BT roads, parks, underground drainage.", "images": ["https://images.unsplash.com/photo-1663672937496-f53fedcacf66?w=800&q=80"], "amenities": ["Gated Community", "Wide Roads", "Park", "Underground Drainage", "Street Lights"], "is_featured": True, "is_new_launch": True, "ready_to_move": False, "agent_name": "Kalaburagi Estates", "agent_phone": "+91 9110278059", "views": 312},
    {"title": "3 BHK Flat for Rent - Aland Road", "type": "rental", "status": "rent", "price": 15000.0, "price_unit": "month", "area": 1200.0, "area_unit": "sqft", "bedrooms": 3, "bathrooms": 2, "location": "Aland Road", "pincode": "585101", "address": "Aland Road, Near School, Kalaburagi", "description": "Well-maintained 3 BHK flat in prime residential area.", "images": ["https://images.pexels.com/photos/5644678/pexels-photo-5644678.jpeg?w=800"], "amenities": ["Semi-Furnished", "Parking", "Security", "Water Supply"], "is_featured": False, "is_new_launch": False, "ready_to_move": True, "agent_name": "Kalaburagi Estates", "agent_phone": "+91 9110278059", "views": 56},
    {"title": "4 BHK Penthouse - New Town", "type": "residential", "status": "sale", "price": 95.0, "price_unit": "lakhs", "area": 3200.0, "area_unit": "sqft", "bedrooms": 4, "bathrooms": 4, "location": "New Town", "pincode": "585101", "address": "New Town Development Area, Kalaburagi", "description": "Exclusive top-floor penthouse with panoramic views of Kalaburagi.", "images": ["https://images.unsplash.com/photo-1767950470198-c9cd97f8ed87?w=800&q=80"], "amenities": ["Private Terrace", "Smart Home", "3 Parking", "24x7 Security", "Jacuzzi"], "is_featured": True, "is_new_launch": True, "ready_to_move": False, "agent_name": "Kalaburagi Estates", "agent_phone": "+91 9110278059", "views": 445},
    {"title": "Farm Land - Shahabad Road", "type": "farm", "status": "sale", "price": 8.0, "price_unit": "lakhs", "area": 1.0, "area_unit": "acres", "bedrooms": None, "bathrooms": None, "location": "Shahabad", "pincode": "585228", "address": "Shahabad Road, 15km from Kalaburagi", "description": "Fertile agricultural land with water source. Good road connectivity.", "images": ["https://images.unsplash.com/photo-1757924432508-d4e92411caeb?w=800&q=80"], "amenities": ["Water Source", "Road Access", "Electricity Nearby"], "is_featured": False, "is_new_launch": False, "ready_to_move": True, "agent_name": "Kalaburagi Estates", "agent_phone": "+91 9110278059", "views": 45},
]

async def seed_db():
    if await db.properties.count_documents({}) == 0:
        props = [{"property_id": str(uuid.uuid4()), **p, "created_at": datetime.now(timezone.utc).isoformat()} for p in SAMPLE_PROPERTIES]
        await db.properties.insert_many(props)
        logger.info(f"Seeded {len(props)} properties")
    if not await db.users.find_one({"email": "admin@kalaburagistates.com"}):
        await db.users.insert_one({
            "user_id": str(uuid.uuid4()), "name": "Admin", "email": "admin@kalaburagistates.com",
            "phone": "+91 9110278059", "password_hash": hash_password("Admin@1234"),
            "role": "admin", "favorites": [], "created_at": datetime.now(timezone.utc).isoformat()
        })
        logger.info("Admin user created")

app.include_router(api_router)
app.add_middleware(
    CORSMiddleware, allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"], allow_headers=["*"],
)

@app.on_event("startup")
async def startup():
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    await seed_db()

@app.on_event("shutdown")
async def shutdown():
    client.close()
