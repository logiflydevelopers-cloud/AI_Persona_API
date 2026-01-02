from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.route_chats import router

app = FastAPI(title="AI Engine API", version="1.0.0")

app.include_router(router)

