from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.route_chats import router

app = FastAPI(title="AI Engine API", version="1.0.0")

# if your frontend is on another domain, configure this properly:
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # replace with your frontend domain(s) in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
