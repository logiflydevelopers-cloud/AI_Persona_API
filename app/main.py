from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
import asyncio

from app.api.route_chats import router
from app.services.rag_services import pinecone_index  # import your index

app = FastAPI(title="AI Engine API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


# ===============================
# Pinecone Keep-Alive Task
# ===============================

async def keep_pinecone_warm():
    dummy_vec = [0.0] * 1536  # text-embedding-3-small dimension

    while True:
        try:
            await run_in_threadpool(
                pinecone_index.query,
                vector=dummy_vec,
                top_k=1,
                namespace="__warmup__",
                include_metadata=False
            )
            print("🔁 Pinecone heartbeat sent")
        except Exception as e:
            print("Warmup failed:", e)

        await asyncio.sleep(300)  # every 5 minutes


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(keep_pinecone_warm())