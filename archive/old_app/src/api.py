from fastapi import APIRouter
from .inference.views import router as inference_router
api_router_v1 = APIRouter()

api_router_v1.include_router(inference_router, prefix="/inference", tags=["inference"])
api_router_root = APIRouter()

@api_router_root.get("/", status_code=200)
def read_root():
    return "This is AWESOME!"