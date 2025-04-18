from fastapi import APIRouter

router = APIRouter(
    tags=["misc"]
)

@router.get("/ping")
async def ping():
    """Simple endpoint to check if the service is running."""
    return {"message": "Data service is running!"} 