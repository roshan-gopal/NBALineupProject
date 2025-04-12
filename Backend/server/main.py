from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api.routes import lineup_routes

app = FastAPI(
    title="NBA Lineup Optimizer API",
    description="API for optimizing NBA team lineups using machine learning",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(lineup_routes.router, prefix="/api", tags=["lineups"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 