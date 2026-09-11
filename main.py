from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app=FastAPI(
	title="traffic dectector",
	version="1.0.0"
)

origin=[
"http://localhost:5173",
"http://127.0.0.1:5173"
]


app.add_middleware(
	CORSMiddleware,
	allow_origins=origin,
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"]
)


@app.get("/")
def root():
    return {"message": "Traffic Detector API is running with CORS enabled"}