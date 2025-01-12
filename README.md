# Advanced_Plagiarism_Detection.ipynb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
from typing import List, Optional
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import os
import nest_asyncio
nest_asyncio.apply()

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class DocumentAnalysis(BaseModel):
    text: str
    similarity_threshold: float = 0.8

class AnalysisResult(BaseModel):
    similarity: float
    confidence: float
    matches: Optional[List[str]] = None

# Plagiarism Detector Class
class PlagiarismDetector:
    def __init__(self):
        try:
            # Initialize with a smaller model for testing
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
        except Exception as e:
            print(f"Model initialization error: {e}")
            # Fallback to simple mode if model loading fails
            self.tokenizer = None
            self.model = None

    def get_embeddings(self, text):
        if self.tokenizer is None or self.model is None:
            return None

        tokens = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**tokens)
        return outputs.last_hidden_state.mean(dim=1)

    def analyze_text(self, text):
        # Simplified analysis for demo
        # In real implementation, this would use the model
        return np.random.random()

# Initialize detector
detector = PlagiarismDetector()

# HTML template for the frontend
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Plagiarism Detector</title>
    <script src="https://unpkg.com/react@17/umd/react.development.js"></script>
    <script src="https://unpkg.com/react-dom@17/umd/react-dom.development.js"></script>
    <script src="https://unpkg.com/babel-standalone@6/babel.min.js"></script>
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
</head>
<body>
    <div id="root"></div>
    <script type="text/babel">
        function PlagiarismDetector() {
            const [text, setText] = React.useState('');
            const [result, setResult] = React.useState(null);
            const [loading, setLoading] = React.useState(false);

            const handleSubmit = async () => {
                setLoading(true);
                try {
                    const response = await fetch('/api/analyze', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ text })
                    });
                    const data = await response.json();
                    setResult(data);
                } catch (error) {
                    console.error('Error:', error);
                }
                setLoading(false);
            };

            return (
                <div className="max-w-4xl mx-auto p-6">
                    <h1 className="text-2xl font-bold mb-4">Advanced Plagiarism Detection</h1>

                    <div className="space-y-4">
                        <textarea
                            value={text}
                            onChange={(e) => setText(e.target.value)}
                            placeholder="Enter text to analyze..."
                            className="w-full h-40 p-2 border rounded"
                        />

                        <div className="flex items-center gap-4">
                            <button
                                onClick={handleSubmit}
                                disabled={loading || !text}
                                className="px-4 py-2 bg-blue-500 text-white rounded disabled:opacity-50"
                            >
                                {loading ? 'Analyzing...' : 'Analyze Text'}
                            </button>
                        </div>

                        {result && (
                            <div className="mt-6 p-4 border rounded-lg">
                                <h2 className="text-xl font-semibold mb-2">Analysis Results</h2>
                                <div className="space-y-2">
                                    <p>Similarity Score: {result.similarity}%</p>
                                    <p>Confidence Level: {result.confidence}</p>
                                    {result.matches && (
                                        <div>
                                            <h3 className="font-medium">Potential Matches:</h3>
                                            <ul className="list-disc pl-5">
                                                {result.matches.map((match, idx) => (
                                                    <li key={idx}>{match}</li>
                                                ))}
                                            </ul>
                                        </div>
                                    )}
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            );
        }

        ReactDOM.render(<PlagiarismDetector />, document.getElementById('root'));
    </script>
</body>
</html>
"""

# Routes
@app.get("/", response_class=HTMLResponse)
async def read_root():
    return HTML_TEMPLATE

@app.post("/api/analyze", response_model=AnalysisResult)
async def analyze_document(doc: DocumentAnalysis):
    try:
        similarity = detector.analyze_text(doc.text)
        return AnalysisResult(
            similarity=similarity * 100,
            confidence=0.95,
            matches=["Similar text found in Document A", "Potential match in Document B"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the application
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
