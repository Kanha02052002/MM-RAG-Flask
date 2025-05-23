from flask import Flask, render_template_string, request, jsonify, session
import requests
import base64
import json
import uuid
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import os
import tempfile
from PIL import Image
import io
from werkzeug.utils import secure_filename
import threading
import time

# Flask app configuration
app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global variables for shared resources
embedding_model = None
chroma_client = None
model_loading_status = {"status": "not_started", "progress": 0}

@dataclass
class Config:
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    VISION_MODEL = "google/gemini-pro-vision"
    TEXT_MODEL = "anthropic/claude-3-sonnet"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    CHROMA_COLLECTION = "openrouter_responses"

# Embedding and Vector Store Classes
class HuggingFaceEmbedding:
    def __init__(self, model_name: str = Config.EMBEDDING_MODEL):
        global model_loading_status
        model_loading_status["status"] = "loading"
        model_loading_status["progress"] = 20
        try:
            self.model = SentenceTransformer(model_name)
            model_loading_status["status"] = "completed"
            model_loading_status["progress"] = 100
        except Exception as e:
            model_loading_status["status"] = "error"
            model_loading_status["error"] = str(e)
            self.model = None
    
    def embed_text(self, text: str) -> List[float]:
        if self.model is None:
            return [0.0] * 384
        try:
            embedding = self.model.encode(text, convert_to_tensor=False)
            return embedding.tolist()
        except Exception as e:
            return [0.0] * 384

class ChromaVectorStore:
    def __init__(self, collection_name: str = Config.CHROMA_COLLECTION):
        try:
            self.client = chromadb.Client(Settings(
                allow_reset=True,
                anonymized_telemetry=False
            ))
            self.collection_name = collection_name
            self.collection = None
            self._initialize_collection()
        except Exception as e:
            self.client = None
    
    def _initialize_collection(self):
        try:
            if self.client:
                self.collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"description": "OpenRouter API responses"}
                )
        except Exception as e:
            pass
    
    def add_document(self, text: str, metadata: Dict[str, Any], embedding: List[float]):
        if not self.collection:
            return False
        try:
            doc_id = str(uuid.uuid4())
            self.collection.add(
                documents=[text],
                embeddings=[embedding],
                metadatas=[metadata],
                ids=[doc_id]
            )
            return True
        except Exception as e:
            return False
    
    def search(self, query_embedding: List[float], n_results: int = 5) -> List[Dict]:
        if not self.collection:
            return []
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            return results
        except Exception as e:
            return []
    
    def get_count(self):
        if not self.collection:
            return 0
        try:
            return self.collection.count()
        except:
            return 0
    
    def reset(self):
        try:
            if self.client:
                self.client.reset()
                self._initialize_collection()
                return True
        except:
            return False

class OpenRouterClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = Config.OPENROUTER_BASE_URL
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://flask-app.local",
            "X-Title": "Flask OpenRouter App"
        }
    
    def encode_image(self, image_file) -> str:
        try:
            if hasattr(image_file, 'read'):
                image_file.seek(0)
                return base64.b64encode(image_file.read()).decode('utf-8')
            else:
                with open(image_file, "rb") as f:
                    return base64.b64encode(f.read()).decode('utf-8')
        except Exception as e:
            return ""
    
    def vision_request(self, prompt: str, image_file, model: str = Config.VISION_MODEL) -> Dict:
        try:
            base64_image = self.encode_image(image_file)
            if not base64_image:
                return {"error": "Failed to encode image"}
            
            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 1000
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=60
            )
            
            return response.json()
        except Exception as e:
            return {"error": f"Vision API request failed: {str(e)}"}
    
    def text_request(self, prompt: str, model: str = Config.TEXT_MODEL, structured: bool = False) -> Dict:
        try:
            messages = [{"role": "user", "content": prompt}]
            
            if structured:
                structured_prompt = f"""
                Please provide a structured response to the following query in JSON format:
                
                {{
                    "summary": "Brief summary of the response",
                    "main_points": ["Key point 1", "Key point 2", "..."],
                    "detailed_response": "Detailed explanation",
                    "confidence_level": "High/Medium/Low",
                    "sources_needed": ["Any sources that would be helpful"],
                    "follow_up_questions": ["Relevant follow-up questions"]
                }}
                
                Query: {prompt}
                """
                messages[0]["content"] = structured_prompt
            
            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": 1500,
                "temperature": 0.7
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=60
            )
            
            return response.json()
        except Exception as e:
            return {"error": f"Text API request failed: {str(e)}"}

def parse_structured_response(response_text: str) -> Dict:
    try:
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        
        if start_idx != -1 and end_idx != -1:
            json_str = response_text[start_idx:end_idx]
            return json.loads(json_str)
        else:
            return {
                "summary": "Response parsing failed",
                "main_points": [],
                "detailed_response": response_text,
                "confidence_level": "Unknown",
                "sources_needed": [],
                "follow_up_questions": []
            }
    except Exception as e:
        return {
            "summary": "JSON parsing failed",
            "main_points": [],
            "detailed_response": response_text,
            "confidence_level": "Unknown",
            "sources_needed": [],
            "follow_up_questions": [],
            "parsing_error": str(e)
        }

def init_models():
    global embedding_model, chroma_client
    def load_models():
        global embedding_model, chroma_client, model_loading_status
        model_loading_status["status"] = "loading"
        model_loading_status["progress"] = 10
        
        embedding_model = HuggingFaceEmbedding()
        model_loading_status["progress"] = 70
        
        chroma_client = ChromaVectorStore()
        model_loading_status["progress"] = 100
        model_loading_status["status"] = "completed"
    
    thread = threading.Thread(target=load_models)
    thread.daemon = True
    thread.start()

# Routes
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/init', methods=['POST'])
def init_components():
    global embedding_model, chroma_client
    if embedding_model is None or chroma_client is None:
        init_models()
    return jsonify({"status": "initializing"})

@app.route('/api/status')
def get_status():
    global model_loading_status, embedding_model, chroma_client
    status = model_loading_status.copy()
    
    if embedding_model and chroma_client:
        status["components_ready"] = True
        status["vector_count"] = chroma_client.get_count()
    else:
        status["components_ready"] = False
        status["vector_count"] = 0
    
    return jsonify(status)

@app.route('/api/chat', methods=['POST'])
def chat():
    global embedding_model, chroma_client
    
    data = request.get_json()
    api_key = data.get('api_key')
    prompt = data.get('prompt')
    model = data.get('model', Config.TEXT_MODEL)
    structured = data.get('structured', False)
    save_to_vector = data.get('save_to_vector', True)
    
    if not api_key or not prompt:
        return jsonify({"error": "Missing API key or prompt"}), 400
    
    try:
        client = OpenRouterClient(api_key)
        response = client.text_request(prompt, model, structured)
        
        if "error" in response:
            return jsonify(response), 400
        
        content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        result = {"content": content}
        
        if structured:
            result["structured"] = parse_structured_response(content)
        
        # Save to vector store
        if save_to_vector and embedding_model and chroma_client:
            try:
                embedding = embedding_model.embed_text(content)
                metadata = {
                    "type": "text_response",
                    "model": model,
                    "prompt": prompt,
                    "timestamp": datetime.now().isoformat(),
                    "structured": structured
                }
                
                success = chroma_client.add_document(content, metadata, embedding)
                result["saved_to_vector"] = success
            except Exception as e:
                result["vector_error"] = str(e)
        
        # Save to session history
        if 'history' not in session:
            session['history'] = []
        
        session['history'].append({
            "type": "text",
            "prompt": prompt,
            "response": content,
            "timestamp": datetime.now().isoformat(),
            "model": model
        })
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/vision', methods=['POST'])
def vision_analysis():
    global embedding_model, chroma_client
    
    api_key = request.form.get('api_key')
    prompt = request.form.get('prompt')
    model = request.form.get('model', Config.VISION_MODEL)
    save_to_vector = request.form.get('save_to_vector') == 'true'
    
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No image selected"}), 400
    
    if not api_key or not prompt:
        return jsonify({"error": "Missing API key or prompt"}), 400
    
    try:
        client = OpenRouterClient(api_key)
        response = client.vision_request(prompt, file, model)
        
        if "error" in response:
            return jsonify(response), 400
        
        content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        result = {"content": content}
        
        # Save to vector store
        if save_to_vector and embedding_model and chroma_client:
            try:
                embedding = embedding_model.embed_text(content)
                metadata = {
                    "type": "vision_response",
                    "model": model,
                    "prompt": prompt,
                    "timestamp": datetime.now().isoformat(),
                    "image_name": secure_filename(file.filename)
                }
                
                success = chroma_client.add_document(content, metadata, embedding)
                result["saved_to_vector"] = success
            except Exception as e:
                result["vector_error"] = str(e)
        
        # Save to session history
        if 'history' not in session:
            session['history'] = []
        
        session['history'].append({
            "type": "vision",
            "prompt": prompt,
            "response": content,
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "image_name": secure_filename(file.filename)
        })
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/search', methods=['POST'])
def vector_search():
    global embedding_model, chroma_client
    
    if not embedding_model or not chroma_client:
        return jsonify({"error": "Components not initialized"}), 400
    
    data = request.get_json()
    query = data.get('query')
    num_results = data.get('num_results', 5)
    
    if not query:
        return jsonify({"error": "Missing query"}), 400
    
    try:
        query_embedding = embedding_model.embed_text(query)
        results = chroma_client.search(query_embedding, num_results)
        
        formatted_results = []
        if results and 'documents' in results and results['documents'][0]:
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )):
                formatted_results.append({
                    "content": doc,
                    "metadata": metadata,
                    "similarity": 1 - distance,
                    "rank": i + 1
                })
        
        return jsonify({"results": formatted_results})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/history')
def get_history():
    history = session.get('history', [])
    return jsonify({"history": history})

@app.route('/api/clear-history', methods=['POST'])
def clear_history():
    session['history'] = []
    return jsonify({"status": "cleared"})

@app.route('/api/clear-vector', methods=['POST'])
def clear_vector():
    global chroma_client
    if chroma_client:
        success = chroma_client.reset()
        return jsonify({"status": "cleared" if success else "error"})
    return jsonify({"error": "Vector store not initialized"}), 400

# HTML Template with Modern UI
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OpenRouter Vision & Vector Store</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://unpkg.com/alpinejs@3.x.x/dist/cdn.min.js" defer></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        body { font-family: 'Inter', sans-serif; }
        
        .gradient-bg {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        
        .glass-effect {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .dark-glass {
            background: rgba(0, 0, 0, 0.3);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .glow-effect {
            box-shadow: 0 0 20px rgba(102, 126, 234, 0.3);
        }
        
        .animate-pulse-slow {
            animation: pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite;
        }
        
        .custom-scrollbar::-webkit-scrollbar {
            width: 6px;
        }
        
        .custom-scrollbar::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 3px;
        }
        
        .custom-scrollbar::-webkit-scrollbar-thumb {
            background: rgba(102, 126, 234, 0.5);
            border-radius: 3px;
        }
        
        .typing-animation {
            animation: typing 1s infinite;
        }
        
        @keyframes typing {
            0%, 50% { opacity: 1; }
            51%, 100% { opacity: 0.5; }
        }
    </style>
</head>
<body class="bg-gray-900 text-white min-h-screen">
    <div x-data="appData()" x-init="init()" class="flex h-screen">
        <!-- Sidebar -->
        <div class="w-80 dark-glass border-r border-gray-700 flex flex-col">
            <!-- Header -->
            <div class="p-6 border-b border-gray-700">
                <h1 class="text-2xl font-bold gradient-text">
                    <i class="fas fa-robot mr-2"></i>
                    OpenRouter AI
                </h1>
                <p class="text-gray-400 text-sm mt-1">Vision & Vector Store</p>
            </div>
            
            <!-- API Key -->
            <div class="p-4 border-b border-gray-700">
                <label class="block text-sm font-medium mb-2">
                    <i class="fas fa-key mr-1"></i>
                    API Key
                </label>
                <input 
                    type="password" 
                    x-model="apiKey" 
                    placeholder="Enter OpenRouter API key"
                    class="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm"
                >
                <button 
                    @click="initComponents()"
                    :disabled="!apiKey || status.status === 'loading'"
                    class="w-full mt-2 px-4 py-2 bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-lg hover:from-blue-600 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 text-sm font-medium"
                >
                    <span x-show="status.status !== 'loading'">
                        <i class="fas fa-cog mr-1"></i>
                        Initialize Components
                    </span>
                    <span x-show="status.status === 'loading'" class="flex items-center justify-center">
                        <i class="fas fa-spinner fa-spin mr-2"></i>
                        Loading... <span x-text="status.progress + '%'"></span>
                    </span>
                </button>
            </div>
            
            <!-- Status -->
            <div class="p-4 border-b border-gray-700">
                <div class="flex items-center justify-between mb-2">
                    <span class="text-sm font-medium">Status</span>
                    <span 
                        :class="{
                            'text-green-400': status.components_ready,
                            'text-yellow-400': status.status === 'loading',
                            'text-red-400': status.status === 'error',
                            'text-gray-400': status.status === 'not_started'
                        }"
                        class="text-xs"
                    >
                        <i 
                            :class="{
                                'fas fa-check-circle': status.components_ready,
                                'fas fa-spinner fa-spin': status.status === 'loading',
                                'fas fa-exclamation-circle': status.status === 'error',
                                'fas fa-circle': status.status === 'not_started'
                            }"
                            class="mr-1"
                        ></i>
                        <span x-text="status.components_ready ? 'Ready' : (status.status === 'loading' ? 'Loading' : (status.status === 'error' ? 'Error' : 'Not Started'))"></span>
                    </span>
                </div>
                
                <div class="bg-gray-800 rounded-lg p-3">
                    <div class="flex justify-between text-xs text-gray-400 mb-1">
                        <span>Vector Store</span>
                        <span x-text="status.vector_count + ' docs'"></span>
                    </div>
                    <div class="w-full bg-gray-700 rounded-full h-1">
                        <div 
                            class="bg-gradient-to-r from-blue-500 to-purple-600 h-1 rounded-full transition-all duration-500"
                            :style="`width: ${Math.min(status.vector_count * 2, 100)}%`"
                        ></div>
                    </div>
                </div>
            </div>
            
            <!-- Navigation -->
            <div class="p-4 border-b border-gray-700">
                <nav class="space-y-2">
                    <button 
                        @click="activeTab = 'chat'"
                        :class="activeTab === 'chat' ? 'bg-blue-600' : 'hover:bg-gray-700'"
                        class="w-full flex items-center px-3 py-2 rounded-lg transition-colors text-left"
                    >
                        <i class="fas fa-comments mr-3"></i>
                        Text Chat
                    </button>
                    <button 
                        @click="activeTab = 'vision'"
                        :class="activeTab === 'vision' ? 'bg-blue-600' : 'hover:bg-gray-700'"
                        class="w-full flex items-center px-3 py-2 rounded-lg transition-colors text-left"
                    >
                        <i class="fas fa-eye mr-3"></i>
                        Vision Analysis
                    </button>
                    <button 
                        @click="activeTab = 'search'"
                        :class="activeTab === 'search' ? 'bg-blue-600' : 'hover:bg-gray-700'"
                        class="w-full flex items-center px-3 py-2 rounded-lg transition-colors text-left"
                    >
                        <i class="fas fa-search mr-3"></i>
                        Vector Search
                    </button>
                    <button 
                        @click="activeTab = 'history'"
                        :class="activeTab === 'history' ? 'bg-blue-600' : 'hover:bg-gray-700'"
                        class="w-full flex items-center px-3 py-2 rounded-lg transition-colors text-left"
                    >
                        <i class="fas fa-history mr-3"></i>
                        History
                    </button>
                </nav>
            </div>
            
            <!-- Actions -->
            <div class="p-4 mt-auto">
                <button 
                    @click="clearVector()"
                    class="w-full px-3 py-2 bg-red-600 hover:bg-red-700 rounded-lg transition-colors text-sm mb-2"
                >
                    <i class="fas fa-trash mr-2"></i>
                    Clear Vector Store
                </button>
                <button 
                    @click="clearHistory()"
                    class="w-full px-3 py-2 bg-gray-600 hover:bg-gray-700 rounded-lg transition-colors text-sm"
                >
                    <i class="fas fa-broom mr-2"></i>
                    Clear History
                </button>
            </div>
        </div>
        
        <!-- Main Content -->
        <div class="flex-1 flex flex-col">
            <!-- Chat Tab -->
            <div x-show="activeTab === 'chat'" class="flex-1 flex flex-col">
                <!-- Header -->
                <div class="border-b border-gray-700 p-4">
                    <h2 class="text-xl font-semibold">Text Chat</h2>
                    <p class="text-gray-400 text-sm">Chat with AI models and get structured responses</p>
                </div>
                
                <!-- Messages -->
                <div class="flex-1 overflow-y-auto custom-scrollbar p-4" x-ref="chatContainer">
                    <div class="space-y-4">
                        <template x-for="message in chatMessages" :key="message.id">
                            <div class="flex" :class="message.type === 'user' ? 'justify-end' : 'justify-start'">
                                <div 
                                    class="max-w-3xl rounded-2xl px-4 py-3"
                                    :class="message.type === 'user' ? 'bg-blue-600' : 'dark-glass'"
                                >
                                    <div x-show="message.type === 'user'" class="text-right">
                                        <div class="font-medium text-sm mb-1">You</div>
                                        <div x-text="message.content"></div>
                                    </div>
                                    
                                    <div x-show="message.type === 'ai'" class="text-left">
                                        <div class="font-medium text-sm mb-2 flex items-center">
                                            <i class="fas fa-robot mr-2"></i>
                                            AI Assistant
                                            <span x-show="message.loading" class="ml-2 typing-animation">●●●</span>
                                        </div>
                                        
                                        <div x-show="!message.structured" x-text="message.content" class="whitespace-pre-wrap"></div>
                                        
                                        <div x-show="message.structured && message.structuredData">
                                            <div class="bg-gray-800 rounded-lg p-3 mb-3">
                                                <div class="text-sm font-medium text-blue-400 mb-2">Summary</div>
                                                <div x-text="message.structuredData?.summary" class="text-sm"></div>
                                            </div>
                                            
                                            <div class="bg-gray-800 rounded-lg p-3 mb-3" x-show="message.structuredData?.main_points?.length">
                                                <div class="text-sm font-medium text-green-400 mb-2">Key Points</div>
                                                <ul class="text-sm space-y-1">
                                                    <template x-for="point in message.structuredData?.main_points">
                                                        <li class="flex items-start">
                                                            <i class="fas fa-chevron-right text-green-400 mr-2 mt-1 text-xs"></i>
                                                            <span x-text="point"></span>
                                                        </li>
                                                    </template>
                                                </ul>
                                            </div>
                                            
                                            <div class="bg-gray-800 rounded-lg p-3 mb-3">
                                                <div class="text-sm font-medium text-yellow-400 mb-2">Detailed Response</div>
                                                <div x-text="message.structuredData?.detailed_response" class="text-sm whitespace-pre-wrap"></div>
                                            </div>
                                            
                                            <div class="flex items-center justify-between text-xs text-gray-400">
                                                <span>
                                                    Confidence: 
                                                    <span 
                                                        :class="{
                                                            'text-green-400': message.structuredData?.confidence_level === 'High',
                                                            'text-yellow-400': message.structuredData?.confidence_level === 'Medium',
                                                            'text-red-400': message.structuredData?.confidence_level === 'Low'
                                                        }"
                                                        x-text="message.structuredData?.confidence_level"
                                                    ></span>
                                                </span>
                                                <span x-show="message.saved_to_vector" class="text-green-400">
                                                    <i class="fas fa-database mr-1"></i>Saved
                                                </span>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </template>
                    </div>
                </div>
                
                <!-- Input -->
                <div class="border-t border-gray-700 p-4">
                    <div class="flex space-x-4 mb-3">
                        <label class="flex items-center">
                            <input type="checkbox" x-model="structuredResponse" class="mr-2">
                            <span class="text-sm">Structured Response</span>
                        </label>
                        <label class="flex items-center">
                            <input type="checkbox" x-model="saveToVector" class="mr-2">
                            <span class="text-sm">Save to Vector Store</span>
                        </label>
                    </div>
                    
                    <div class="flex space-x-2">
                        <textarea 
                            x-model="chatInput"
                            @keydown.enter.prevent="if(!$event.shiftKey) sendMessage()"
                            placeholder="Type your message... (Shift+Enter for new line)"
                            class="flex-1 px-4 py-3 bg-gray-800 border border-gray-600 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
                            rows="2"
                        ></textarea>
                        <button 
                            @click="sendMessage()"
                            :disabled="!chatInput.trim() || !apiKey || chatLoading"
                            class="px-6 py-3 bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-xl hover:from-blue-600 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 font-medium"
                        >
                            <i class="fas fa-paper-plane" x-show="!chatLoading"></i>
                            <i class="fas fa-spinner fa-spin" x-show="chatLoading"></i>
                        </button>
                    </div>
                </div>
            </div>
            
            <!-- Vision Tab -->
            <div x-show="activeTab === 'vision'" class="flex-1 flex flex-col">
                <!-- Header -->
                <div class="border-b border-gray-700 p-4">
                    <h2 class="text-xl font-semibold">Vision Analysis</h2>
                    <p class="text-gray-400 text-sm">Upload images and get AI-powered analysis</p>
                </div>
                
                <!-- Content -->
                <div class="flex-1 overflow-y-auto custom-scrollbar p-4">
                    <!-- Image Upload -->
                    <div class="mb-6">
                        <div 
                            @drop.prevent="handleImageDrop($event)"
                            @dragover.prevent
                            @dragenter.prevent
                            class="border-2 border-dashed border-gray-600 rounded-xl p-8 text-center hover:border-blue-500 transition-colors"
                        >
                            <div x-show="!selectedImage">
                                <i class="fas fa-cloud-upload-alt text-4xl text-gray-400 mb-4"></i>
                                <p class="text-gray-400 mb-2">Drag and drop an image here, or</p>
                                <input 
                                    type="file" 
                                    @change="handleImageSelect($event)"
                                    accept="image/*"
                                    class="hidden"
                                    x-ref="imageInput"
                                >
                                <button 
                                    @click="$refs.imageInput.click()"
                                    class="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors"
                                >
                                    Browse Files
                                </button>
                            </div>
                            
                            <div x-show="selectedImage" class="space-y-4">
                                <img :src="selectedImage" alt="Selected image" class="max-w-full max-h-64 mx-auto rounded-lg">
                                <button 
                                    @click="clearImage()"
                                    class="px-4 py-2 bg-red-600 hover:bg-red-700 rounded-lg transition-colors"
                                >
                                    <i class="fas fa-times mr-2"></i>Remove Image
                                </button>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Vision Input -->
                    <div class="space-y-4">
                        <textarea 
                            x-model="visionPrompt"
                            placeholder="What would you like me to analyze about this image?"
                            class="w-full px-4 py-3 bg-gray-800 border border-gray-600 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
                            rows="3"
                        ></textarea>
                        
                        <div class="flex items-center justify-between">
                            <label class="flex items-center">
                                <input type="checkbox" x-model="saveVisionToVector" class="mr-2">
                                <span class="text-sm">Save analysis to Vector Store</span>
                            </label>
                            
                            <button 
                                @click="analyzeImage()"
                                :disabled="!selectedImage || !visionPrompt.trim() || !apiKey || visionLoading"
                                class="px-6 py-3 bg-gradient-to-r from-green-500 to-blue-600 text-white rounded-xl hover:from-green-600 hover:to-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 font-medium"
                            >
                                <i class="fas fa-eye" x-show="!visionLoading"></i>
                                <i class="fas fa-spinner fa-spin" x-show="visionLoading"></i>
                                <span class="ml-2">Analyze Image</span>
                            </button>
                        </div>
                    </div>
                    
                    <!-- Vision Results -->
                    <div x-show="visionResult" class="mt-6 dark-glass rounded-xl p-6">
                        <h3 class="text-lg font-semibold mb-4 flex items-center">
                            <i class="fas fa-search mr-2 text-green-400"></i>
                            Analysis Result
                        </h3>
                        <div x-text="visionResult" class="whitespace-pre-wrap text-gray-300"></div>
                    </div>
                </div>
            </div>
            
            <!-- Search Tab -->
            <div x-show="activeTab === 'search'" class="flex-1 flex flex-col">
                <!-- Header -->
                <div class="border-b border-gray-700 p-4">
                    <h2 class="text-xl font-semibold">Vector Search</h2>
                    <p class="text-gray-400 text-sm">Search through your stored conversations and analyses</p>
                </div>
                
                <!-- Content -->
                <div class="flex-1 overflow-y-auto custom-scrollbar p-4">
                    <!-- Search Input -->
                    <div class="mb-6">
                        <div class="flex space-x-2">
                            <input 
                                x-model="searchQuery"
                                @keydown.enter="performSearch()"
                                placeholder="Search your stored responses..."
                                class="flex-1 px-4 py-3 bg-gray-800 border border-gray-600 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                            >
                            <input 
                                type="number" 
                                x-model="numResults"
                                min="1" 
                                max="20" 
                                class="w-20 px-3 py-3 bg-gray-800 border border-gray-600 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent text-center"
                            >
                            <button 
                                @click="performSearch()"
                                :disabled="!searchQuery.trim() || searchLoading"
                                class="px-6 py-3 bg-gradient-to-r from-purple-500 to-pink-600 text-white rounded-xl hover:from-purple-600 hover:to-pink-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 font-medium"
                            >
                                <i class="fas fa-search" x-show="!searchLoading"></i>
                                <i class="fas fa-spinner fa-spin" x-show="searchLoading"></i>
                            </button>
                        </div>
                    </div>
                    
                    <!-- Search Results -->
                    <div x-show="searchResults.length > 0" class="space-y-4">
                        <h3 class="text-lg font-semibold">
                            <i class="fas fa-list mr-2 text-purple-400"></i>
                            Search Results (<span x-text="searchResults.length"></span>)
                        </h3>
                        
                        <template x-for="(result, index) in searchResults" :key="index">
                            <div class="dark-glass rounded-xl p-4">
                                <div class="flex items-center justify-between mb-3">
                                    <div class="flex items-center space-x-2">
                                        <span class="bg-purple-600 text-white text-xs px-2 py-1 rounded-full" x-text="'#' + result.rank"></span>
                                        <span 
                                            class="text-xs px-2 py-1 rounded-full"
                                            :class="result.metadata.type === 'vision_response' ? 'bg-green-600 text-white' : 'bg-blue-600 text-white'"
                                            x-text="result.metadata.type === 'vision_response' ? 'Vision' : 'Text'"
                                        ></span>
                                        <span class="text-xs text-gray-400" x-text="new Date(result.metadata.timestamp).toLocaleDateString()"></span>
                                    </div>
                                    <div class="flex items-center space-x-2">
                                        <span class="text-xs text-gray-400">Similarity:</span>
                                        <div class="w-16 bg-gray-700 rounded-full h-2">
                                            <div 
                                                class="bg-gradient-to-r from-purple-500 to-pink-600 h-2 rounded-full"
                                                :style="`width: ${result.similarity * 100}%`"
                                            ></div>
                                        </div>
                                        <span class="text-xs font-mono" x-text="(result.similarity * 100).toFixed(1) + '%'"></span>
                                    </div>
                                </div>
                                
                                <div class="text-sm text-gray-300 mb-2">
                                    <strong>Prompt:</strong> <span x-text="result.metadata.prompt"></span>
                                </div>
                                
                                <div class="text-sm text-gray-100">
                                    <strong>Response:</strong> 
                                    <div x-text="result.content.length > 300 ? result.content.substring(0, 300) + '...' : result.content" class="mt-1 whitespace-pre-wrap"></div>
                                </div>
                            </div>
                        </template>
                    </div>
                    
                    <div x-show="searchResults.length === 0 && searchPerformed" class="text-center py-12">
                        <i class="fas fa-search text-4xl text-gray-600 mb-4"></i>
                        <p class="text-gray-400">No results found for your search query.</p>
                    </div>
                </div>
            </div>
            
            <!-- History Tab -->
            <div x-show="activeTab === 'history'" class="flex-1 flex flex-col">
                <!-- Header -->
                <div class="border-b border-gray-700 p-4">
                    <h2 class="text-xl font-semibold">Conversation History</h2>
                    <p class="text-gray-400 text-sm">View and manage your conversation history</p>
                </div>
                
                <!-- Content -->
                <div class="flex-1 overflow-y-auto custom-scrollbar p-4">
                    <!-- Filter -->
                    <div class="mb-6">
                        <select 
                            x-model="historyFilter"
                            @change="filterHistory()"
                            class="px-4 py-2 bg-gray-800 border border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                        >
                            <option value="all">All Conversations</option>
                            <option value="text">Text Only</option>
                            <option value="vision">Vision Only</option>
                        </select>
                    </div>
                    
                    <!-- History Items -->
                    <div x-show="filteredHistory.length > 0" class="space-y-4">
                        <template x-for="(item, index) in filteredHistory" :key="index">
                            <div class="dark-glass rounded-xl p-4">
                                <div class="flex items-center justify-between mb-3">
                                    <div class="flex items-center space-x-2">
                                        <span 
                                            class="text-xs px-2 py-1 rounded-full"
                                            :class="item.type === 'vision' ? 'bg-green-600 text-white' : 'bg-blue-600 text-white'"
                                            x-text="item.type === 'vision' ? 'Vision' : 'Text'"
                                        ></span>
                                        <span class="text-xs text-gray-400" x-text="new Date(item.timestamp).toLocaleString()"></span>
                                        <span x-show="item.image_name" class="text-xs text-green-400" x-text="'📷 ' + item.image_name"></span>
                                    </div>
                                    <span class="text-xs text-gray-500" x-text="item.model"></span>
                                </div>
                                
                                <div class="text-sm text-gray-300 mb-2">
                                    <strong>Prompt:</strong> <span x-text="item.prompt"></span>
                                </div>
                                
                                <div class="text-sm text-gray-100">
                                    <strong>Response:</strong> 
                                    <div x-text="item.response.length > 400 ? item.response.substring(0, 400) + '...' : item.response" class="mt-1 whitespace-pre-wrap"></div>
                                </div>
                            </div>
                        </template>
                    </div>
                    
                    <div x-show="filteredHistory.length === 0" class="text-center py-12">
                        <i class="fas fa-history text-4xl text-gray-600 mb-4"></i>
                        <p class="text-gray-400">No conversation history yet. Start chatting or analyzing images!</p>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <!-- Loading Overlay -->
    <div x-show="status.status === 'loading'" class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <div class="dark-glass rounded-2xl p-8 text-center max-w-md mx-4">
            <div class="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-500 mx-auto mb-4"></div>
            <h3 class="text-lg font-semibold mb-2">Initializing Components</h3>
            <p class="text-gray-400 mb-4">Loading embedding models and vector store...</p>
            <div class="w-full bg-gray-700 rounded-full h-2">
                <div 
                    class="bg-gradient-to-r from-blue-500 to-purple-600 h-2 rounded-full transition-all duration-500"
                    :style="`width: ${status.progress}%`"
                ></div>
            </div>
            <div class="text-sm text-gray-400 mt-2" x-text="status.progress + '%'"></div>
        </div>
    </div>
    
    <!-- Notification Toast -->
    <div 
        x-show="notification.show" 
        x-transition:enter="transition ease-out duration-300"
        x-transition:enter-start="opacity-0 transform translate-y-2"
        x-transition:enter-end="opacity-100 transform translate-y-0"
        x-transition:leave="transition ease-in duration-200"
        x-transition:leave-start="opacity-100 transform translate-y-0"
        x-transition:leave-end="opacity-0 transform translate-y-2"
        class="fixed top-4 right-4 z-50"
    >
        <div 
            class="rounded-lg p-4 max-w-sm"
            :class="{
                'bg-green-600': notification.type === 'success',
                'bg-red-600': notification.type === 'error',
                'bg-blue-600': notification.type === 'info',
                'bg-yellow-600': notification.type === 'warning'
            }"
        >
            <div class="flex items-center">
                <i 
                    :class="{
                        'fas fa-check': notification.type === 'success',
                        'fas fa-exclamation-triangle': notification.type === 'error',
                        'fas fa-info': notification.type === 'info',
                        'fas fa-exclamation': notification.type === 'warning'
                    }"
                    class="mr-2"
                ></i>
                <span x-text="notification.message"></span>
            </div>
        </div>
    </div>

    <script>
        function appData() {
            return {
                // State
                apiKey: '',
                activeTab: 'chat',
                status: {
                    status: 'not_started',
                    progress: 0,
                    components_ready: false,
                    vector_count: 0
                },
                
                // Chat
                chatInput: '',
                chatMessages: [],
                chatLoading: false,
                structuredResponse: true,
                saveToVector: true,
                
                // Vision
                selectedImage: null,
                selectedImageFile: null,
                visionPrompt: '',
                visionResult: '',
                visionLoading: false,
                saveVisionToVector: true,
                
                // Search
                searchQuery: '',
                numResults: 5,
                searchResults: [],
                searchLoading: false,
                searchPerformed: false,
                
                // History
                history: [],
                filteredHistory: [],
                historyFilter: 'all',
                
                // Notification
                notification: {
                    show: false,
                    type: 'info',
                    message: ''
                },
                
                // Methods
                init() {
                    this.updateStatus();
                    this.loadHistory();
                    
                    // Poll status every 2 seconds
                    setInterval(() => {
                        this.updateStatus();
                    }, 2000);
                },
                
                async updateStatus() {
                    try {
                        const response = await fetch('/api/status');
                        this.status = await response.json();
                    } catch (error) {
                        console.error('Error updating status:', error);
                    }
                },
                
                async initComponents() {
                    if (!this.apiKey) {
                        this.showNotification('error', 'Please enter your API key first');
                        return;
                    }
                    
                    try {
                        await fetch('/api/init', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({})
                        });
                        this.showNotification('info', 'Initializing components...');
                    } catch (error) {
                        this.showNotification('error', 'Failed to initialize components');
                    }
                },
                
                async sendMessage() {
                    if (!this.chatInput.trim() || !this.apiKey || this.chatLoading) return;
                    
                    const userMessage = {
                        id: Date.now() + Math.random(),
                        type: 'user',
                        content: this.chatInput,
                        timestamp: new Date().toISOString()
                    };
                    
                    const aiMessage = {
                        id: Date.now() + Math.random() + 1,
                        type: 'ai',
                        content: '',
                        structured: this.structuredResponse,
                        structuredData: null,
                        loading: true,
                        timestamp: new Date().toISOString()
                    };
                    
                    this.chatMessages.push(userMessage, aiMessage);
                    const prompt = this.chatInput;
                    this.chatInput = '';
                    this.chatLoading = true;
                    
                    this.$nextTick(() => {
                        this.$refs.chatContainer.scrollTop = this.$refs.chatContainer.scrollHeight;
                    });
                    
                    try {
                        const response = await fetch('/api/chat', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                                api_key: this.apiKey,
                                prompt: prompt,
                                structured: this.structuredResponse,
                                save_to_vector: this.saveToVector
                            })
                        });
                        
                        const data = await response.json();
                        
                        if (response.ok) {
                            aiMessage.content = data.content;
                            aiMessage.loading = false;
                            aiMessage.saved_to_vector = data.saved_to_vector;
                            
                            if (this.structuredResponse && data.structured) {
                                aiMessage.structuredData = data.structured;
                            }
                            
                            if (data.saved_to_vector) {
                                this.showNotification('success', 'Response saved to vector store');
                            }
                        } else {
                            aiMessage.content = `Error: ${data.error}`;
                            aiMessage.loading = false;
                        }
                    } catch (error) {
                        aiMessage.content = `Error: ${error.message}`;
                        aiMessage.loading = false;
                    }
                    
                    this.chatLoading = false;
                    this.$nextTick(() => {
                        this.$refs.chatContainer.scrollTop = this.$refs.chatContainer.scrollHeight;
                    });
                },
                
                handleImageSelect(event) {
                    const file = event.target.files[0];
                    if (file) {
                        this.processImageFile(file);
                    }
                },
                
                handleImageDrop(event) {
                    const file = event.dataTransfer.files[0];
                    if (file && file.type.startsWith('image/')) {
                        this.processImageFile(file);
                    }
                },
                
                processImageFile(file) {
                    this.selectedImageFile = file;
                    const reader = new FileReader();
                    reader.onload = (e) => {
                        this.selectedImage = e.target.result;
                    };
                    reader.readAsDataURL(file);
                },
                
                clearImage() {
                    this.selectedImage = null;
                    this.selectedImageFile = null;
                    this.visionResult = '';
                },
                
                async analyzeImage() {
                    if (!this.selectedImageFile || !this.visionPrompt.trim() || !this.apiKey || this.visionLoading) return;
                    
                    this.visionLoading = true;
                    this.visionResult = '';
                    
                    const formData = new FormData();
                    formData.append('image', this.selectedImageFile);
                    formData.append('api_key', this.apiKey);
                    formData.append('prompt', this.visionPrompt);
                    formData.append('save_to_vector', this.saveVisionToVector);
                    
                    try {
                        const response = await fetch('/api/vision', {
                            method: 'POST',
                            body: formData
                        });
                        
                        const data = await response.json();
                        
                        if (response.ok) {
                            this.visionResult = data.content;
                            if (data.saved_to_vector) {
                                this.showNotification('success', 'Analysis saved to vector store');
                            }
                        } else {
                            this.visionResult = `Error: ${data.error}`;
                            this.showNotification('error', data.error);
                        }
                    } catch (error) {
                        this.visionResult = `Error: ${error.message}`;
                        this.showNotification('error', error.message);
                    }
                    
                    this.visionLoading = false;
                },
                
                async performSearch() {
                    if (!this.searchQuery.trim() || this.searchLoading) return;
                    
                    this.searchLoading = true;
                    this.searchResults = [];
                    
                    try {
                        const response = await fetch('/api/search', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                                query: this.searchQuery,
                                num_results: this.numResults
                            })
                        });
                        
                        const data = await response.json();
                        
                        if (response.ok) {
                            this.searchResults = data.results;
                            this.searchPerformed = true;
                        } else {
                            this.showNotification('error', data.error);
                        }
                    } catch (error) {
                        this.showNotification('error', error.message);
                    }
                    
                    this.searchLoading = false;
                },
                
                async loadHistory() {
                    try {
                        const response = await fetch('/api/history');
                        const data = await response.json();
                        this.history = data.history || [];
                        this.filterHistory();
                    } catch (error) {
                        console.error('Error loading history:', error);
                    }
                },
                
                filterHistory() {
                    if (this.historyFilter === 'all') {
                        this.filteredHistory = [...this.history];
                    } else {
                        this.filteredHistory = this.history.filter(item => item.type === this.historyFilter);
                    }
                },
                
                async clearHistory() {
                    try {
                        await fetch('/api/clear-history', { method: 'POST' });
                        this.history = [];
                        this.filteredHistory = [];
                        this.showNotification('success', 'History cleared');
                    } catch (error) {
                        this.showNotification('error', 'Failed to clear history');
                    }
                },
                
                async clearVector() {
                    try {
                        const response = await fetch('/api/clear-vector', { method: 'POST' });
                        const data = await response.json();
                        
                        if (data.status === 'cleared') {
                            this.showNotification('success', 'Vector store cleared');
                            this.searchResults = [];
                        } else {
                            this.showNotification('error', 'Failed to clear vector store');
                        }
                    } catch (error) {
                        this.showNotification('error', 'Failed to clear vector store');
                    }
                },
                
                showNotification(type, message) {
                    this.notification = { show: true, type, message };
                    setTimeout(() => {
                        this.notification.show = false;
                    }, 5000);
                }
            }
        }
    </script>
</body>
</html>
'''

if __name__ == '__main__':
    # Initialize models on startup
    init_models()
    app.run(debug=True, host='0.0.0.0', port=5000)