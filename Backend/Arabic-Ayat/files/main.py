from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
import zipfile
import os

app = FastAPI()

# === Startup config ===
class QueryRequest(BaseModel):
    question: str

llm = None
retriever = None
chain = None

@app.on_event("startup")
def load_components():
    global llm, retriever, chain

    api_key = os.getenv('api_key')

    # --- Load LLM ---
    llm = ChatGroq(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0,
        max_tokens=1024,
        api_key=api_key
    )

    # --- Load Embeddings ---
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # --- Unzip Vectorstore if needed ---
    zip_path = "faiss_index.zip"
    extract_path = "faiss_index"
    if not os.path.exists(extract_path):
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(extract_path)
        print("✅ Unzipped FAISS index.")

    # --- Load FAISS Vectorstore & create retriever ---
    vectorstore = FAISS.load_local(
        extract_path,
        embeddings,
        allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    print("✅ FAISS index loaded.")

    # --- Prepare prompt template ---
    ayat_finder_prompt = '''
    You are an Arabic Ayat Finder assistant.
    Your goal is to provide an accurate and concise answer by extracting the exact Arabic verse (Ayah) from the provided retrieved context.
    Your task is to output only the exact Ayah in Arabic as it appears in the context, along with its full reference details.
    
    Instructions:
    1. Locate the segment in the retrieved context that directly contains the requested Ayah.
    2. Output only the Arabic text of the Ayah, without any additional commentary or translation.
    3. Provide the complete reference alongside the Ayah, including:
       - Surah number and name (Arabic and English)
       - Ayah (verse) number
       - Any other available metadata from the context (e.g., Juz number, Hizb, revelation order) if present.
    4. If the exact Ayah cannot be found in the context, respond with "لا أعلم".
    5. Do not add, infer, or modify any content; strictly reproduce what exists in the context.
    
    Retrieved context:
    {context}
    
    User's question:
    {question}
    
    Your response:
    '''  

    prompt = PromptTemplate(
        template=ayat_finder_prompt,
        input_variables=["context", "question"]
    )

    # --- Assemble a stateless RetrievalQA chain (no memory) ---
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt},
        verbose=False,
    )

@app.get("/")
def root():
    return {"message": "Quran Ayat Finder API is up."}

@app.post("/query")
def query(request: QueryRequest):
    try:
        result = chain.invoke({"query": request.question})
        return {"answer": result["result"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
