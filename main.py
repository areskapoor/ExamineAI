from fastapi import FastAPI
from pydantic import BaseModel
from upsertData import initPinecone
import pinecone
import makeQuery
from config import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME, limit


app = FastAPI()

class Query(BaseModel):
    query: str

#Initialize DB
initPinecone()
index = pinecone.Index(PINECONE_INDEX_NAME)

@app.post('/query/')
async def getResponse(query: Query):
    prompt, context = makeQuery.retrieve(query.text, index)
    response = makeQuery.complete(prompt)
    
    #all urls used for response (no duplicates)
    urls = list(set([x["metadata"]['url'] for x in context]))
    headers = list(set([x["metadata"]['header'] for x in context]))
    
    makeQuery.writePromptResponse(query.text, response)
    
    return {
        "query": query.text,
        "response": response,
        "headers": headers,
        "URLs": urls
    }