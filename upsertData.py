import openai
import pinecone
import csv
from time import sleep
from tqdm.auto import tqdm
from config import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME

#Read Data
def readData():
    data = []
    with open('data.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    
    return data

openai.api_key = OPENAI_API_KEY
index_name = PINECONE_INDEX_NAME
embed_model = "text-embedding-ada-002"

def initPinecone():
    #Initialization to get data for Pinecone Initialization
    res = openai.Embedding.create(
        input=[
            "Sample document text goes here",
            "there will be several phrases in each batch"
        ], engine=embed_model
    )

    # initialize connection to pinecone (get API key at app.pinecone.io)
    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment="us-east-1-aws"
    )

    # check if index already exists (it shouldn't if this is first time)
    if index_name not in pinecone.list_indexes():
        # if does not exist, create index
        pinecone.create_index(
            index_name,
            dimension=len(res['data'][0]['embedding']),
            metric='cosine',
            metadata_config={'indexed': ['channel_id', 'published']}
        )

def upsertData(index, data):
    batch_size = 100  # how many embeddings we create and insert at once

    for i in tqdm(range(0, len(data), batch_size)):
        # find end of batch
        i_end = min(len(data), i+batch_size)
        meta_batch = data[i:i_end]
        # get ids
        ids_batch = [x['id'] for x in meta_batch]
        # get texts to encode
        texts = [x['header'] + ":" + x['text'] for x in meta_batch]
        # create embeddings (try-except added to avoid RateLimitError)
        try:
            res = openai.Embedding.create(input=texts, engine=embed_model)
        except:
            done = False
            while not done:
                sleep(5)
                try:
                    res = openai.Embedding.create(input=texts, engine=embed_model)
                    done = True
                except:
                    pass
        embeds = [record['embedding'] for record in res['data']]
        # cleanup metadata
        meta_batch = [{
            'title': x['title'],
            'header': x['header'],
            'text': x['text'],
            'url': x['url'],
        } for x in meta_batch]
        to_upsert = list(zip(ids_batch, embeds, meta_batch))
        # upsert to Pinecone
        index.upsert(vectors=to_upsert)

# initPinecone()
# # connect to index
# index = pinecone.Index(index_name)

# data = readData()
# upsertData(index,data)