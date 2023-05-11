import openai
import pinecone
import upsertData
from config import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME, limit


openai.api_key = OPENAI_API_KEY
embed_model = "text-embedding-ada-002"

# #Use when running this file directly
# upsertData.initPinecone()
# index = pinecone.Index(PINECONE_INDEX_NAME)

def complete(prompt):
    # query text-davinci-003
    res = openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt,
        temperature=0,
        max_tokens=400,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    return res['choices'][0]['text'].strip()


def retrieve(query, index):
    res = openai.Embedding.create(
        input=[query],
        engine=embed_model
    )

    # retrieve from Pinecone
    xq = res['data'][0]['embedding']

    # get relevant contexts
    res = index.query(xq, top_k=5, include_metadata=True)
    contexts = [
       x['metadata']['title'] + ": " + x['metadata']['header'] + ": " + x['metadata']['text'] + "    score:" + str(x['score']) for x in res['matches']
    ]

    # build our prompt with the retrieved contexts included
    prompt_start = (
        """You are a doctor who is answering a patient's health question. 
            You are helpful, articulate and concise, and make sure to only speak the truth. 
            You are able to answer any health-related questions, but mainly refer to the
            information given to you in the context below. You take more information from 
            the context paragraphs with a higher score number.""" + 
        "Answer the question based on the context below.\n\n"+
        "Context:\n"
    )
    prompt_end = (
        f"\n\nQuestion: {query}\nAnswer (Do not include number references. These should be in this form: [4] with any number in between the brackets.):"
    )
    # append contexts until hitting limit
    for i in range(1, len(contexts)):
        if len("\n\n---\n\n".join(contexts[:i])) >= limit:
            prompt = (
                prompt_start +
                "\n\n---\n\n".join(contexts[:i-1]) +
                prompt_end
            )
            break
        elif i == len(contexts)-1:
            prompt = (
                prompt_start +
                "\n\n---\n\n".join(contexts) +
                prompt_end
            )
    return prompt, res['matches']


def writePromptResponse(query, response):
    with open('outputs.txt', 'a') as f:
        queryMessage = 'query: ' + query + '\n'
        responseMessage = 'response: ' + response + '\n\n'
        f.write(queryMessage)
        f.write(responseMessage)