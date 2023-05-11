# ExamineAI
AI Chatbot that gives evidence-based responses from Examine.com information

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository
2. Install the Dependencies (Note: project was written and tested in Python 3.9.10):
```
pip install -r requirements.txt
```
3. Store Pinecone, OpenAI API keys and Pinecone index name in config.py

## Usage

1. Run the functions specified in upsertData in order to upsert data.csv to your Pinecone database
2. Run the following command to start the app
```
uvicorn main:app --reload
```
It should give you the information in the form: `Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)`
3. Then go to `http://127.0.0.1:8000/docs` (replace URL with the URL given to you by uvicorn) to see how to interact with the API and make requests.
Your requests inputs and outputs will be written to output.txt locally.

**Note:** Do not run any of the other files directly. For context:

* *webScraper.py* is run to scrape Examine.com and store information needed in data.csv
* *upsertData.py* embeds the data from the csv file using openai's text embeddings and then upserts them to the given Pinecone database
* *makeQuery.py* takes in a query, finds the k most similar embeddings to the query from the database and creates a prompt that gets sent to openai's text completion API which returns the response that is returned to the user
* *main.py* contains the endpoints and makes calls to makeQuery.py to get a response based on a user's query.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.


## License

This project is licensed under the MIT License - see the LICENSE.md file for details.
