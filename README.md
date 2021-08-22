# Backend API for Text Summarization
## Docker Image
Available on [Docker Hub](https://hub.docker.com/repository/docker/jdehorty/summarizer)


## Swagger

![img.png](https://i.gyazo.com/61f0fcf02ad7f6c2c0297cb135c92892.png)

- `GET` can be used from Swagger
- `POST` does NOT work from Swagger but can be tested as shown below.

### Sample POST (Python)
```python
import requests
import json

url = "http://192.168.1.167:8000/api/summarize"

payload = json.dumps({
  "text": "In order to find the most relevant sentences in text, a graph is constructed where the vertices of the graph represent each sentence in a document and the edges between sentences are based on content overlap, namely by calculating the number of words that 2 sentences have in common. Based on this network of sentences, the sentences are fed into the Pagerank algorithm which identifies the most important sentences. When we want to extract a summary of the text, we can now take only the most important sentences. In order to find relevant keywords, the textrank algorithm constructs a word network. This network is constructed by looking which words follow one another. A link is set up between two words if they follow one another, the link gets a higher weight if these 2 words occur more frequenctly next to each other in the text. On top of the resulting network the Pagerank algorithm is applied to get the importance of each word. The top 1/3 of all these words are kept and are considered relevant. After this, a keywords table is constructed by combining the relevant words together if they appear following one another in the text.",
  "ratio": 0.2
})
headers = {
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)
```
### Sample POST (CURL)
```bash
curl --location --request POST 'http://192.168.1.167:8000/api/summarize' \
--header 'Content-Type: application/json' \
--data-raw '{
    "text": "In order to find the most relevant sentences in text, a graph is constructed where the vertices of the graph represent each sentence in a document and the edges between sentences are based on content overlap, namely by calculating the number of words that 2 sentences have in common. Based on this network of sentences, the sentences are fed into the Pagerank algorithm which identifies the most important sentences. When we want to extract a summary of the text, we can now take only the most important sentences. In order to find relevant keywords, the textrank algorithm constructs a word network. This network is constructed by looking which words follow one another.",
    "ratio": 0.10
}'
```

### Sample Response
```json
{
    "ratio": 0.20,
    "summary": "In order to find relevant keywords, the textrank algorithm constructs a word network."
}
```

## Unit Tests
- Found in `test_summarizer.py`
- Includes basic tests for text pre-processing and error handling

## Other
- `exceptions.py` - Includes the base clase for summarizer related errors and custom error for input text that is too short.
- `Dockerrun.aws.json` - Used for deployment to AWS (in case we want to delploy there one day)