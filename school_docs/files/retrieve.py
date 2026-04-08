from elasticsearch import Elasticsearch

# Create an Elasticsearch client instance
es = Elasticsearch("http://localhost:9200/")

# Define the index to query
index_name = "cord19_2020-03-20"

# Define the search query
search_query = {
    "multi_match": {
        "query": "covid vaccine",
        "fields": ["title^3", "abstract"]
    }
}

# Use 'body=' instead of 'query=' for Elasticsearch 7.x
result = es.search(index=index_name, body={"query": search_query})

# Print the results
maxscore = result['hits']['max_score']
print("Max score:", maxscore)  # Used to normalize scores

print("Received %d documents." % result['hits']['total']['value'])

for hit in result['hits']['hits']:
    print(hit['_score']/maxscore, hit['_source'].get('authors', 'Unknown'), hit['_source'].get('title', 'No Title'))
