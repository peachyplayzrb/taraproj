from elasticsearch import Elasticsearch
import csv

indexname = "cord19_2020-03-20"

# Connect to Elasticsearch
es = Elasticsearch(["http://localhost:9200"], request_timeout=30)

# Ensure the index exists
if not es.indices.exists(index=indexname):
    es.indices.create(index=indexname)

count = 0
with open('/Users/anirban/IR/ElasticSearch/2020-03-20/metadata.csv', encoding='utf-8-sig') as f_in:
    reader = csv.DictReader(f_in)

    # Normalize column names
    reader.fieldnames = [col.strip().lower() for col in reader.fieldnames]

    for row in reader:
        cord_uid = row.get('doi', 'unknown')  # Using 'doi' instead of 'cord_uid'
        title = row.get('title', 'No Title Available')
        abstract = row.get('abstract', 'No Abstract Available')
        authors = row.get('authors', '').split('; ') if row.get('authors') else []

        print(f"Indexing document {count}: {cord_uid}...")
        count += 1
        indexDoc = {
            'authors': authors,
            'title': title,
            'abstract': abstract
        }

        # Use 'body=' for Elasticsearch 7.x
        res = es.index(index=indexname, id=cord_uid, body=indexDoc)

print(f"Indexed {count} documents successfully.")

