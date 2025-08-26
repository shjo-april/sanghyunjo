import sanghyunjo as shjo

data = [{'image_id': '00001', 'tags': ['한글', 'english', '123']}]

shjo.jswrite('./examples/example_orjson.json', data, backend='orjson')
shjo.jswrite('./examples/example_json.json', data, backend='json')

orjson_data = shjo.jsread('./examples/example_orjson.json', backend='orjson')
json_data = shjo.jsread('./examples/example_json.json', backend='json')

print(orjson_data)
print(json_data)

shjo.jslwrite('./examples/example_orjson.jsonl', data, backend='orjson')
data = shjo.jslread('./examples/example_orjson.jsonl', backend='orjson')
print(data)