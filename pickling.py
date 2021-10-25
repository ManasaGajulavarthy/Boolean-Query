import pickle

payload = {"ip": "3.144.180.153", "port": "9999", "name": "execute_query"}


filename = 'project2_index_details.pickle'
outfile = open(filename, 'wb')

pickle.dump(payload, outfile)
outfile.close()

with open('project2_index_details.pickle', 'rb') as f:
 b = pickle.load(f)

print(payload == b)

