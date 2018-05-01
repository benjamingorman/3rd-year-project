import json

for s in ["baum_encoding", "som_encoding"]:
    print(s)
    for x in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
        with open("{}/work{}/cmm_stats.json".format(s, x), 'r') as f:
            data = json.loads(f.read())
            acc = data["correct"]/float(data["wrong"]+data["correct"])
            print(acc)
