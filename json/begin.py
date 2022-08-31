import json 
people_string = '''
{

"people":[
    {
        "name":"john",
        "phone":"615-555-7164",
        "emails":["123@mail.com,456@mail.com"],
        "license":false
    },
    {
        "name":"jane",
        "phone":"560-555-5153",
        "emails":null,
        "license":true
    }
]
}
'''


data = json.loads(people_string)
for person in data["people"]:
    # print(person['name'])
    del person['phone']

new_string = json.dumps(data,indent=2,sort_keys=True)
print(new_string)