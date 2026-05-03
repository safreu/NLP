class CuadDTO:
    def __init__(self, json_data):
        version: str = json_data["version"]
        data = json_data["data"]
        print(version)
        #print(data)


