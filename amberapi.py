import time
import urllib.request
import json
import socket


def internet(host="8.8.8.8", port=53, timeout=3):
  try:
    socket.setdefaulttimeout(timeout)
    socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
    return True
  except socket.error as ex:
    print(ex)
    return False


while(True):
    if(internet()):
        plates = []
        page = urllib.request.urlopen('http://api.missingkids.org/missingkids/servlet/JSONDataServlet?action=amberAlert')
        result = page.read()

        dict = json.loads(result)

        # print(dict)

        people = dict['persons']

        for person in people:
            id = person['amberId']
            # print(id)
            resultPage = urllib.request.urlopen('http://api.missingkids.org/missingkids/servlet/JSONDataServlet?action=amberDetail&amberId=' + str(id))
            info = json.loads(resultPage.read())
            peopleList = info['childBean']['personList']
            for personList in peopleList:
                if('vehicleList' in personList.keys()):
                    vehicles = personList['vehicleList']
                    for vehicle in vehicles:
                        # print(vehicle)
                        vehicle = {"licensePlateText": "abc123"}
                        if('licensePlateText' in vehicle.keys()):
                            licensePlate = vehicle['licensePlateText']
                            if(licensePlate and (not licensePlate in plates)):
                                # print("Plate: " + licensePlate)
                                plates.append(licensePlate)

    p = open("plates.txt", "r")
    contents = p.read().split("\n")
    p.close()
    write = ""
    for plate in plates:
        write += "," + plate
    if(len(contents) > 1):
        write += "\n" + contents[1]
    p = open("plates.txt", "w")
    p.write(write.lstrip(","))
    p.close()
    print("written")
    time.sleep(1800)