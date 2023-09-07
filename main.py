# import pymongo

# # Provide the mongodb localhost url to connect python to mongodb.
# client = pymongo.MongoClient("mongodb://localhost:27017/neurolabDB")

# # Database Name
# dataBase = client["neurolabDB"]

# # Collection  Name
# collection = dataBase['Products']

# # Sample data
# d = {'companyName': 'iNeuron',
#      'product': 'Affordable AI',
#      'courseOffered': 'Machine Learning with Deployment'}

# # Insert above records in the collection
# rec = collection.insert_one(d)

# # Lets Verify all the record at once present in the record with all the fields
# all_record = collection.find()

# # Printing all records present in the collection
# for idx, record in enumerate(all_record):
#      print(f"{idx}: {record}")

from sensor.exception import SensorException
from sensor.logger import logging
import sys,os
from sensor.utils import get_collection_as_dataframe
try:
     # logging.info("addition started")
     # print(1/0)
     # logging.info("addition completed")
     get_collection_as_dataframe(database_name="aps", collection_name="sensor")

except Exception as e:
     logging.info(e)
     raise SensorException(e, sys)