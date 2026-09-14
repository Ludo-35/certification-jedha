import boto3
from dotenv import load_dotenv
import os

load_dotenv()
s3 = boto3.client('s3')
bucket_name = "testludo35"

print(f"--- Contenu du bucket {bucket_name} ---")
response = s3.list_objects_v2(Bucket=bucket_name)

if 'Contents' in response:
    for obj in response['Contents']:
        print(f"Fichier trouvé : {obj['Key']}")
else:
    print("Le bucket est vide !")