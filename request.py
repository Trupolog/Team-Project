import requests
from typing import Dict, Union

# Define the URL of the prediction endpoint
url: str = 'http://localhost:8000/predict'

# Specify the path to the image file
image_path: str = 'tmp.jpg'

try:
    # Open the image file in binary mode
    with open(image_path, 'rb') as f:
        # Create a dictionary containing the image file
        files: Dict[str, Union[str, bytes]] = {'image': f}
        print(files)
        print("Sending request...")

        # Send a POST request to the prediction endpoint with the image file
        response = requests.post(url, files=files)
        print("Response received.")

        try:
            # Raise an exception if the response status code indicates an HTTP error
            response.raise_for_status()

            print("Status code:", response.status_code)
            print("Headers:", response.headers)
            print("Response text:", response.text)

            # Parse the response JSON
            json_response: Dict[str, Union[int, str]] = response.json()
            print("JSON response:", json_response)

        except (requests.exceptions.JSONDecodeError, KeyError) as e:
            # Handle errors related to JSON decoding or missing keys
            print("Error processing server response:", e)
            print("Response text:", response.text)

except FileNotFoundError as e:
    # Handle the case when the image file is not found
    print("Image file not found:", e)
except requests.exceptions.RequestException as e:
    # Handle errors related to sending the request
    print("Error sending request:", e)