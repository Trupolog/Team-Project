import requests

url = 'http://localhost:8000/predict'
image_path = 'tmp.jpg'

try:
    with open(image_path, 'rb') as f:
        files = {'image': f}
        print(files)
        print("Отправка запроса...")
        response = requests.post(url, files=files)
        print("Ответ получен.")

        try:
            response.raise_for_status()  # Вызывает исключение при ошибке HTTP
            print("Код состояния:", response.status_code)
            print("Заголовки:", response.headers)
            print("Текст ответа:", response.text)

            json_response = response.json()
            print("JSON-ответ:", json_response)

        except (requests.exceptions.JSONDecodeError, KeyError) as e:
            print("Ошибка при обработке ответа сервера:", e)
            print("Текст ответа:", response.text)

except FileNotFoundError as e:
    print("Файл изображения не найден:", e)
except requests.exceptions.RequestException as e:
    print("Ошибка при отправке запроса:", e)