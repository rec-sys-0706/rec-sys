import unittest
from flask_jwt_extended import create_access_token
from app import create_app
from server.models.user import User

class UserApiTest(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.client = self.app.test_client()
        self.app_context = self.app.app_context()
        self.app_context.push()

        # 假設已經創建了數據庫並有用戶數據
        self.user = User.query.first()  # 獲取測試用戶
        self.access_token = create_access_token(identity=self.user.uuid)  # 為用戶生成 JWT token

    def tearDown(self):
        self.app_context.pop()

    # 測試 GET 請求，帶上 JWT token
    def test_get_user(self):
        headers = {
            'Authorization': f'Bearer {self.access_token}'
        }
        response = self.client.get(f'/api/user/{self.user.uuid}', headers=headers)
        self.assertEqual(response.status_code, 200)
        self.assertIn('data', response.json)

    # 測試 PUT 請求，更新用戶數據
    def test_update_user(self):
        headers = {
            'Authorization': f'Bearer {self.access_token}'
        }
        payload = {
            'email': 'new_email@example.com'
        }
        response = self.client.put