from django.db import models
 
 
class Movie(models.Model):
    user_id = models.CharField(max_length=100000)  # 用戶id
    item_id = models.CharField(max_length=100000)  # 電影id
    rating = models.CharField(max_length=100000)  # 評分
    timestamp = models.CharField(max_length=100000) # 時間戳