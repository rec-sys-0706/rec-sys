from .user import User, UserSchema
from .item import Item, ItemSchema
from .behavior import Behavior, BehaviorSchema
# from .stop_view import Stopview, StopviewSchema
# from .unclicked_picks import Unclickedpicks, UnclickedpicksSchema
# from .recommendation_score import Recommendationscore, RecommendationscoreSchema
from .recommendation_log import Recommendationlog, RecommendationlogSchema

def init_db_models():
    # 這裡的 init 可以確保所有模型都被正確地初始化
    pass