import os
from dotenv import load_dotenv

load_dotenv()
LINE_CHANNEL_ID = os.getenv("CHANNEL_ID")
REDIRECT_URI = "https://recsys.csie.fju.edu.tw/api/callback/login"
state = os.urandom(16).hex()
# https://access.line.me/oauth2/v2.1/login?returnUri=%2Foauth2%2Fv2.1%2Fauthorize%2Fconsent%3Fresponse_type%3Dcode%26client_id%3D2006450824%26redirect_uri%3Dhttps%253A%252F%252Frecsys.csie.fju.edu.tw%252Fapi%252Fcallback%252Flogin%26state%3D174d9dce34afbeea4ddaaf97d5529688%26scope%3Dprofile%2520openid%2520email&loginChannelId=2006450824&loginState=idRQvl8AfAwLs7pU2BFNBa
login_url = (
    f"https://access.line.me/oauth2/v2.1/authorize?response_type=code"
    f"&client_id={LINE_CHANNEL_ID}&redirect_uri={REDIRECT_URI}"
    f"&state={state}&scope=profile%20openid%20email"
)
print(login_url)


