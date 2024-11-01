import google.generativeai as genai
#import os

genai.configure(api_key="AIzaSyBG-9zVx3nflBoemGyin-u_Fq3zi22M2mw")
#genai.configure(api_key="AIzaSyBG-9zVx3nflBoemGyin-u_Fq3zi22M2mw", transport="rest")
#genai.configure(api_key=os.environ["API_KEY"])

model = genai.GenerativeModel("gemini-1.5-flash")
response = model.generate_content("Write a story about a magic backpack.")
print(response.text)
