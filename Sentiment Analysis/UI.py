from turtle import title
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt 
from wordcloud import WordCloud,STOPWORDS

import re
import nltk
import matplotlib.pyplot as plt 
import seaborn as sns

st.set_page_config(layout="wide")
# 压制SettingWithCopyWarning的warning
pd.set_option('mode.chained_assignment', None)
# pd.set_option('display.max_colwidth', None)

import numpy as np
from PIL import Image

import warnings
warnings.filterwarnings("ignore")

# 标题
st.title('Tweeter analytics 💨')
st.subheader("                 - Find insights of tweets about Russia-Ukraine war.")


from os import listdir
def find_csv_filenames(path_to_dir, suffix=".csv" ):
    filenames = listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix ) ]

st.write("### Select a date or all dates(02.24 to 04-19)")
keyword1 = "Russia"
path1 = f"all_label_data_{keyword1}"
russian_csv_filesnames = find_csv_filenames(f"./{path1}")
dates1 = sorted([filename.split("_")[2] for filename in russian_csv_filesnames])
# print(dates1)

keyword2 = "Ukraine"
path2 = f"all_label_data_{keyword2}"
ukr_csv_filesnames = find_csv_filenames(f"./{path2}")
dates2 = sorted([filename.split("_")[1] for filename in ukr_csv_filesnames])
# print(dates2)

dates = sorted(list(set(dates1).intersection(set(dates2))))
# print("文件共有日期", dates)


date_selected = st.selectbox("Date", dates)
st.write(f"Show analytics of {date_selected} tweets:")

# 0. 读取csv data
data1 = pd.read_csv(f"./{path1}/label_en_{date_selected}_{keyword1}.csv")
data1 = data1.dropna(subset=['content', 'label', 'id', 'username'])
data1.drop_duplicates(inplace=True)
# st.dataframe(data1[['content']][:5], width=50000)

data2 = pd.read_csv(f"./{path2}/label_{date_selected}_{keyword2}.csv")
data2 = data2.dropna(subset=['content', 'label', 'id', 'username'])
data2.drop_duplicates(inplace=True)
# st.write(data2[:5])

# 1.读取wordcloud
st.write(f"### Wordcloud: 2 keywaord searches(filter out topic and non-meaningful words)")

wordcloud_path = "all_wordclouds/"
filename1 = f"wordcloud_{keyword1}_{date_selected}"
image1 = Image.open(f'{wordcloud_path}{filename1}.jpg')
# st.image(image, caption=filename, width=500)

filename2 = f"wordcloud_{keyword2}_{date_selected}"
image2 = Image.open(f'{wordcloud_path}{filename2}.jpg')

st.write(f"Wordclouds for {keyword1} VS {keyword2} on {date_selected}")
st.image([image1, image2], width=700)
# st.image(image2, caption=filename, width=500)

# 2. 获取预测stances分布
st.write(f"### Predicted Stance Distributuons on 2 sides")
def mapping(x):
    if x == 1.0:
        return "Supportive"
    elif x == 0.0:
        return "Neutral"
    else:
        return "Against"
dict1 = data1['label'].value_counts(dropna=False, normalize=True)
labels1 = list(map(mapping, dict1.keys()))
counts1 = dict1.iloc[:]

dict2 = data2['label'].value_counts(dropna=False, normalize=True)
labels2 = list(map(mapping, dict2.keys()))
counts2 = dict2.iloc[:]
 
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5))
# plt.suptitle(f'Stance on {date_selected}, {keyword1} VS {keyword2}', fontsize=10)
ax1.pie(counts1,labels = labels1,autopct = '%1.2f%%') 
ax1.set_title(f"Stance on {date_selected}, {keyword1}")
ax2.pie(counts2,labels = labels2,autopct = '%1.2f%%') 
ax2.set_title(f"Stance on {date_selected}, {keyword2}")
st.pyplot(fig)


# 3. 获取top tags, 除去搜索关键词
main_keywords = ["Russian", "Russia","Ukraine", "Ukrainian", "Ukrainians", "Putin", "Putins", 
                 "Russians", "Russias",
                 "Ukraina", "RussiaUkraineCrisis", "RussiaUkraine", "UkraineRussiaCrisis", "UkraineRussiaConflict",
                 "RussiaUkraineConflict", "UkraineRussia", "UkraineRussie", "BREAKING", 'UkraineRussiaWar',
                 "UkraineRussianWar", "RussiaUkraineWar", "War", "UkraineWar"]
main_keywords += [s.lower() for s in main_keywords] # 加上小写

tag1 = []
for index,row in data1.iterrows():
    if type(row["tags"]).__name__ == "str":
        tag1.extend(list(filter(None, re.sub(r"'|\[|\]|#", "", row["tags"]).split(", "))))
tag_filtered1 = [tag for tag in tag1 if tag not in main_keywords]

tag2 = []
for index,row in data2.iterrows():
    if type(row["tags"]).__name__ == "str":
        tag2.extend(list(filter(None, re.sub(r"'|\[|\]|#", "", row["tags"]).split(", "))))
tag_filtered2 = [tag for tag in tag2 if tag not in main_keywords]

freq1, freq2 = nltk.FreqDist(tag_filtered1), nltk.FreqDist(tag_filtered2)
K_tags = 7
dist1 = pd.DataFrame({'Tags': list(freq1.keys()), 'Count': list(freq1.values())}) \
        .nlargest(columns = "Count", n = K_tags)
dist2 = pd.DataFrame({'Tags': list(freq2.keys()), 'Count': list(freq2.values())}) \
        .nlargest(columns = "Count", n = K_tags) 

fig, axs = plt.subplots(1,2, figsize=(20,10), squeeze=False,)
fig.suptitle(f'Top {K_tags} tags in {keyword1} VS {keyword2}', fontsize=30)
sns.barplot(data = dist1, x = "Tags", y = "Count", ax = axs[0][0])
sns.barplot(data = dist2, x = "Tags", y = "Count", ax = axs[0][1])
st.pyplot(fig)


# 4. 获取点赞多的tweets
topK = 6
st.write(f"### Show top-likes tweets")

tweet_Russia = data1.nlargest(columns="like_count", n = 20)[["content", "like_count", "label"]]
tweet_Russia["like_count"] = tweet_Russia["like_count"].astype(int)
tweet_Russia["label"] = tweet_Russia["label"].map(mapping)
st.write(f"Top-{topK}-likes tweets on searching {keyword1}")
st.table(tweet_Russia[:topK])


tweet_Ukraine = data2.nlargest(columns="like_count", n = 20)[["content", "like_count", "label"]]
tweet_Ukraine["like_count"] = tweet_Ukraine["like_count"].astype(int)
tweet_Ukraine["label"] = tweet_Ukraine["label"].map(mapping)
st.write(f"Top-{topK}-likes tweets on searching {keyword2}")
st.table(tweet_Ukraine[:topK])




# st.pyplot(fig)