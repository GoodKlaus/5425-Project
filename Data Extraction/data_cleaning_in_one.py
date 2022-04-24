import pandas as pd
from cleantext import clean
import re
import datetime

# 我的cleantext包没有no_emoji的arg, 换了个方式去除emoji
def deEmojify(text):
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',text)

def filter_punc(text):
    punc = '"$%&\'()*+-/:<=>?@[\\]^_`{|}~'
    temp = ''.join([c for c in text if ord(c)<128])
    return temp.translate(str.maketrans('', '', punc))

def run_processing_steps(filename):
    tweets_df = pd.read_csv(filename, lineterminator='\n')

    tmp = tweets_df.content.str.findall("#\w+")
    # print(tmp[tmp.apply(lambda x: x!=[])]) # 查看存在#的行

    tmp = tweets_df.content.str.findall("@\w+")
    # print(tmp[tmp.apply(lambda x: x!=[])]) # 查看存在@的行
    
    # 移除content的@ 和 \n
    tweets_df.content = tweets_df.content.str.replace("\n", "")
    # 去除https
    tweets_df.content = tweets_df.content.str.replace(r"https*\S+", "", regex=True)
    # &符号 -> and
    tweets_df.content = tweets_df.content.str.replace("&amp;", "and")
    tweets_df.content = tweets_df.content.str.replace(r"@\w+", "", regex=True)
    # 1提取tag
    tmp = tweets_df.content.str.findall("#\w+")
    # print(tmp[tmp.apply(lambda x: x!=[])]) # 查看存在#的行

    # 新建列接受#的内容，列`tags`
    tweets_df['tags'] = tmp # 存为list，新的一列
    # print(tmp)

    # 2 并移除# 【但保留word】 因为【很多#，是内容的一部分】
    # tweets_df.content = tweets_df.content.str.replace(r"#\w+", "", regex=True)
    tweets_df.content = tweets_df.content.str.replace(r"#", " ", regex=True)
    
    tweets_df.content = tweets_df.content.apply(lambda x: deEmojify(x))
    tweets_df.content = tweets_df.content.str.replace(r'\s{2,}', " ", regex=True)
    tweets_df.content = tweets_df.content.apply(filter_punc)
    
    # 处理完所有的 再drop
    tweets_df.drop_duplicates(inplace=True)
    tweets_df.dropna(subset=['content'], inplace=True)
    return tweets_df

def main(start, end, path):
    dates = list(pd.date_range(start, end, freq='1D').strftime("%m.%d"))
    print(dates)
    Country = ['Russia', 'Ukraine']
    #path = '../../Project/'
    for c in Country:
        for i in dates:
            file = f"en_{i}_{c}.csv"
            df = run_processing_steps(path+file)
            df.to_csv(f"./process/processed_en_{i}_{c}.csv",index=False)