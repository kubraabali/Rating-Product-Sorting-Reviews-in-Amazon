import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler


pd.set_option("display.max_columns", None)
pd.set_option('display.max_rows', None)
pd.set_option("display.width", 200)

df= pd.read_csv("datasets/amazon_review.csv")
df.head()

# Average Rating'i Güncel Yorumlara Göre Hesaplayınız ve Var Olan Average Rating ile Kıyaslayınız.

# Average Rate
df["overall"].mean() #4.587589013224822

df.describe().T


#Zaman bazlı ağırlıklandırma
def time_based_weighted_average(dataframe, w1=40, w2=30, w3=20, w4=10):
    return dataframe.loc[dataframe["day_diff"] <= dataframe["day_diff"].quantile(0.25), "overall"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > dataframe["day_diff"].quantile(0.25)) & (dataframe["day_diff"] <= dataframe["day_diff"].quantile(0.50)), "overall"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > dataframe["day_diff"].quantile(0.50)) & (dataframe["day_diff"] <= dataframe["day_diff"].quantile(0.75)), "overall"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > dataframe["day_diff"].quantile(0.75)), "overall"].mean() * w4 / 100

time_based_weighted_average((df)) #4.628116998159475


###################################################
# Ürün için Ürün Detay Sayfasında Görüntülenecek 20 Review'i Belirlemek
###################################################

df["helpful_no"] = df["total_vote"]- df["helpful_yes"]
df.sort_values("total_vote", ascending=False).head(12)

def wilson_lower_bound(up, down, confidence=0.95):
    """
    Wilson Lower Bound Score hesapla

    - Bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
    - Hesaplanacak skor ürün sıralaması için kullanılır.
    - Not:
    Eğer skorlar 1-5 arasıdaysa 1-3 negatif, 4-5 pozitif olarak işaretlenir ve bernoulli'ye uygun hale getirilebilir.
    Bu beraberinde bazı problemleri de getirir. Bu sebeple bayesian average rating yapmak gerekir.

    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)


df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)
df.head()


def score_pos_neg_diff(df, up, down):
    df["score_pos_neg_diff"] = df[up] - df[down]

score_pos_neg_diff(df,"helpful_yes","helpful_no")

def score_average_rating (df,up,all):
    df["score_average_rating"] = df[up]/ df[all]

score_average_rating(df,"helpful_yes","total_vote")


df.sort_values("score_pos_neg_diff", ascending=False).head(20)

df.sort_values("score_average_rating", ascending=False).head(20)

#*
df.sort_values("wilson_lower_bound", ascending=False).head(20)