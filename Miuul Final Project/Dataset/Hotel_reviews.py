# CSV dosyası 17 alan içeriyor. Her bir alanın açıklaması aşağıdaki gibidir:

# Hotel_Address: Otelin adresi.
# Review_Date: Yorumcunun ilgili yorumu gönderdiği tarih.
# Average_Score: Otelin son bir yıldaki en son yorumlara göre hesaplanan ortalama puanı.
# Hotel_Name: Otelin adı.
# Reviewer_Nationality: Yorumcunun milliyeti.
# Negative_Review: Yorumcunun otele verdiği olumsuz yorum. Yorumcu olumsuz yorum yapmamışsa, 'No Negative' olarak belirtilir.
# Review_Total_Negative_Word_Counts: Olumsuz yorumdaki kelime sayısının toplamı.
# Positive_Review: Yorumcunun otele verdiği olumlu yorum. Yorumcu olumlu yorum yapmamışsa, 'No Positive' olarak belirtilir.
# Review_Total_Positive_Word_Counts: Olumlu yorumdaki kelime sayısının toplamı.
# Reviewer_Score: Yorumcunun otel deneyimine dayanarak verdiği puan.
# Total_Number_of_Reviews_Reviewer_Has_Given: Yorumcunun geçmişte verdiği toplam yorum sayısı.
# Total_Number_of_Reviews: Otel için geçerli olan toplam yorum sayısı.
# Tags: Yorumcunun otele verdiği etiketler.
# days_since_review: Yorum tarihi ile veri çekme tarihi arasındaki süre.
# Additional_Number_of_Scoring: Bazı misafirler sadece hizmete puan verip yorum yapmamışlardır. Bu sayı, geçerli yorumsuz puanların sayısını gösterir.
# lat: Otelin enlem değeri.
# lng: Otelin boylam değeri.
# Metin verilerini temiz tutmak amacıyla, metin verilerindeki unicode karakterler ve noktalama işaretleri kaldırılmış ve metin küçük harfe dönüştürülmüştür. Başka bir ön işleme yapılmamıştır.

#Bu yukarıdaki avrupa verisinin hikayesi



import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_colwidth', None)

#Hotel Reviews https://www.kaggle.com/datasets/datafiniti/hotel-reviews
df1 = pd.read_csv("Proje_Odevi/Datasets/7282_1.csv")
df2 = pd.read_csv("Proje_Odevi/Datasets/Datafiniti_Hotel_Reviews.csv")
df3 = pd.read_csv("Proje_Odevi/Datasets/Datafiniti_Hotel_Reviews_Jun19.csv")

#515k hotel reviews data in europe https://www.kaggle.com/datasets/jiashenliu/515k-hotel-reviews-data-in-europe/data
df = pd.read_csv("Proje_Odevi/Datasets/Hotel_Reviews.csv")


df.head()
df.shape
df["Hotel_Name"].nunique()
df["Hotel_Address"].nunique()

df1.head()
df1.shape #10 bin satır 19 stun
df1["city"].value_counts() #virginia beach , newburgh, san antonio, newyork gibi yerlerde çok otel var
df1["city"].nunique() #761 farklı city mevcut
df2["primaryCategories"].value_counts()
df2["primaryCategories"].nunique() # 6 kategoriye ayrışmış ama büyük çoğunluk Accommodation & Food Services
df2["country"].value_counts()
df2["country"].nunique() # tek bir country var. Sadece amerikadaki oteller var.

df2.head()
df2.shape #10 bin satır 25 stun
df2["city"].value_counts()  #las vegas chicago virginia beach gibi yerlerde çok otel var.
df2["city"].nunique() #1021 farklı city mevcut
df2["primaryCategories"].value_counts()
df2["primaryCategories"].nunique() # 6 kategoriye ayrışmış ama büyük çoğunluk Accommodation & Food Services
df2["country"].value_counts()
df2["country"].nunique() # tek bir country var. Sadece amerikadaki oteller var.

df3.head()
df3.shape #10 bin satır 26 stun
df3["city"].value_counts()  #san diego san francisco new orleans atlanta gibi yerlerde çok otel var
df3["city"].nunique() #842 farklı city mevcut
df3["primaryCategories"].value_counts()
df3["primaryCategories"].nunique() # 4 kategoriye ayrışmış ama büyük çoğunluk Accommodation & Food Services
df3["country"].value_counts()
df3["country"].nunique() # tek bir country var. Sadece amerikadaki oteller var.