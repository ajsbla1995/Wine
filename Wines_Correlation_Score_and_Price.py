#!/usr/bin/env python
# coding: utf-8

# In[386]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os as os
import seaborn as sns
import mysql.connector as sql
from wordcloud import WordCloud, STOPWORDS




# # -------------------------------------------------------------

# ## Connect to the mysql session and select the necessary queries to export to excel

# 

# In[388]:


os.chdir('/Users/andresabala/Downloads/Data Analysis Projects/Wine Project')


# In[389]:


db_connection = sql.connect(host='localhost', database='Wine', user='root', password='')
db_cursor= db_connection.cursor()


# In[390]:


db_cursor.execute('SELECT * FROM wine_ranks_per_varietal')
table_rows = db_cursor.fetchall()
winery_df = pd.DataFrame(table_rows)
winery_df.dtypes                        
winery_df[7]=pd.to_numeric(winery_df[7])
winery_df.dtypes
winery_df

#When trying to aggregate Price/Bottle (column=6), inf was returned because the datatype was an object. 
    #Converted datatype to float64, and was able to aggregate 



# ## Determine the overall correlation between Score and Price for all rows in winery_df

# In[391]:


correlation_overall=(winery_df
 .rename(columns={0:'Country', 1:'Province', 2:'Region',3:'Winery', 4: 'Wine', 5:'Varietal', 6:'Score/Bottle',7:'Price/Bottle', 8:'Rank'})
 .loc[:,['Score/Bottle', 'Price/Bottle']]
 #.mean()
 #.agg({'Score/Bottle':np.mean,'Price/Bottle':np.mean}) 
 .corr()
                    )

correlation_overall


# ## Insights
#        -- The overall correlation of Score and Price for all wines reviewed is +0.43, suggesting a Moderate Positive Correlation--> meaning that as price goes up, the score given may also go up (more possibility than by just chance alone).

# In[ ]:





# # Plot Avg Score vs Price per Varietal Wine for Each Country

# In[392]:


country_score_price_plot=(winery_df
 .rename(columns={0:'Country', 1:'Province', 2:'Region',3:'Winery', 4: 'Wine', 5:'Varietal', 6:'Score/Bottle',7:'Price/Bottle', 8:'Rank'})
 .groupby(['Varietal'])
 .agg({'Score/Bottle':'mean','Price/Bottle':'mean'})        
                         )

country_score_price_plot
sns.relplot(data=country_score_price_plot, x='Score/Bottle', y='Price/Bottle', hue='Varietal').set(title='Avg Price and Score/Bottle per Varietal')


# ## Insights:
#        --Riesling scored higher on average than varietals that were priced similarly (i.e. Merlot, Tempranillo, Zinfandel, Malbec). Additionally, Riesling was in the top 3 of highest scoring varietals, despite being in the lower 55% with respect to price/bottle
#         --Pinot Noir was overall the highest-rated varietal while Cabernet Sauvignon was the highest-priced bottle
#         --Merlot was the lowest-rated varietal and Sauvignon Blanc was the lowest-priced bottle

# # Determine Correlation for each Varietal 

# In[393]:


varietal_correlation=(winery_df
 .rename(columns={0:'Country', 1:'Province', 2:'Region',3:'Winery', 4: 'Wine', 5:'Varietal', 6:'Score/Bottle',7:'Price/Bottle', 8:'Rank'})
 #.loc[:,['Varietal','Wine','Winery', 'Score/Bottle', 'Price/Bottle']]
 .groupby(['Varietal'])[['Price/Bottle', 'Score/Bottle']]
 #.mean()
 #.agg({'Score/Bottle':np.mean,'Price/Bottle':np.mean})
 .corr()
               )

varietal_correlation




# ## Insights
#     -- Cabernet Sauvignon had the strongest correlation between score and price, followed by Tempranillo and then 
#            Malbec
#     -- The lowest correlation between score and price is seen in Pinot Noir, followed by Zinfandel and then Syrah

# # Correlation for each Varietal by Country

# In[394]:


varietal_country_correlation=(winery_df
 .rename(columns={0:'Country', 1:'Province', 2:'Region',3:'Winery', 4: 'Wine', 5:'Varietal', 6:'Score/Bottle',7:'Price/Bottle', 8:'Rank'})
 .groupby(['Varietal', 'Country'])[['Price/Bottle', 'Score/Bottle']]
 .corr()
 .replace(1.0,'NaN')
 #.plot(rot=45)
               )


(sns.relplot(data=varietal_country_correlation, x='Country', y='Score/Bottle', hue='Varietal')
.set(title='Correlation between Price and Score for Varietals in Each Country')
)
plt.xticks(rotation=45)

varietal_country_correlation


# ## Insights:
#     --r-value of Score to Price (of all wines in the dataframe)= +0.43. This indicates a moderate-strong postive relationship. As price goes up, score is may likely go up as well and vice versa.

# # Import wine_price_segment query from MySQL, convert object dtypes to numeric

# In[423]:


db_cursor.execute('SELECT * FROM wine_price_segments')
table_rows = db_cursor.fetchall()
segments_df = pd.DataFrame(table_rows)
segments_df[1]=pd.to_numeric(segments_df[1])
segments_df[2]= pd.to_numeric(segments_df[2])
segments_df=segments_df.rename(columns={0:'Country', 1:'Avg_Score/Bottle', 2:'Avg_Price/Bottle', 3:'Price_Segment'})
segments_df

# to view the wine price segments --> https://media.winefolly.com/wine-pricing-segments.jpg


# In[422]:


segments_plot=(segments_df
 .groupby(['Country'])[['Avg_Price/Bottle','Avg_Score/Bottle','Price_Segment']]
 .plot(secondary_y=['Avg Price/Bottle', 'Avg Score/Bottle'])
)



# In[ ]:





# # Visualization of Price Segments

# In[397]:


db_cursor.execute('SELECT * FROM price_segment_countries')
table_rows = db_cursor.fetchall()
segments_df = pd.DataFrame(table_rows)
segments_df.dtypes                
segments_df[3]=pd.to_numeric(segments_df[3])
segments_df[4]=pd.to_numeric(segments_df[4])
#change columns 3 and 4 to numeric


# In[398]:


segments_df


# 

# In[399]:


segments_plot=(segments_df
.rename(columns= {0:'country', 1:'price_segment', 2:'#_wines_in_segment', 3:'avg_score/bottle', 4:'avg_price/bottle'})
.loc[:,['country', 'price_segment', 'avg_score/bottle','avg_price/bottle']]
.set_index('country')

#.plot.scatter(x='avg_price/bottle', y='avg_score/bottle')
)

sns.relplot(data=segments_plot, x='avg_score/bottle', y='avg_price/bottle', hue='price_segment', style='country').set(title='Avg Price, Score/Bottle per Country for Each Price Segment')




# ## Insights:
#   --Germany consistently had higher scores on avg in each price segment compared to its peers, except for the
#          Icon price segment.
#          
#  -- Portugal was in the top 2 in avg score/bottle in the Popular Premium and Value price segments. However, 
#           Portugal tended to have lower avg score/bottle in both Super Luxury and Luxury segments.
#           
#  -- France, Germany and Spain (all Old World wines) had the highest scores in the Icon segment. 
#  
#  -- Old World Wines (France, Italy, Germany, Spain, Portugal) tended to occupy the top 3 spots with respect to 
#            avg score/bottle in the Ultra Premium, Luxury, Super Luxury, and Icon Price Segments
#            
#  -- New World Wines (Australia, Argentina, the US) tended to do better in Extreme Value, Value, Popular Premium and Super Premium segments 
#       
#         

# In[ ]:





# ## Make Word Cloud of for each wine in each Country wine segment

# In[410]:


db_cursor.execute('SELECT * FROM word_cloud')
table_rows = db_cursor.fetchall()
word_df = pd.DataFrame(table_rows)
word_df=word_df.rename(columns={0:'Country', 1:'Varietal', 2:'Wine', 3:'Winery', 4: 'Description', 5:'Score', 6:'Price',7:'Wine_Segment'})                        
word_df.dtypes
#word_df
word_df



# In[689]:


# get all the reviews into one variable
#use for loop to iterate over each description, return the word cloud for for each wine segment



stop_words=set(STOPWORDS)
stop_words= ['fruit', 'show', 'shows','note','notes', 'nose', 'finish', 'aroma', 'aromas', 'palate','flavor', 'flavors','drink', 'wine',' wine','wine ', 'Drink'] + list(STOPWORDS)

text_eval= ''.join(description for description in word_df[word_df['Wine_Segment']=='Extreme Value']['Description'].astype(str))
text_val= ''.join(description for description in word_df[word_df['Wine_Segment']=='Value']['Description'].astype(str))
text_pop= ''.join(description for description in word_df[word_df['Wine_Segment']=='Popular Premium']['Description'].astype(str))
text_prem= ''.join(description for description in word_df[word_df['Wine_Segment']=='Premium']['Description'].astype(str))
text_sup_prem= ''.join(description for description in word_df[word_df['Wine_Segment']=='Super Premium']['Description'].astype(str))
text_ult_prem= ''.join(description for description in word_df[word_df['Wine_Segment']=='Ultra Premium']['Description'].astype(str))
text_lux= ''.join(description for description in word_df[word_df['Wine_Segment']=='Luxury']['Description'].astype(str))
text_sup_lux= ''.join(description for description in word_df[word_df['Wine_Segment']=='Super Luxury']['Description'].astype(str))
text_icon= ''.join(description for description in word_df[word_df['Wine_Segment']=='Icon']['Description'].astype(str))


wordcloud_eval= WordCloud(stopwords=stop_words).generate(text_eval)
wordcloud_val= WordCloud(stopwords=stop_words).generate(text_val)
wordcloud_pop= WordCloud(stopwords=stop_words).generate(text_pop)
wordcloud_prem= WordCloud(stopwords=stop_words).generate(text_prem)
wordcloud_sup_prem= WordCloud(stopwords=stop_words).generate(text_sup_prem)
wordcloud_ult_prem= WordCloud(stopwords=stop_words).generate(text_ult_prem)
wordcloud_lux= WordCloud(stopwords=stop_words).generate(text_lux)
wordcloud_sup_lux= WordCloud(stopwords=stop_words).generate(text_sup_lux)
wordcloud_icon= WordCloud(stopwords=stop_words).generate(text_icon)


# ## Visualize the Word Cloud
# 

# In[1118]:


def generate_wordcloud(wine_segment):
    wc= WordCloud(stopwords=stop_words).generate(''.join(description for description in word_df[word_df['Wine_Segment']==wine_segment]['Description'].astype(str)))
    for segment in wine_segment:
        if wine_segment== 'Extreme Value':
            return wc 
        if wine_segment == 'Value':
            return wc
        if wine_segment == 'Popular Premium':
            return wc
        if wine_segment == 'Premium':
            return wc
        if wine_segment == 'Super Premium':
            return wc
        if wine_segment == 'Ultra Premium':
            return wc
        if wine_segment == 'Luxury':
            return wc
        if wine_segment == 'Super Luxury':
            return wc
        if wine_segment == 'Icon':
            return wc

def display_wordcloud(wine_segment):
    wc= WordCloud(stopwords=stop_words).generate(''.join(description for description in word_df[word_df['Wine_Segment']==wine_segment]['Description'].astype(str)))
    
    for segment in wine_segment:
        if wine_segment== 'Extreme Value':
            return wc 
        if wine_segment == 'Value':
            return wc
        if wine_segment == 'Popular Premium':
            return wc
        if wine_segment == 'Premium':
            return wc
        if wine_segment == 'Super Premium':
            return wc
        if wine_segment == 'Ultra Premium':
            return wc
        if wine_segment == 'Luxury':
            return wc
        if wine_segment == 'Super Luxury':
            return wc
        if wine_segment == 'Icon':
            return wc
        plt.subplot(j, 9, i).set_title('Plot #'+ str(s))
        plt.plot()
        plt.imshow(wc)
        plt.axis='off'
        fig.suptitle(title)
        plt.show()

#generate_wordcloud('Extreme Value')
display_wordcloud('Value')


# # Use function to make code more efficient

# In[1044]:


stop_words=set(STOPWORDS)
stop_words= ['will', 'fruit', 'show', 'shows','note','notes', 'nose', 'finish', 'aroma', 'aromas', 'palate','flavor', 'flavors','drink', 'wine',' wine','wine ', 'Drink'] + list(STOPWORDS)

def generate_wordcloud(wine_segment):
    wc= WordCloud(stopwords=stop_words).generate(''.join(description for description in word_df[word_df['Wine_Segment']==wine_segment]['Description'].astype(str)))
    return wc

extr_value= generate_wordcloud('Extreme Value') 
#value= generate_wordcloud('Value')
#pop_prem=generate_wordcloud('Popular Premium')
#prem=generate_wordcloud('Premium')
#sup_prem=generate_wordcloud('Super Premium')
#ult_prem=generate_wordcloud('Ultra Premium')
#lux=generate_wordcloud('Luxury')
#sup_lux= generate_wordcloud('Super_Luxury')
#icon=generate_wordcloud('Icon')

#extr_value= generate_wordcloud('Extreme Value')
#value = generate_wordcloud('Value')
#pop_prem= generate_wordcloud('Popular Premium')
#prem= generate_wordcloud('Premium')
#sup_prem= generate_wordcloud('Super Premium')
#ult_prem= generate_wordcloud('Ultra Premium')
#lux= generate_wordcloud('Luxury')
#sup_lux= generate_wordcloud('Super_Luxury')
#icon= generate_wordcloud('Icon')

def display_wordcloud(segment_generate, title):
    plt.figure()
    #j= np.ceil(num_plots/9)
    #for s in range(num_plots):
        #i=s+1
        #plt.subplot(j, 9, i).set_title('Plot #'+ str(s))
    plt.plot()
    plt.imshow(segment_generate)
    plt.axis='off'
    fig.suptitle(title)
    plt.show()

display_wordcloud(extr_value, 'extreme')
#display_wordcloud(value, 'value')


# In[691]:


plt.suptitle('Most Used Descriptors used per Price Segment')


# Extreme Value

plt.subplot(121)
plt.axis("off")
#plt.figure( figsize=(10,20))
plt.title('Extreme Value')
plt.tight_layout(pad=0)
plt.imshow(wordcloud_eval, interpolation='bilinear') # change this value to show different variables 
plt.show()

# Value

plt.subplot(121)
plt.axis("off")
#plt.figure( figsize=(10,20))
plt.title('Value')
plt.tight_layout(pad=0)
plt.imshow(wordcloud_val, interpolation='bilinear') # change this value to show different variables 
plt.show()

# Popular Premium

plt.subplot(121)
plt.axis("off")
#plt.figure( figsize=(10,20))
plt.title('Popular Premium')
plt.tight_layout(pad=0)
plt.imshow(wordcloud_pop, interpolation='bilinear') # change this value to show different variables 
plt.show()

# Premium

plt.subplot(121)
plt.axis("off")
#plt.figure( figsize=(10,20))
plt.title('Premium')
plt.tight_layout(pad=0)
plt.imshow(wordcloud_prem, interpolation='bilinear') # change this value to show different variables 
plt.show()

# Super Premium

plt.subplot(121)
plt.axis("off")
#plt.figure( figsize=(10,20))
plt.title('Super Premium')
plt.tight_layout(pad=0)
plt.imshow(wordcloud_sup_prem, interpolation='bilinear') # change this value to show different variables 
plt.show()

# Ultra Premium

plt.subplot(121)
plt.axis("off")
#plt.figure( figsize=(10,20))
plt.title('Ultra Premium')
plt.tight_layout(pad=0)
plt.imshow(wordcloud_ult_prem, interpolation='bilinear') # change this value to show different variables 
plt.show()

# Luxury

plt.subplot(121)
plt.axis("off")
#plt.figure( figsize=(10,20))
plt.title('Luxury')
plt.tight_layout(pad=0)
plt.imshow(wordcloud_lux, interpolation='bilinear') # change this value to show different variables 
plt.show()

# Super Luxury

plt.subplot(121)
plt.axis("off")
#plt.figure( figsize=(10,20))
plt.title('Super Luxury')
plt.tight_layout(pad=0)
plt.imshow(wordcloud_sup_lux, interpolation='bilinear') # change this value to show different variables 
plt.show()

# Icon

plt.subplot(121)
plt.axis("off")
#plt.figure( figsize=(10,20))
plt.title('Icon')
plt.tight_layout(pad=0)
plt.imshow(wordcloud_icon, interpolation='bilinear') # change this value to show different variables 
plt.show()




# ## Insights
# 
# --The descriptors used for the cheapest price range segments (Extreme Value and Value) tended to emphasize berry and fruit flavors (i.e. 'plum' , 'berry', 'cherry', 'plum', 'apple', etc.). In the higher price range segments, especially Luxury to Icon, descriptors words de-emphasize berry flavors and instead emphasize mouth-feel descriptors such as 'acidity' , 'tannin', ' 'structure' , 'rich' , 'ripe'.
# 
# -- The descriptor 'tannin' became more prevalent in judge descriptions after the Popular Premium wine segment. Interestingly, 'tannin' appears to become more important as the wine price segment increases
# 
# -- More ubiquitous terms in the most expensive price segments (Premium to Icon) are 'ripe' and 'rich'
# 

# ## WordClouds of Descriptors in Score Ranges

# ## Make a function
#         1. Group all the wines into their respective score ranges (80-84, 85-89, 90-94, 95-100) using bins
#         2. For each price group, make a word cloud of the Description column 
#       
#             

# In[970]:


# select the dataframe to show Score and Description columns

desc_score= word_df[['Score', 'Description']]
bins= [80,85,90,95,100]
labels= ['Low 80s', 'High 80s', 'Low 90s', 'High 90s']

# Use .cut() to section the wines into bins based off of Scores, concatenate all descriptions for each bin

grouped= desc_score.groupby(pd.cut(df_1['Score'],bins=bins, labels=labels, include_lowest= True))

concatenated= grouped['Description'].apply(lambda x: ''.join(row for row in x.astype(str)))

#make DataFrame of this Series ^

concatenated_df= pd.DataFrame([concatenated])



#pts_bins= pd.cut(df_1['Score'],[80,85,90,95,100])

#final= word_df.groupby(pts_bins)['Description'].apply(lambda x:''.join(x.astype(str)))
#df_2=pd.DataFrame([final])

# reset index of concatenated_df 
concatenated_reset_df = concatenated.reset_index()
concatenated_reset_df

# Check datatype of each row in column 'Description', ensure that it is a string so can be used in WordCloud
type(concatenated_reset_df['Description'][0])
type(concatenated_reset_df['Description'][1])
type(concatenated_reset_df['Description'][2])
type(concatenated_reset_df['Description'][3])






# In[ ]:





# In[979]:


stop_words=set(STOPWORDS)
stop_words= ['will', 'fruit', 'show', 'shows','note','notes', 'nose', 'finish', 'aroma', 'aromas', 'palate','flavor', 'flavors','drink', 'wine',' wine','wine ', 'Drink'] + list(STOPWORDS)

low_80s= concatenated_reset_df['Description'][0]
low_80s

high_80s=  concatenated_reset_df['Description'][1]
high_80s

low_90s= concatenated_reset_df['Description'][2]
low_90s

high_90s= concatenated_reset_df['Description'][3]
high_90s



wc_low80= WordCloud(stopwords=stop_words).generate(low_80s)
wc_low80
wc_high80= WordCloud(stopwords=stop_words).generate(high_80s)
wc_low90= WordCloud(stopwords=stop_words).generate(low_90s)
wc_high90= WordCloud(stopwords=stop_words).generate(high_90s)
wc_high90


# In[ ]:





# In[980]:


plt.suptitle('Most Used Descriptors used per Score Range')


# 80-84

plt.subplot()
plt.axis("off")
#plt.figure( figsize=(10,20))
plt.title('Scores: 80-84')
plt.tight_layout(pad=0)
plt.imshow(wc_low80, interpolation='bilinear') # change this value to show different variables 
plt.show()

# 85-89

plt.subplot()
plt.axis("off")
#plt.figure( figsize=(10,20))
plt.title('Scores: 85-89')
plt.tight_layout(pad=0)
plt.imshow(wc_high80, interpolation='bilinear') # change this value to show different variables 
plt.show()

# 90-94

plt.subplot()
plt.axis("off")
#plt.figure( figsize=(10,20))
plt.title('Scores:90-94')
plt.tight_layout(pad=0)
plt.imshow(wc_low90, interpolation='bilinear') # change this value to show different variables 
plt.show()

# 95-100

plt.subplot()
plt.axis("off")
#plt.figure( figsize=(10,20))
plt.title('Scores: 95-100')
plt.tight_layout(pad=0)
plt.imshow(wc_high90, interpolation='bilinear') # change this value to show different variables 
plt.show()



# ## Insights
# -- The descriptors used for wines in the highest score range (95-100) emphasize 'tannin', 'rich', 'acidity' and 'ripe' and 'now'. These are descriptors that usually correspond with age and complexity, suggesting that older wines may net better scores. Future research is needed to test this hypothesis.
# 
# -- In the lowest score range (80-84), the descriptors 'soft', 'sweet', 'acidity' and 'dry' are dominant. Interestingly, many of the descriptors correspond to flavors 
# 
# -- In the highest score range (95-100), terms that indicate a higher level of detail such as 'vineyard' and 'vintage' have higher frequencies in wine review descriptions than at lower score ranges
# 
# 

# In[ ]:





# In[ ]:





# In[ ]:



        
    


# In[ ]:





# In[ ]:


#with pd.ExcelWriter('WineTableauData.xlsx', mode='a') as writer:
   # provinces_df.to_excel(writer, sheet_name='provinces_details')
with pd.ExcelWriter('WineTableauDataFinal.xlsx', mode='a') as writer:
    winery_df.to_excel(writer, sheet_name='cabsauv_incl_rank_per_varietal')


