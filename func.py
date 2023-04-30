import pandas as pd
import numpy as np
import plotly.express as px
import re
from fuzzywuzzy import fuzz

#visualize combined data sets to choropleth
def vis(ld,rd):
    ld_vis = ld[["name","state"]]
    rd_vis = rd[["name","state"]]
    md_vis = ld_vis.append(rd_vis)
    md_vis = md_vis.groupby('state').size().reset_index(name='count')
    md_vis["count"] = md_vis["count"].astype(int)
    fig = px.choropleth(
    md_vis,
    locations='state',
    locationmode='USA-states',
    color='count',
    color_continuous_scale='oranges',
    scope='usa',
    title='Choropleth Map of Companies by State',
    labels={'count': 'Number of Companies', 'state': 'State'}
    )
    fig.write_image("company_visualization.png")



#algorithm to find similarity
def alg(ld,rd):
    #select needed columns
    ld = ld[["entity_id","name","address","city","state","postal_code"]]
    rd = rd[["business_id","name","address","city","state","zip_code"]]

    #process zip code column
    ld = ld[ld["postal_code"].notna()]
    ld["postal_code"] = ld["postal_code"].astype(int)
    def clean_zc(x):
        x = int(x.split("-")[0])
        return x
    rd["zip_code"] = rd["zip_code"].apply(clean_zc)    
    rd = rd.rename({"zip_code":"postal_code"},axis = 1)

    #cleaning
    ld["name"] = ld["name"].str.lower().str.replace('[^a-zA-Z0-9 \n\.]', '')
    ld["address"] = ld["address"].str.lower()
    ld["city"] = ld["city"].str.lower()
    ld["state"] = ld["state"].str.lower()
    rd["name"] = rd["name"].str.lower().str.replace('[^a-zA-Z0-9 \n\.]', '')
    rd["address"] = rd["address"].str.lower()
    rd["city"] = rd["city"].str.lower()
    rd["state"] = rd["state"].str.lower()

    #clean address and keep values before the comma
    def clean_address(x):
        if x is np.nan:
            x = ""
        else:
            x = x.strip()
            x = x.split(", ")[0]
        return x

    ld["address"] = ld["address"].apply(clean_address)
    rd["address"] = rd["address"].apply(clean_address)

    #clean name and remove inc, llc, ltd
    def clean_name(x):
        x = x.strip()
        x = re.sub(r'\b(inc|llc|ltd)\b', '', x)
        return x.strip()

    ld["name"] = ld["name"].apply(clean_name)
    rd["name"] = rd["name"].apply(clean_name)

    #extract first word of the name
    def fword(x):
        if x is not np.nan:
            x = x.strip()
            x = x.split(" ")[0]
        else:
            x = ""
        return x

    ld["fname"] = ld["name"].apply(fword)
    ld["faddress"] = ld["address"].apply(fword)
    rd["fname"] = rd["name"].apply(fword)
    rd["faddress"] = rd["address"].apply(fword)

    #merge dataset
    mdf = ld.merge(rd, on=['postal_code', 'city','state','fname','faddress'])

    #preparation for fuzzywuzzy
    mdf['address_x'] = mdf['address_x'].apply(lambda x: str(x).replace(' ', '') if isinstance(x, str) else x)
    mdf['name_x'] = mdf['name_x'].apply(lambda x: str(x).lower().replace(' ', ''))
    mdf['address_y'] = mdf['address_y'].apply(lambda x: str(x).replace(' ', '') if isinstance(x, str) else x)
    mdf['name_y'] = mdf['name_y'].apply(lambda x: str(x).replace(' ', ''))

    # create a new column with calculated string similarity
    mdf['name_sim'] = mdf.apply(lambda row: fuzz.partial_ratio(row['name_x'], row['name_y']), axis=1)
    mdf['address_sim'] = mdf.apply(lambda row: fuzz.partial_ratio(row['address_x'], row['address_y']), axis=1)
    
    #leave wanted records and output
    mdf["score"] = (mdf['name_sim'] + mdf['address_sim'])/200
    mdf = mdf[mdf["score"]>= 0.8].sort_values("score",ascending = False)
    fmdf = mdf[["entity_id","business_id","score"]]
    fmdf = fmdf.rename(columns={'entity_id': 'left_id', 'business_id': 'right_id'})
    fmdf = fmdf.reset_index(drop = True)
    fmdf.to_csv('result_df.csv', index=True)