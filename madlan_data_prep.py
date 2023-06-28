def prepare_data():
    import pandas as pd
    import re
    import numpy as np
    from datetime import datetime 
    data=pd.read_csv('output_all_students_Train_v10.csv')
    raw_df=pd.DataFrame(data)
    
    print(raw_df.info())
    
    #-------------------------------------------------------------------------------------------------------------------------------------------
    #cleaning the data:
    raw_df['price'] = pd.to_numeric(raw_df['price'], errors='coerce')
    raw_df['Area'] = pd.to_numeric(raw_df['Area'], errors='coerce')
    Nulls_list=['nan','Na','NULL','NAN','NaN','Nan']
    new_data = raw_df.replace(Nulls_list, np.nan).copy()
    new_data = new_data.dropna(subset=['price'])
    new_text_cols = ['Street', 'description ', 'city_area','City']
    for col in new_text_cols:
        new_data.loc[:, col] = new_data[col].apply(lambda x: re.sub(r'\d+|\W+', ' ', str(x)))
    new_data['floor_out_of']=new_data['floor_out_of'].astype('string')
    new_data.loc[:, 'floor'] = new_data['floor_out_of'].apply(lambda x: int(re.search(r'\b(\d+)\b', str(x)).group()) if re.search(r'\b(\d+)\b', str(x)) else None)
    new_data.loc[new_data['floor_out_of'] == 'קומת קרקע', 'floor'] = 0
    
    new_data.loc[:, 'total_floors'] = new_data['floor_out_of'].apply(lambda x: int(re.findall(r'\d+', str(x))[-1]) if re.findall(r'\d+', str(x)) else None)
    new_data.loc[new_data['floor_out_of'] == 'קומת קרקע', 'total_floors'] = 0
    
    new_data[['total_floors', 'floor', 'floor_out_of']]
    new_data=new_data.drop(columns='floor_out_of')
    
    new_data['room_number'] = new_data['room_number'].apply(lambda x: int(re.findall(r'\d+', str(x))[0]) if re.findall(r'\d+', str(x)) else None)
    new_data['number_in_street'] = new_data['number_in_street'].apply(lambda x: int(re.findall(r'\d+', str(x))[0]) if re.findall(r'\d+', str(x)) else None)
    new_data['Area']=new_data['room_number'].apply(lambda x: int(re.findall(r'\d+', str(x))[0]) if re.findall(r'\d+', str(x)) else None)
    new_data['publishedDays ']=new_data['publishedDays '].apply(lambda x: int(re.findall(r'\d+', str(x))[0]) if re.findall(r'\d+', str(x)) else None)
    new_data['condition '] = new_data['condition '].replace({'משופץ': 'renovated', 'שמור': 'conserved', 'חדש': 'new','FALSE':None,False:None,'לא צויין': ' not_defined','ישן':'old'})
    new_data['furniture '] =new_data['furniture '].replace({'חלקי': 'partial', 'מלא': 'full', 'לא צויין': ' not_defined','אין':None})
    new_data['entranceDate '] = new_data['entranceDate '].str.strip()
    new_data['entranceDate '] = new_data['entranceDate '].replace({'גמיש': 'flexible', 'לא צויין': 'not_defined','מיידי':'less_than_6_months'})
    
    # Convert current_date to pandas.Timestamp
    current_date = pd.Timestamp(datetime.now().date())
    
    # Define categories using conditional statements
    new_data['entrance_date'] = new_data['entranceDate ']
    new_data.loc[(new_data['entranceDate '] != 'flexible') & (new_data['entranceDate '] != 'not_defined'), 'entranceDate '] = pd.to_datetime(new_data['entranceDate '],format='%Y-%m-%d', errors='coerce')
    new_data['months_diff'] = ((current_date - pd.to_datetime(new_data['entranceDate '], errors='coerce')).dt.days / 30).astype(float)
    new_data.loc[(new_data['months_diff'] < 6), 'entrance_date'] = 'less_than_6_months'
    new_data.loc[(new_data['months_diff'] >= 6) & (new_data['months_diff'] <= 12), 'entrance_date'] = 'months_6_12'
    new_data.loc[new_data['months_diff'] > 12, 'entrance_date'] = 'above_year'
    new_data=new_data.drop('entranceDate ',axis=1)
    new_data=new_data.drop('months_diff',axis=1)
    
    ones_list = ['יש מחסן', True, 'Yes', 'yes', 'כן','יש','יש סורגים','יש מעלית','יש ממ״ד','יש מרפסת','נגיש לנכים','יש מיזוג אויר']
    new_data[['hasStorage ','hasParking ','hasElevator ','hasMamad ','hasBalcony ','hasAirCondition ','handicapFriendly ','hasBars ']]= np.where(new_data[['hasStorage ','hasParking ','hasElevator ','hasMamad ','hasBalcony ','hasAirCondition ','handicapFriendly ','hasBars ']].isin(ones_list), 1, 0)
    
    
    Bools_list=['hasStorage ','hasParking ','hasElevator ','hasMamad ','hasBalcony ','hasAirCondition ','handicapFriendly ','hasBars ']
    ones_list = ['יש מחסן', True, 'Yes', 'yes', 'כן','יש','יש סורגים','יש מעלית','יש ממ״ד','יש מרפסת','נגיש לנכים','יש מיזוג אויר']
    new_data[Bools_list]= np.where(new_data[Bools_list].isin(ones_list), 1, 0)
    #-----------------------------------------------------------------------------------------------------------------------------------------------
    #Data Visualization:
    import seaborn as sns
    import matplotlib.pyplot as plt
    print(new_data.head())
    
    #heatmap
    #after some heatmaps attempts, we found that the following columns were less correlative with the price column, therefore took them out for better corr visualisation:
    smaller_data= new_data.drop(['number_in_street','hasMamad ','num_of_images','entrance_date','handicapFriendly ','hasBars ','floor', 'total_floors','hasElevator ','hasParking ','hasAirCondition ','publishedDays ','hasStorage '],axis=1)
    num_cols = smaller_data[[col for col in smaller_data.columns if smaller_data[col].dtypes != 'O' and smaller_data[col].dtypes != 'category']]
    cor= num_cols.corr()
    print(sns.heatmap(cor))
    #price is a bit more correlated with room number, area, balcony and mamad. lets check them.
    # numericals = [col for col in smaller_data.columns if smaller_data[col].dtypes!='O' ]
    # categorials = [col for col in smaller_data.columns if smaller_data[col].dtypes=='O' ]
    
    print(sns.scatterplot(x=smaller_data[ 'Area'], y=smaller_data['price']))
    print(sns.scatterplot(x=smaller_data['room_number'], y=smaller_data['Area']))
    
    smaller_data=smaller_data.loc[smaller_data['room_number'] <10]
    
    plt.figure(figsize=(25, 6))
    sorted_nb = smaller_data.groupby(['City'])['price'].median().sort_values(ascending=False)
    print(sns.boxplot(x=smaller_data['City'], y=smaller_data['price'], order=list(sorted_nb.index)))
    #after seeing the boxplot ans realizing there are aptmnts in Dimona with a price too high - taking them off:
    smaller_data = smaller_data.loc[~((smaller_data['City'] == 'דימונה') & (smaller_data['price'] >= 1000000))]
    
    plt.figure(figsize=(20, 6))
    sorted_type = smaller_data.groupby(['type'])['price'].median().sort_values(ascending=False)
    print(sns.boxplot(x=smaller_data['type'], y=smaller_data['price'], order=list(sorted_type.index)))
    smaller_data=smaller_data.loc[~((smaller_data['type'] == 'דירה') & (smaller_data['price'] >= 10000000)&(smaller_data['City'] != 'תל אביב'))]
    return smaller_data