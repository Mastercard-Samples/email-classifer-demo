#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 11:43:58 2021

@author: Akshay Dubey
"""
import pandas as pd
import datetime
import re
from .content_scanner import Scanner

class ParseMailData:
    
    def __init__(self, path):
        df = pd.read_csv(path, encoding="ISO-8859-1")
        df.columns = ['Subject', 'Body', 'From: (Name)', 'From: (Address)',
                    'From: (Type)', 'To: (Name)', 'To: (Address)', 'To: (Type)',
                    'CC: (Name)', 'CC: (Address)', 'CC: (Type)', 'BCC: (Name)',
                    'BCC: (Address)', 'BCC: (Type)', 'Billing Information', 'Categories',
                    'Importance', 'Mileage', 'Sensitivity']
        df['Subject'] = df['Subject'].str.lower()
        obj = Scanner()
        data = obj.updating_subject(df)
        df_distinct = data.drop_duplicates(data.columns[0]).reset_index(drop=True)
        df_distinct = df_distinct.fillna('dummy')
        self.content_parsed = df_distinct.iloc[:, :2]
        self.parse_df = self.content_parsed
        
    def parse(self):
        day = 14
        month = 'August'
        year = 2020
        day_list = list()
        month_list = list()
        year_list = list()
        for data in self.content_parsed.Body:
            data = data.replace("\r\n"," ")
            
            # Fetching first occurrence of Date|Sent:
            string = re.search("([te]:.*\d{4})", data)
            
            if (string != None):
                # Format: 
                # Sent: 17 August 2020 02:03
                # Date: Monday 17 August 2020 at 10:27
                sent = re.search("^[te]:\s*\w* (\d+) ([aA-zZ]+) (\d+)", string.group(1), re.IGNORECASE)
                
                # Format:
                # Date|Sent: Friday, July 31, 2020 at 6:30 PM
                sent2 = re.search("^[te]: \w*, ([aA-zZ]+) (\d+), (\d+)", string.group(1), re.IGNORECASE)
                
                # Format:
                # Sent|Date: 02 24 2021 21:51
                sent3 = re.search("^[te]: (\d+) (\d+) (\d+)", string.group(1), re.IGNORECASE)
               
                if (sent and sent.group(1)):
                    day = sent.group(1)
                    month = sent.group(2)
                    year = sent.group(3)
                elif (sent2 and sent2.group(1)):
                    month = sent2.group(1)
                    day = sent2.group(2)
                    year = sent2.group(3)
                elif (sent3 and sent3.group(1)):
                    if (any(char.isdigit() for char in sent3.group(1))):
                        # Changing month number to month name
                        datetime_object = datetime.datetime.strptime(sent3.group(1), "%m")
                        month_name = datetime_object.strftime("%b")
                        month = month_name
                    else:
                        month = sent3.group(1)
                    day = sent3.group(2)
                    year = sent3.group(3)

            day_list.append(day)
            month_list.append(month)
            year_list.append(year)
        
        self.parse_df['Day'] = day_list
        self.parse_df['Day'] = self.parse_df['Day'].astype(str).astype(int)
        self.parse_df['Month'] = month_list
        # Slicing the month name to 3 characters
        self.parse_df['Month'] = self.parse_df['Month'].str.capitalize()
        self.parse_df['Month'] = self.parse_df['Month'].str.slice(stop=3)
        d = {'Jan':1, 'Feb':2, 'Mar':3, 'Apr':4, 'May':5, 'Jun':6, 'Jul':7,
             'Aug':8, 'Sep':9, 'Oct':10, 'Nov':11, 'Dec':12}
        # create column
        #self.parse_df['Month'] = self.parse_df['Month'].map(d)
        self.parse_df['Year'] = year_list
        self.parse_df['Year'] = self.parse_df['Year'].astype(str).astype(int)
        self.parse_df['Date'] = pd.to_datetime(self.parse_df.Year*10000+self.parse_df.Month.map(d)*100+self.parse_df.Day,format='%Y%m%d')
        
        return self.parse_df


# df = pd.read_pickle('../ML_testData/Predictions/knn_test.pickle')
# obj = ParseMailData(df)
# par = obj.parse()

# obj = ParseMailData('../ML_testData/test_data/ML_test_October2021.CSV')
# parse_dffffff = obj.parse()
# parse_dffffff.to_csv('testing.csv', index=False)
