#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 18:34:48 2021

@author: Akshay Dubey
"""

import pandas as pd
import re
from random import random

class Scanner:
    
    def __init__(self):
        self.counter  = 0
        
    def content_scan_body(self, body):
        all_category_df, category_df = self.category(body)
        return all_category_df, category_df

    
    def content_scan_subject(self, body):
        all_category_df, category_df = self.category(body)
        return all_category_df, category_df
      
    
    def category(self, body):
        all_category_df = pd.DataFrame({})
        category_df = pd.DataFrame({})
        # regex_mc_dev_notification = re.findall(('Attention: New DFSA Submission for|Mastercard Developers Invitation|Mastercard Developers Sandbox Enabler'), str(columnData), re.IGNORECASE)
        for columnData in body:    
            if (re.findall(('onboarding|exchange|feature|offboarding|offboarded|decommission(ing|ed)?'), str(columnData), re.IGNORECASE)):
                all_category_df = all_category_df.append({'Onboarding generic queries' : 1}, ignore_index=True)
                category_df = category_df.append({'Category' : '2'}, ignore_index=True)
            elif (re.findall(('zally|rule|Linter Tool'), str(columnData), re.IGNORECASE)):
                all_category_df = all_category_df.append({'Zally' : 1}, ignore_index=True)
                category_df = category_df.append({'Category' : '6'}, ignore_index=True)
            elif (re.findall(('Access|tool'), str(columnData), re.IGNORECASE)):
                all_category_df = all_category_df.append({'Access to Tool queries' : 1}, ignore_index=True)
                category_df = category_df.append({'Category' : '4'}, ignore_index=True)
            elif (re.findall(('Assessment|early|(re)?score|self|rescoring|exception(s)? request(s)?|exception(s)'), str(columnData), re.IGNORECASE)):
                all_category_df = all_category_df.append({'Assessment/rescore queries/early spec/exception requests' : 1}, ignore_index=True)
                category_df = category_df.append({'Category' : '3'}, ignore_index=True)
            elif (re.findall(('Jamstack content|scan|reviewer|JAMStack Content Reviewer Scan'), str(columnData), re.IGNORECASE)):
                all_category_df = all_category_df.append({'Jamstack content reviewer' : 1}, ignore_index=True)
                category_df = category_df.append({'Category' : '8'}, ignore_index=True)
            elif (re.findall(('proxy|GW|configuration|API\s?[Gateway|GW]|XML\s?'), str(columnData), re.IGNORECASE)):
                all_category_df = all_category_df.append({'Service Proxy troubleshooting / APIGW' : 1}, ignore_index=True)
                category_df = category_df.append({'Category' : '1'}, ignore_index=True)
            elif (re.findall(('github|lib'), str(columnData), re.IGNORECASE)):
                all_category_df = all_category_df.append({'Client libs' : 1}, ignore_index=True)
                category_df = category_df.append({'Category' : '7'}, ignore_index=True)
            elif (re.findall(('Axon Topic|Axon control panel|Axon'), str(columnData), re.IGNORECASE)):
                all_category_df = all_category_df.append({'Axon Queries' : 1}, ignore_index=True)
                category_df = category_df.append({'Category' : '9'}, ignore_index=True)
            # elif (re.findall(('Dogfooding|Dog|fooding'), str(columnData), re.IGNORECASE)):
            #     all_category_df = all_category_df.append({'Dogfooding Queries' : 1}, ignore_index=True)
            #     category_df = category_df.append({'Category' : '10'}, ignore_index=True)
            elif (re.findall(('Attention: New DFSA Submission for.*(?<!\d)$|Mastercard Developers Invitation.*(?<!\d)$|Mastercard Developers Sandbox Enabler.*(?<!\d)$'), str(columnData), re.IGNORECASE)):
                all_category_df = all_category_df.append({'Mastercard Developers Notification' : 1}, ignore_index=True)
                category_df = category_df.append({'Category' : '10'}, ignore_index=True)
            else:
                all_category_df = all_category_df.append({'API Standards queries' : 1}, ignore_index=True)
                category_df = category_df.append({'Category' : '5'}, ignore_index=True)
        return all_category_df, category_df

    
    def updating_subject(self, data):
        for index, subject in enumerate(data['Subject']):
            if (re.findall(('Attention: New DFSA Submission for.*(?<!\d)$|Mastercard Developers Invitation.*(?<!\d)$|Mastercard Developers Sandbox Enabler.*(?<!\d)$'), str(subject), re.IGNORECASE)):
                data['Subject'][index] = subject + " " + str(random())
        return data

# ###### Needed if running independently
# # sheet = pd.read_csv('../ML_testData/test_data/ML_test_October2021.CSV', encoding="ISO-8859-1")
# sheet = pd.read_csv('testing.csv', encoding="ISO-8859-1")

# # sheet.columns = ['Subject', 'Body', 'From: (Name)', 'From: (Address)',
# #         'From: (Type)', 'To: (Name)', 'To: (Address)', 'To: (Type)',
# #         'CC: (Name)', 'CC: (Address)', 'CC: (Type)', 'BCC: (Name)',
# #         'BCC: (Address)', 'BCC: (Type)', 'Billing Information', 'Categories',
# #         'Importance', 'Mileage', 'Sensitivity']
# sheet['Subject'] = sheet['Subject'].str.lower()
# # content_parsed = sheet.iloc[:, :2]
# # content_parsed.columns = ["Subject", "Body"]


# obj = Scanner()
# data = obj.updating_subject(sheet)
# subject = data['Subject']

# df_distinct = data.drop_duplicates(sheet.columns[0]).reset_index(drop=True)
# # sheet = sheet.rename(columns={: "Subject"})


# all_category_subject_df, category_subject_df = obj.content_scan_subject(df_distinct['Subject'])
# total_count_subject = all_category_subject_df.apply(pd.value_counts)





