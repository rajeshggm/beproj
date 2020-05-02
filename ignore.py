#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 15:37:52 2019

@author: nickchen
"""

import json
import argparse
from os.path import isfile, join
import re
import numpy as np
import pprint
import pickle

def saving_data(data_dir = 'Data'):
    
    #Setting data file names
	train_ques_json = join(data_dir, 'OpenEnded_abstract_v002_train2015_questions.json')
	train_anno_json = join(data_dir, 'abstract_v002_train2015_annotations.json')

	val_ques_json = join(data_dir, 'OpenEnded_abstract_v002_val2015_questions.json')
	val_anno_json = join(data_dir, 'abstract_v002_val2015_annotations.json')
    
	qa_data_file = join(data_dir, 'qa_data_file.pkl')
	vocab_file = join(data_dir, 'vocab_file.pkl')
    
    #If data are already extracted
	if isfile(qa_data_file):
		return

    #loading json files
	print("Loading Training questions")
	with open(train_ques_json) as f:
		train_ques = json.loads(f.read())
	
	print("Loading Training annotations")
	with open(train_anno_json) as f:
		train_anno = json.loads(f.read())

	print("Loading Val questions")
	with open(val_ques_json) as f:
		val_ques = json.loads(f.read())
	
	print("Loading Val annotations")
	with open(val_anno_json) as f:
		val_anno = json.loads(f.read())
        
    #
        
