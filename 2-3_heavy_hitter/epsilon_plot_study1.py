

import csv
import matplotlib.pyplot as plt
import numpy as np
import math 
#import statistics as st
# open and read the csv file into memory

def read_csv_data(file_name, line1, line2):
    # open and read the csv file into memory
	file1 = open(file_name)
	reader1 = csv.reader(file1)
	# iterate through the lines and print them to stdout
	# the csv module returns us a list of lists and we
	# simply iterate through it
	out_array1 = [];
	out_array2 = [];
	for line in reader1:
		out_array1.append(line[line1])
		out_array2.append(line[line2])
	out_array1 = map(float, out_array1)
	out_array2 = map(float, out_array2)
	return out_array1[10:], out_array2[10:]
	
def plot_data(figure_name, data1, data2, data3, data_dim):
	fig = plt.gcf()
	fig.set_size_inches(6, 4.5)
	
	if(data_dim == 3):
		plt.plot(data1)
		plt.plot(data2)
		plt.plot(data3)
		plt.xlabel('Time (s)')			
		plt.ylabel('Overselection Rate')
		plt.ylim(0.0,0.03)
		plt.grid()
		plt.tight_layout()
		plt.savefig(figure_name)
		plt.close()
	if(data_dim == 1):
		plt.plot(data1,'ro')
		plt.xlabel('Time (s)')
		plt.ylabel('#slots')
		plt.grid()
		plt.tight_layout()
		plt.savefig(figure_name)
		plt.close()
		
def plot_line(figure_name, data1, data2, data3, data_dim):
	fig = plt.gcf()
	fig.set_size_inches(6, 4.5)
	
	if(data_dim == 3):
		plt.plot(data1)
		plt.plot(data2)
		plt.plot(data3)
		plt.xlabel('Time (s)')			
		plt.ylabel('Overselection Rate')
		plt.grid()
		plt.tight_layout()
		plt.savefig(figure_name)
		plt.close()
	if(data_dim == 1):
		plt.plot(data1)
		plt.xlabel('Time (s)')			
		plt.ylabel('Overselection Rate')
		plt.grid()
		plt.tight_layout()
		plt.savefig(figure_name)
		plt.close()
		
def plot_error(figure_name, x_data1, data1, data2, data3, y_error, data_dim, data_type):
	fig = plt.gcf()
	fig.set_size_inches(6, 4.5)
	
	plt.plot(x_data1, data1, linestyle="dashed", marker="o", color="green")
	plt.errorbar(x_data1, data1,yerr=y_error, linestyle="None", marker="None", color="green")
	#plt.plot(data2)
	#plt.plot(data3)
	if(data_type == 1):
		plt.xlabel('Time (s)')
		plt.ylabel('Overselection Rate')
		plt.ylim(0.0,0.03)
	if(data_type == 2):
		plt.xlabel('Time (s)')
		plt.ylabel('#slots')
	plt.grid()
	plt.tight_layout()
	plt.savefig(figure_name)
	plt.close()	
		
def column(matrix, i):
    return [row[i] for row in matrix]

# compute setling time
def setling_time(col_data, error_band, target):
	col_len = len(col_data)
	error_value = error_band*target
	set_time = 10;
	for i in range(0, col_len):
		if((col_data[i]-target)>error_value):
			set_time = i
	return set_time
	
def addToFile(file_name, what):
	#with open (file_name,'a') as f:
    f = open(file_name, 'a').write(what)
"""	wtr = csv.writer(f)
	wtr.writerow(what)
	wtr.writerow(what) """
# col 1 is base
def false_pos_neg(col1, col2):
	same_num = 0
	diff_num = 0
	for line2 in col2:
		equal_flag = 0
		for line1 in col1:
			if(line2 == line1):
				same_num = same_num +1
				equal_flag = 1
				break
		if(equal_flag == 0):
			diff_num = diff_num + 1
			
	rate = float(diff_num)/float(diff_num + same_num)
	return rate

# ----------------------------------------------------------------------
# outfile load and plot
dim = 11;
alpha = 0.1
gamma = 0.8
ebuse = 0.5

recv1_2D = [];
recv2_2D = [];
recv3_2D = [];

recv1_set_time = [];
recv2_set_time = [];
recv3_set_time = [];

setling_time_range = 0.1

para_str = 'ebuse'

for i in range(0,dim):
	prefix1 = 'standard/outfile0_threshold_'
	prefix2 = 'table_2500/outfile0_threshold_'
	prefix3 = 'filter_2500/outfile0_threshold_'
	
	infile_name='_mapped-caida-' + str(i*5+1) +'_out_cnt_'
	infile_name_true='_mapped-caida-' + str(i*5+1) +'_out_true_cnt_'  
	
	counter_num1 = 300000
	counter_num2=2500

	threshold_rate=0.0025
	item_type=0

	postfix='.csv'
	
	file_name = prefix1+str(threshold_rate)+'_type_'+str(item_type)+infile_name_true+str(counter_num1)+'_table'+postfix
	ip_true, ip_true_freq = read_csv_data(file_name, 1, 3)
	file_name = prefix2+str(threshold_rate)+'_type_'+str(item_type)+infile_name_true+str(counter_num2)+'_table'+postfix
	ip_table, ip_table_freq = read_csv_data(file_name, 1, 3)
	file_name = prefix3+str(threshold_rate)+'_type_'+str(item_type)+infile_name+str(counter_num2)+'_table'+postfix
	ip_filter, ip_filter_freq = read_csv_data(file_name, 1, 3)
	
	false_pos_table = false_pos_neg(ip_true, ip_table)
	false_neg_table = false_pos_neg(ip_table, ip_true)
	
	false_pos_filter = false_pos_neg(ip_true, ip_filter)
	false_neg_filter = false_pos_neg(ip_filter, ip_true)
	
	print i, false_pos_table, false_neg_table, false_pos_filter, false_neg_filter
	
	

	







