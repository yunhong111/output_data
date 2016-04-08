

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
	#out_array1 = map(float, out_array1)
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
def false_pos_neg(col1, col2, is_pos):
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
	
	if (is_pos == 0):
		rate = float(diff_num)/float(len(col1))
	
	# precision
	if(is_pos == 2):
		rate = float(same_num)/float(len(col2))
	#print diff_num + same_num, len(col2)
	return rate

def differ_data(col1, col2, num):
	same_num = 0
	diff_num = 0
	for i in range(0,num):
		line2 = col2[i]
		equal_flag = 0
		for line1 in col1:
			if(line2 == line1):
				same_num = same_num +1
				equal_flag = 1
				break
		if(equal_flag == 0):
			diff_num = diff_num + 1
			
	return diff_num
	
def error_rate(col1, threshold_error):
	
	diff_num = 0
	for line1 in col1:
		if(line1 >= threshold_error):
			diff_num = diff_num + 1
			
	rate = float(diff_num)/float(len(col1))
	return rate

# ----------------------------------------------------------------------
# outfile load and plot
dim = 12;
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

# 0.000025 0.00005 0.000075 0.0001 0.00025 0.0005 0.00075 0.001
# 800000	1600000	2400000	3200000	8000000	16000000 24000000 32000000
#threshold_rate_vector = [0.000025, 0.00005, 0.000075, 0.0001, 0.00025, 0.0005, 0.00075, 0.001]
#threshold_rate=(3200000 6400000 9600000 12800000 16000000 19200000 22400000 25600000 28800000 32000000)

threshold_rate_vector=(6250,	12500,	18750,	25000,	31250,	37500,	43750,	50000,	56250,	62500)

#threshold_rate_vector = [3200000, 6400000, 9600000, 12800000, 16000000, 19200000, 22400000, 25600000, 28800000, 32000000]

for ti in range(0,10):
	threshold_rate=threshold_rate_vector[ti]
	threshold_rate_str = str(threshold_rate)#"0.0005"

	summary_file_name = "summary_average_diff_pkts.csv"
		
	addToFile(summary_file_name, threshold_rate_str+"\n")

	for i in range(0,dim):
		prefix1 = 'true_/outfile0_threshold_'
		prefix2 = 'table/outfile0_threshold_'
		prefix3 = 'filter/outfile0_threshold_'
		prefix4 = 'noise_filter/outfile0_threshold_'
		
		infile_name='_mapped-caida-' + str(i*5+1) +'_out'
		infile_name_true='_mapped-caida-' + str(i*5+1) +'_out_true'  
		
		counter_num1 = 2500
		counter_num2=2500
		
		item_type=1

		postfix='.csv'
		
		file_name = prefix1+threshold_rate_str+'_type_'+str(item_type)+infile_name_true+'_table'+postfix
		ip_true, ip_true_freq = read_csv_data(file_name, 1, 5)
		ip_true_freq, ip_true = (list(x) for x in zip(*sorted(zip(ip_true_freq, ip_true))))
		
		file_name = prefix2+threshold_rate_str+'_type_'+str(item_type)+infile_name_true+'_table'+postfix
		ip_table, ip_table_freq = read_csv_data(file_name, 1, 3)
		ip_table_freq, ip_table = (list(x) for x in zip(*sorted(zip(ip_table_freq, ip_table))))
		
		file_name = prefix3+threshold_rate_str+'_type_'+str(item_type)+infile_name+'_table'+postfix
		ip_filter, ip_filter_freq = read_csv_data(file_name, 1, 3)
		ip_filter_freq, ip_filter = (list(x) for x in zip(*sorted(zip(ip_filter_freq, ip_filter))))
		
		file_name = prefix4+threshold_rate_str+'_type_'+str(item_type)+infile_name+'_table'+postfix
		ip_noise_filter, ip_noise_filter_freq = read_csv_data(file_name, 1, 3)
		ip_noise_filter_freq, ip_noise_filter = (list(x) for x in zip(*sorted(zip(ip_noise_filter_freq, ip_noise_filter))))
		
		false_pos_table = false_pos_neg(ip_true, ip_table,1)
		false_neg_table = false_pos_neg(ip_table, ip_true,0)
		precision_table = false_pos_neg(ip_true, ip_table,2)
		diff_num_table = differ_data(ip_true, ip_table,len(ip_true))
		
		false_pos_filter = false_pos_neg(ip_true, ip_filter,1)
		false_neg_filter = false_pos_neg(ip_filter, ip_true,0)
		precision_filter = false_pos_neg(ip_true, ip_filter,2)
		diff_num_filter = differ_data(ip_true, ip_filter,len(ip_true))

		false_pos_noise_filter = false_pos_neg(ip_true, ip_noise_filter,1)
		false_neg_noise_filter = false_pos_neg(ip_noise_filter, ip_true,0)
		precision_noise_filter = false_pos_neg(ip_true, ip_noise_filter,2)
		diff_num_noise = differ_data(ip_true, ip_noise_filter,len(ip_true))
		
		print i, false_pos_table, false_neg_table, false_pos_filter, false_neg_filter, precision_table, precision_filter

		addToFile(summary_file_name, str(i)+",")
		addToFile(summary_file_name, str(false_pos_table)+",")
		addToFile(summary_file_name, str(false_neg_table)+",")
		addToFile(summary_file_name, str(precision_table)+",")
		
		addToFile(summary_file_name, str(false_pos_filter)+",")
		addToFile(summary_file_name, str(false_neg_filter)+",")
		addToFile(summary_file_name, str(precision_filter)+",")
		
		addToFile(summary_file_name, str(false_pos_noise_filter)+",")
		addToFile(summary_file_name, str(false_neg_noise_filter)+",")
		addToFile(summary_file_name, str(precision_noise_filter)+",")

		addToFile(summary_file_name, str(diff_num_table)+",")
		addToFile(summary_file_name, str(diff_num_filter)+",")
		addToFile(summary_file_name, str(diff_num_noise)+",")
		
		addToFile(summary_file_name, str(len(ip_true))+",")
		addToFile(summary_file_name, str(len(ip_filter))+",")
		addToFile(summary_file_name, str(len(ip_noise_filter))+"\n")
	
	

# compute error

"""for i in range(0,dim):
	
	prefix1 = 'table_CM_error/outfile0_threshold_'
	
	infile_name_true='_mapped-caida-' + str(i*5+1) +'_out_true'  
	
	counter_num1 = 2500
	
	item_type=0

	postfix='.csv'
	
	file_name = prefix1+threshold_rate_str+'_type_'+str(item_type)+infile_name_true+'_table'+postfix
	ip_true, ip_true_freq = read_csv_data(file_name, 1,7)
	
	threshold_error = 3e+8
	false_rate = error_rate(ip_true_freq, threshold_error)
	print false_rate"""








