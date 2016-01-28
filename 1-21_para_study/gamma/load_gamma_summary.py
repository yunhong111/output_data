import csv
import matplotlib.pyplot as plt
import numpy as np
import math 

def plot_error(figure_name, x_data1, data1, data2, data3, y_error, data_dim, data_type):
	fig = plt.gcf()
	fig.set_size_inches(6, 4.5)
	plt.plot(x_data1, data1, linestyle="dashed", marker="o", color="green")
	plt.errorbar(x_data1, data1,yerr=y_error, linestyle="None", marker="None", color="green")
	#plt.plot(data2)
	#plt.plot(data3)
	if(data_type == 1):
		plt.xlabel('Time (s)')
		plt.ylabel('Average setling time')
		plt.xlim(0.0,1.0)
	if(data_type == 2):
		plt.xlabel('Alpha (s)')
		plt.ylabel('#slots')
	plt.grid()
	plt.savefig(figure_name)
	plt.close()
	
# load the summary file
file_name = 'gamma_summary.csv'
from collections import defaultdict

columns = defaultdict(list) # each value in each column is appended to a list

with open(file_name) as f:
    reader = csv.DictReader(f) # read rows into a dictionary format
    for row in reader: # read a row as {column1: value1, column2: value2,...}
        for (k,v) in row.items(): # go over each column name and value 
            columns[k].append(float(v)) # append the value into the appropriate list
                                 # based on column name k

print(columns['Recv1_avg_time'])

recv2=[]
recv3=[]
figure_name = 'figure/std_alpha_recv1.jpg'
plot_error(figure_name, columns['gamma'], columns['Recv1_avg_time'], recv2, recv3, columns['Recv1_std_time'], 1, 1)

figure_name = 'figure/std_alpha_recv2.jpg'
plot_error(figure_name, columns['gamma'], columns['Recv2_avg_time'], recv2, recv3, columns['Recv2_std_time'], 1, 1)

figure_name = 'figure/std_alpha_recv3.jpg'
plot_error(figure_name, columns['gamma'], columns['Recv3_avg_time'], recv2, recv3, columns['Recv3_std_time'], 1, 1)
