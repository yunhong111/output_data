import csv
import matplotlib.pyplot as plt
import numpy as np
import math 

def plot_error(figure_name, x_data1, data1, data2, data3, y_error, data_dim, data_type):
	#fig = plt.gcf()
	fig, ax = plt.subplots()
	fig.set_size_inches(6, 4.5)
	plt.plot(x_data1, data1, linestyle="dashed", marker="o", color="green")
	plt.errorbar(x_data1, data1,yerr=y_error, linestyle="None", marker="None", color="green")
	#plt.plot(data2)
	#plt.plot(data3)
	if(data_type == 1):
		plt.xlabel('ebuse')
		plt.ylabel('Average settling time (s)')
		
	if(data_type == 2):
		plt.xlabel('recv')
		plt.ylabel('Overselection Rate')
	plt.grid()
	plt.xlim(0.5, 3.5)
	ax.set_xticks((1,2,3))
	plt.ylim(0.00, 0.03)
	plt.tight_layout()
	plt.savefig(figure_name)
	plt.close()
	
# load the summary file
setling_time_range = 0.05
para_str = 'alpha'

file_name = para_str+'_summary'+str(setling_time_range)+'.csv'
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
figure_name = 'figure/std_alpha_recv1'+str(setling_time_range)+'.jpg'
plot_error(figure_name, columns[para_str], columns['Recv1_avg_time'], recv2, recv3, columns['Recv1_std_time'], 1, 1)

figure_name = 'figure/std_alpha_recv2'+str(setling_time_range)+'.jpg'
plot_error(figure_name, columns[para_str], columns['Recv2_avg_time'], recv2, recv3, columns['Recv2_std_time'], 1, 1)

figure_name = 'figure/std_alpha_recv3'+str(setling_time_range)+'.jpg'
plot_error(figure_name, columns[para_str], columns['Recv3_avg_time'], recv2, recv3, columns['Recv3_std_time'], 1, 1)

figure_name = 'figure/std_alpha_total'+str(setling_time_range)+'.jpg'
plot_error(figure_name, columns[para_str], columns['settling_time'], recv2, recv3, columns['settling_time_std'], 1, 1)

figure_name = 'figure/std_alpha_total_stable_mean'+str(setling_time_range)+'.jpg'
plot_error(figure_name, columns[para_str], columns['stable_value'], recv2, recv3, columns['stable_value_std'], 1, 2)

figure_name = 'figure/std_alpha_total_stable_mean_recvs'+str(setling_time_range)+'.jpg'

mean_stable_value_list = []
for line in columns['recv1_mean_stable_value']:
	mean_stable_value_list.append(line);
for line in columns['recv2_mean_stable_value']:
	mean_stable_value_list.append(line);
for line in columns['recv3_mean_stable_value']:
	mean_stable_value_list.append(line);

std_stable_value_list = []
for line in columns['recv1_std_stable_value']:
	std_stable_value_list.append(line);
for line in columns['recv2_std_stable_value']:
	std_stable_value_list.append(line);
for line in columns['recv3_std_stable_value']:
	std_stable_value_list.append(line);

x_data = [1, 2, 3]
print mean_stable_value_list
print std_stable_value_list

plot_error(figure_name, x_data, mean_stable_value_list, recv2, recv3,std_stable_value_list, 1, 2)


