

import csv
import matplotlib.pyplot as plt
import numpy as np
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
		plt.savefig(figure_name)
		plt.close()
	if(data_dim == 1):
		plt.plot(data1,'ro')
		plt.xlabel('Time (s)')
		plt.ylabel('#slots')
		plt.grid()
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
		plt.savefig(figure_name)
		plt.close()
	if(data_dim == 1):
		plt.plot(data1)
		plt.xlabel('Time (s)')			
		plt.ylabel('Overselection Rate')
		plt.grid()
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
	plt.savefig(figure_name)
	plt.close()	
		
def column(matrix, i):
    return [row[i] for row in matrix]

# ----------------------------------------------------------------------
# outfile load and plot
dim = 2;
alpha = 0.9
recv1_2D = [];
recv2_2D = [];
recv3_2D = [];
for i in range(0,dim):

	prefix = 'outfile0_simple_120_0.1_tstNum_2000000_b1000_s'
	count_str='_c'
	alpha_str= '_a'
	gamma_str='_r'
	ebuse_str='_e'
	postfix='.csv'
	file_name = prefix+str(0)+count_str+str(i)+alpha_str+str(alpha)+gamma_str+str(0.8)+ebuse_str+str(0.2)+postfix
	recv1, recv2 = read_csv_data(file_name, 23, 25)
	file_name = prefix+str(1)+count_str+str(i)+alpha_str+str(alpha)+gamma_str+str(0.8)+ebuse_str+str(0.2)+postfix
	recv3, recv4 = read_csv_data(file_name, 23, 25)

	figure_name = 'figure/'+prefix +str(0)+count_str+str(i)+'.jpg'
	#plot_data(figure_name, recv1, recv2, recv3, 3)
	
	print i
	
	recv1_2D.append(recv1);
	recv2_2D.append(recv2);
	recv3_2D.append(recv3);
	
# ----------------------------------------------------------------------	
# compute average and stdev and plot error bar
recv1_avg = []
recv1_std = []
recv2_avg = []
recv2_std = []
recv3_avg = []
recv3_std = []
time_data = []

for row_i in range(0,len(recv1_2D[0])-1):
	col_value = column(recv1_2D, row_i);
	recv1_avg.append(np.mean(col_value))
	recv1_std.append(np.std(col_value))
	
	col_value = column(recv2_2D, row_i);
	recv2_avg.append(np.mean(col_value))
	recv2_std.append(np.std(col_value))
	
	col_value = column(recv3_2D, row_i);
	recv3_avg.append(np.mean(col_value))
	recv3_std.append(np.std(col_value))
	time_data.append(row_i)
	#print np.std(col_value)
	#print column(recv1_2D, row_i)

figure_name = 'figure/std_'+prefix +str(0)+count_str+str(i)+alpha_str+str(alpha)+gamma_str+str(0.8)+ebuse_str+str(0.2)+'.jpg'
plot_error(figure_name, time_data[0::100], recv1_avg[0::100], recv2, recv3, recv1_std[0::100], 1, 1)

figure_name = 'figure/std_'+prefix +str(1)+count_str+str(i)+alpha_str+str(alpha)+gamma_str+str(0.8)+ebuse_str+str(0.2)+'.jpg'
plot_error(figure_name, time_data[0::100], recv2_avg[0::100], recv2, recv3, recv2_std[0::100], 1, 1)


figure_name = 'figure/std_'+prefix +str(2)+count_str+str(i)+alpha_str+str(alpha)+gamma_str+str(0.8)+ebuse_str+str(0.2)+'.jpg'
plot_error(figure_name, time_data[0::100], recv3_avg[0::100], recv2, recv3, recv3_std[0::100], 1, 1)


# ----------------------------------------------------------------------
# resource file load and plot
recv1_2D = [];
recv2_2D = [];
recv3_2D = [];
recv4_2D = [];
recv5_2D = [];
recv6_2D = [];

for i in range(0,dim):

	file_name = file_name = prefix+str(0)+count_str+str(i)+alpha_str+str(alpha)+gamma_str+str(0.8)+ebuse_str+str(0.2)+postfix
	recv1, recv2 = read_csv_data(file_name, 0,1)
	file_name = file_name = prefix+str(1)+count_str+str(i)+alpha_str+str(alpha)+gamma_str+str(0.8)+ebuse_str+str(0.2)+postfix
	recv3, recv4 = read_csv_data(file_name, 0,1)
	file_name = file_name = prefix+str(2)+count_str+str(i)+alpha_str+str(alpha)+gamma_str+str(0.8)+ebuse_str+str(0.2)+postfix
	recv5, recv6 = read_csv_data(file_name, 0,1)

	figure_name = 'figure1/'+prefix +str(0)+count_str+str(i)+str(1)+'.jpg'
	#plot_data(figure_name, recv1, recv2, recv3, 1)
	
	figure_name = 'figure1/'+prefix +str(0)+count_str+str(i)+str(1)+'.jpg'
	#plot_data(figure_name, recv2, recv2, recv3, 1)
	
	figure_name = 'figure1/'+prefix +str(0)+count_str+str(i)+str(1)+'.jpg'
	#plot_data(figure_name, recv3, recv2, recv3, 1)
	
	figure_name = 'figure1/'+prefix +str(0)+count_str+str(i)+str(1)+'.jpg'
	#plot_data(figure_name, recv4, recv2, recv3, 1)
	
	figure_name = 'figure1/'+prefix +str(0)+count_str+str(i)+str(1)+'.jpg'
	#plot_data(figure_name, recv5, recv2, recv3, 1)
	
	figure_name = 'figure1/'+prefix +str(0)+count_str+str(i)+str(1)+'.jpg'
	#plot_data(figure_name, recv6, recv2, recv3, 1)
	
	recv1_2D.append(recv1);
	recv2_2D.append(recv2);
	recv3_2D.append(recv3);
	recv4_2D.append(recv4);
	recv5_2D.append(recv5);
	recv6_2D.append(recv6);

# ----------------------------------------------------------------------
# compute average and stdev and plot errobar
recv1_avg = []
recv1_std = []
recv2_avg = []
recv2_std = []
recv3_avg = []
recv3_std = []
recv4_avg = []
recv4_std = []
recv5_avg = []
recv5_std = []
recv6_avg = []
recv6_std = []

recv1_avg_t = []
recv1_std_t = []
recv2_avg_t = []
recv2_std_t = []
recv3_avg_t = []
recv3_std_t = []

time_data = []

for row_i in range(0,len(recv1_2D[0])-1):
	col_value1 = column(recv1_2D, row_i);
	recv1_avg.append(np.mean(col_value1))
	recv1_std.append(np.std(col_value1))
	
	col_value2 = column(recv2_2D, row_i)
	recv2_avg.append(np.mean(col_value2))
	recv2_std.append(np.std(col_value2))
	
	col_value3 = column(recv3_2D, row_i)
	recv3_avg.append(np.mean(col_value3))
	recv3_std.append(np.std(col_value3))
	
	col_value4 = column(recv4_2D, row_i)
	recv4_avg.append(np.mean(col_value4))
	recv4_std.append(np.std(col_value4))
	
	col_value5 = column(recv5_2D, row_i)
	recv5_avg.append(np.mean(col_value5))
	recv5_std.append(np.std(col_value5))
	
	col_value6 = column(recv6_2D, row_i)
	recv6_avg.append(np.mean(col_value6))
	recv6_std.append(np.std(col_value6))
	
	col_1 = [x + y for x, y in zip(col_value1, col_value4)]
	col_2 = [x + y for x, y in zip(col_value2, col_value5)]
	col_3 = [x + y for x, y in zip(col_value3, col_value6)]
	
	recv1_avg_t.append(np.mean(col_1))
	recv1_std_t.append(np.std(col_1))
	
	recv2_avg_t.append(np.mean(col_2))
	recv2_std_t.append(np.std(col_2))
	
	recv3_avg_t.append(np.mean(col_3))
	recv3_std_t.append(np.std(col_3))
	
	time_data.append(5*row_i)
	#print np.std(col_value)
	#print column(recv1_2D, row_i)

inv = 10;
figure_name = 'figure/std_res_'+prefix +str(0)+count_str+str(i)+alpha_str+str(alpha)+gamma_str+str(0.8)+ebuse_str+str(0.2)+'.jpg'
plot_error(figure_name, time_data[0::inv], recv1_avg[0::inv], recv2, recv3, recv1_std[0::inv], 1, 2)

figure_name = 'figure/std_res_'+prefix +str(1)+count_str+str(i)+alpha_str+str(alpha)+gamma_str+str(0.8)+ebuse_str+str(0.2)+'.jpg'
plot_error(figure_name, time_data[0::inv], recv2_avg[0::inv], recv2, recv3, recv2_std[0::inv], 1, 2)

figure_name = 'figure/std_res_'+prefix +str(2)+count_str+str(i)+alpha_str+str(alpha)+gamma_str+str(0.8)+ebuse_str+str(0.2)+'.jpg'
plot_error(figure_name, time_data[0::inv], recv3_avg[0::inv], recv2, recv3, recv3_std[0::inv], 1, 2)

figure_name = 'figure/std_res_'+prefix +str(3)+count_str+str(i)+alpha_str+str(alpha)+gamma_str+str(0.8)+ebuse_str+str(0.2)+'.jpg'
plot_error(figure_name, time_data[0::inv], recv4_avg[0::inv], recv2, recv3, recv4_std[0::inv], 1, 2)

figure_name = 'figure/std_res_'+prefix +str(4)+count_str+str(i)+alpha_str+str(alpha)+gamma_str+str(0.8)+ebuse_str+str(0.2)+'.jpg'
plot_error(figure_name, time_data[0::inv], recv5_avg[0::inv], recv2, recv3, recv5_std[0::inv], 1, 2)

figure_name = 'figure/std_res_'+prefix +str(5)+count_str+str(i)+alpha_str+str(alpha)+gamma_str+str(0.8)+ebuse_str+str(0.2)+'.jpg'
plot_error(figure_name, time_data[0::inv], recv6_avg[0::inv], recv2, recv3, recv6_std[0::inv], 1, 2)

figure_name = 'figure/std_res_t_'+prefix +str(0)+count_str+str(i)+alpha_str+str(alpha)+gamma_str+str(0.8)+ebuse_str+str(0.2)+'.jpg'
plot_error(figure_name, time_data[0::inv], recv1_avg_t[0::inv], recv2, recv3, recv1_std_t[0::inv], 1, 2)

figure_name = 'figure/std_res_t_'+prefix +str(1)+count_str+str(i)+alpha_str+str(alpha)+gamma_str+str(0.8)+ebuse_str+str(0.2)+'.jpg'
plot_error(figure_name, time_data[0::inv], recv2_avg_t[0::inv], recv2, recv3, recv2_std_t[0::inv], 1, 2)

figure_name = 'figure/std_res_t_'+prefix +str(2)+count_str+str(i)+alpha_str+str(alpha)+gamma_str+str(0.8)+ebuse_str+str(0.2)+'.jpg'
plot_error(figure_name, time_data[0::inv], recv3_avg_t[0::inv], recv2, recv3, recv3_std_t[0::inv], 1, 2)



