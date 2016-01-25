

import csv
import matplotlib.pyplot as plt
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
	return out_array1, out_array2
	
def plot_data(figure_name, data1, data2, data3, data_dim):
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
		plt.plot(data1,'ro')
		plt.xlabel('Round')
		plt.ylabel('#slots')
		plt.grid()
		plt.savefig(figure_name)
		plt.close()


"""for i in range(0,10):

	prefix = 'outfile0_simple_120_0.1_tstNum_2000000_b1000_s'
	count_str='_c'
	postfix='.csv'
	file_name = prefix+str(0)+count_str+str(i)+postfix
	recv1, recv2 = read_csv_data(file_name, 23, 25)
	file_name = prefix+str(1)+count_str+str(i)+postfix
	recv3, recv4 = read_csv_data(file_name, 23, 25)

	figure_name = 'figure/'+prefix +str(0)+count_str+str(i)+'.svg'
	plot_data(figure_name, recv1, recv2, recv3, 3)"""

# compute average and stdev and plot error bar


for i in range(0,10):
	prefix = 'resouce_assign_120_0.1_tstNum_2000000_b1000_s'
	count_str='_c'
	postfix='.csv'
	file_name = prefix+str(0)+count_str+str(i)+postfix
	recv1, recv2 = read_csv_data(file_name, 0,1)
	file_name = prefix+str(1)+count_str+str(i)+postfix
	recv3, recv4 = read_csv_data(file_name, 0,1)

	figure_name = 'figure1/'+prefix +str(0)+count_str+str(i)+str(1)+'.svg'
	plot_data(figure_name, recv1, recv2, recv3, 1)
	
	figure_name = 'figure1/'+prefix +str(0)+count_str+str(i)+str(1)+'.svg'
	plot_data(figure_name, recv1, recv2, recv3, 1)
	
	figure_name = 'figure1/'+prefix +str(0)+count_str+str(i)+str(1)+'.svg'
	plot_data(figure_name, recv1, recv2, recv3, 1)

# compute average and stdev and plot errobar
