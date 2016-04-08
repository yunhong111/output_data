

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
		plt.savefig(figure_name, dpi = 300)
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
		plt.savefig(figure_name, dpi = 300)
		plt.close()
	if(data_dim == 1):
		plt.plot(data1)
		plt.xlabel('Time (s)')			
		plt.ylabel('Overselection Rate')
		plt.grid()
		plt.tight_layout()
		plt.savefig(figure_name, dpi = 300)
		plt.close()
		
def plot_error(figure_name, x_data1, data1, data2, data3, y_error, data_dim, data_type, ylim_max):
	fig = plt.gcf()
	fig.set_size_inches(6, 4.5)
	fig.autofmt_xdate()
	
	textsize = 9
	#plt.plot(x_data1, data1, linestyle="dashed", marker="o", markersize=5, color='blue', linewidth=0.5)
	
	plt.errorbar(x_data1, data1,yerr=y_error, linestyle="None", marker="None", color="red")
	plt.plot(x_data1, data1, marker="o", markersize=2, color='blue', linewidth=1.5)
	
	#plt.plot(data2)
	#plt.plot(data3)
	if(data_type == 1):
		plt.xlabel('Time (s)')
		plt.ylabel('Overselection Rate')
	if(data_type == 2):
		plt.xlabel('Time (s)')
		plt.ylabel('#slots')
	if(data_type == 3):
		plt.xlabel('Time (s)')
		plt.ylabel('Reward')
	plt.ylim(0.0,ylim_max)	
	plt.grid()
	plt.tight_layout()
	plt.savefig(figure_name, dpi = 300)
	plt.close()

# subplot
def plot_error_sub(plt, subplot_seq, figure_name, x_data1, data1, data2, data3, y_error, data_dim, data_type, ylim_max):
	
	plt.subplot(subplot_seq)
	
	textsize = 9
	plt.errorbar(x_data1, data1,yerr=y_error, linestyle="None", marker="None", color="red")
	plt.plot(x_data1, data1, marker="o", markersize=2,  color='blue', linewidth=1.5)
	
	
	#plt.plot(data2)
	#plt.plot(data3)
	if(data_type == 1):
		plt.xlabel('Time (s)')
		plt.ylabel('Overselection Rate')
	if(data_type == 2):
		plt.xlabel('Time (s)')
		plt.ylabel('#slots')	
	if(data_type == 3):
		plt.xlabel('Time (s)')
		plt.ylabel('Reward')
	plt.ylim(0.0,ylim_max)
	plt.grid()
	plt.tight_layout()

		
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

def confidence(col_data, n):
	z = 1.96
	for i in range(0, len(col_data)):
		col_data[i] = z*col_data[i]/math.sqrt(n)
	
def load_reward(prefix, start, inv, res_row_size, f_name, rewad_ylim_max, rewad_ylim_max_total,  data_type):
	# resource file load and plot
	recv1_2D = [];
	recv2_2D = [];
	recv3_2D = [];
	recv4_2D = [];

	recv_total_2D = []


	for i in range(0,dim):
		#prefix = 'outfile0_simple_360_0.1_tstNum_2000000_b1000'
		count_str='_c'
		alpha_str= '_a'
		gamma_str='_g'
		ebuse_str='_e'
		rep_str = '_r'
		postfix='.csv'
		if(data_type == 2):
			gamma_str='_e'
			ebuse_str='_r'
			
		file_name = prefix+alpha_str+str(alpha)+gamma_str+str(gamma)+ebuse_str+str(ebuse)+rep_str+str(i)+postfix 
		
		if(data_type == 2):
			recv1, recv2 = read_csv_data(file_name, start,start+1)
			recv3, recv4 = read_csv_data(file_name, start+2,start+3)
		else:
			recv1, recv2 = read_csv_data(file_name, start,start+2)
			recv3, recv4 = read_csv_data(file_name, start+4,start+6)
		#print recv1
		
		zipped_list = zip(recv1, recv2, recv3, recv4)
		recv_total = [sum(item) for item in zipped_list]


		figure_name = 'figure1/'+prefix +str(0)+count_str+str(i)+str(1)+'.jpg'
		#plot_data(figure_name, recv1, recv2, recv3, 1)
		
		figure_name = 'figure1/'+prefix +str(0)+count_str+str(i)+str(1)+'.jpg'
		#plot_data(figure_name, recv2, recv2, recv3, 1)
		
		figure_name = 'figure1/'+prefix +str(0)+count_str+str(i)+str(1)+'.jpg'
		#plot_data(figure_name, recv3, recv2, recv3, 1)
		
		figure_name = 'figure1/'+prefix +str(0)+count_str+str(i)+str(1)+'.jpg'
		#plot_data(figure_name, recv4, recv2, recv3, 1)
		
		
		recv1_2D.append(recv1);
		recv2_2D.append(recv2);
		recv3_2D.append(recv3);
		recv4_2D.append(recv4);
		
		recv_total_2D.append(recv_total);


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

	recv_avg_total = []
	recv_std_total = []

	time_data = []

	#print np.shape(recv1_2D)
	#res_row_size = 3590
	for row_i in range(0,res_row_size):
		
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
		
		col_value_total = column(recv_total_2D, row_i)
		recv_avg_total.append(np.mean(col_value_total))
		recv_std_total.append(np.std(col_value_total))
		
		if(data_type == 2):
			time_data.append(5*row_i)
		else:
			time_data.append(row_i)
		#print np.std(col_value)
		#print column(recv1_2D, row_i)
		
	# ---------------------------------------------------------------------
	# compute confidence
	confidence(recv1_std,dim)
	confidence(recv2_std,dim)
	confidence(recv3_std,dim)
	confidence(recv4_std,dim)
	confidence(recv_std_total,dim)
	
	#rewad_ylim_max = 4;
	figure_name = f_name+prefix +str(0)+count_str+str(i)+'.jpg'
	plot_error(figure_name, time_data[0::inv], recv1_avg[0::inv], recv2, recv3, recv1_std[0::inv], 1, data_type, rewad_ylim_max)

	figure_name = f_name+prefix +str(1)+count_str+str(i)+'.jpg'
	#plot_error(figure_name, time_data[0::inv], recv2_avg[0::inv], recv2, recv3, recv2_std[0::inv], 1, 2)
	plot_error(figure_name, time_data[0::inv], recv2_avg[0::inv], recv2, recv3, recv2_std[0::inv], 1, data_type, rewad_ylim_max)

	figure_name = f_name+prefix +str(2)+count_str+str(i)+'.jpg'
	plot_error(figure_name, time_data[0::inv], recv3_avg[0::inv], recv2, recv3, recv3_std[0::inv], 1, data_type, rewad_ylim_max)

	figure_name = f_name+prefix +str(3)+count_str+str(i)+'.jpg'
	plot_error(figure_name, time_data[0::inv], recv4_avg[0::inv], recv2, recv3, recv4_std[0::inv], 1, data_type, rewad_ylim_max)

	figure_name = f_name+'total_'+prefix +str(3)+count_str+str(i)+'.jpg'
	plot_error(figure_name, time_data[0::inv], recv_avg_total[0::inv], recv2, recv3, recv_std_total[0::inv], 1, data_type, rewad_ylim_max_total)

	# subfigure
	fig = plt.gcf()
	fig.set_size_inches(6.5, 10)
	
	"""ax1 = fig.add_subplot(4, 1, 1)
	ax2 = fig.add_subplot(4, 1, 2, sharex=ax1, sharey=ax1)
	ax3 = fig.add_subplot(4, 1, 3, sharex=ax1, sharey=ax1)
	ax4 = fig.add_subplot(4, 1, 4, sharex=ax1, sharey=ax1)
	ax1.axhline(linewidth=4, color = 'black')
	ax1.axvline(linewidth=4, color = 'black')
	
	ax2.axhline(linewidth=4, color = 'black')
	ax2.axvline(linewidth=4, color = 'black')
	ax3.axhline(linewidth=4, color = 'black')
	ax3.axvline(linewidth=4, color = 'black')
	ax4.axhline(linewidth=4, color = 'black')
	ax4.axvline(linewidth=4, color = 'black')"""

	figure_name = f_name+'.jpg'
	plot_error_sub(plt, 411, figure_name, time_data[0::inv], recv1_avg[0::inv], recv2, recv3, recv1_std[0::inv], 1, data_type, rewad_ylim_max)

	#plot_error(figure_name, time_data[0::inv], recv2_avg[0::inv], recv2, recv3, recv2_std[0::inv], 1, 2)
	plot_error_sub(plt, 412, figure_name, time_data[0::inv], recv2_avg[0::inv], recv2, recv3, recv2_std[0::inv], 1, data_type, rewad_ylim_max)

	plot_error_sub(plt, 413, figure_name, time_data[0::inv], recv3_avg[0::inv], recv2, recv3, recv3_std[0::inv], 1, data_type, rewad_ylim_max)

	plot_error_sub(plt, 414, figure_name, time_data[0::inv], recv4_avg[0::inv], recv2, recv3, recv4_std[0::inv], 1, data_type, rewad_ylim_max)

	plt.savefig(figure_name, dpi = 300)
	plt.close()


# ----------------------------------------------------------------------
# outfile load and plot
dim = 32;
alpha = 0.2
gamma = 0.95
ebuse = 0.01

recv1_2D = [];
recv2_2D = [];
recv3_2D = [];
recv4_2D = [];

recv1_set_time = [];
recv2_set_time = [];
recv3_set_time = [];
recv4_set_time = [];

setling_time_range = 0.1

para_str = 'ebuse'

for i in range(0,dim):

	prefix = 'outfile0_simple_360_0.1_tstNum_2000000_b1000'
	count_str='_c'
	alpha_str= '_a'
	gamma_str='_g'
	ebuse_str='_e'
	rep_str = '_r'
	postfix='.csv'
	file_name = prefix+alpha_str+str(alpha)+gamma_str+str(gamma)+ebuse_str+str(ebuse)+rep_str+str(i)+postfix #+str(0)+count_str+str(i)+alpha_str+str(alpha)+gamma_str+str(gamma)+ebuse_str+str(ebuse)+postfix
	recv1, recv2 = read_csv_data(file_name, 27, 29)
	recv3, recv4 = read_csv_data(file_name, 31, 33)

	figure_name = 'figure/'+prefix +str(0)+count_str+str(i)+'.jpg'
	#plot_data(figure_name, recv1, recv2, recv3, 3)
	
	print i
	
	recv1_2D.append(recv1);
	recv2_2D.append(recv2);
	recv3_2D.append(recv3);
	recv4_2D.append(recv4);

	# setling time
	recv1_stime = setling_time(recv1, setling_time_range, 0.01)
	recv2_stime = setling_time(recv2, setling_time_range, 0.01)
	recv3_stime = setling_time(recv3, setling_time_range, 0.01)
	recv4_stime = setling_time(recv4, setling_time_range, 0.01)

	print recv1_stime,recv2_stime,recv3_stime

	recv1_set_time.append(recv1_stime);
	recv2_set_time.append(recv2_stime);
	recv3_set_time.append(recv3_stime);
	recv4_set_time.append(recv4_stime);

# compute the average setling time
recv1_stime_avg = np.mean(recv1_set_time)
recv2_stime_avg = np.mean(recv2_set_time)
recv3_stime_avg = np.mean(recv3_set_time)
recv4_stime_avg = np.mean(recv4_set_time)

recv1_stime_std = np.std(recv1_set_time)
recv2_stime_std = np.std(recv2_set_time)
recv3_stime_std = np.std(recv3_set_time)
recv4_stime_std = np.std(recv4_set_time)

print '* setling time avg and std:'
print recv1_stime_avg
print recv2_stime_avg
print recv3_stime_avg

print recv1_stime_std
print recv2_stime_std
print recv3_stime_std

file_name = para_str+'_summary'+str(setling_time_range)+'.csv'

#addToFile(file_name, para_str+',Recv1_avg_time,Recv2_avg_time,Recv3_avg_time,Recv1_std_time,Recv2_std_time,Recv3_std_time,settling_time,settling_time_std,recv1_mean_stable_value,recv2_mean_stable_value,recv3_mean_stable_value,recv1_std_stable_value,recv2_std_stable_value,recv3_std_stable_value,stable_value,stable_value_std\n')

addToFile(file_name, str(ebuse)+',')
addToFile(file_name, str(recv1_stime_avg)+',')
addToFile(file_name, str(recv2_stime_avg)+',')
addToFile(file_name, str(recv3_stime_avg)+',')
addToFile(file_name, str(recv4_stime_avg)+',')

addToFile(file_name, str(recv1_stime_std)+',')
addToFile(file_name, str(recv2_stime_std)+',')
addToFile(file_name, str(recv3_stime_std)+',')
addToFile(file_name, str(recv4_stime_std)+',')

# ----------------------------------------------------------------------
# compute max settling time
max_stime = max([recv1_stime_avg, recv2_stime_avg,recv3_stime_avg,recv4_stime_avg])
max_stime_std = max([recv1_stime_std,recv2_stime_std,recv3_stime_std,recv4_stime_std])
print max_stime

addToFile(file_name, str(max_stime)+',')
addToFile(file_name, str(max_stime_std)+',')
	
# ----------------------------------------------------------------------	
# compute average and stdev and plot error bar
recv1_avg = []
recv1_std = []
recv2_avg = []
recv2_std = []
recv3_avg = []
recv3_std = []
recv4_avg = []
recv4_std = []
time_data = []

# total sample variance
recv1_stable_std = 0
recv2_stable_std = 0
recv3_stable_std = 0
recv4_stable_std = 0

recv1_stable_mean = 0
recv2_stable_mean = 0
recv3_stable_mean = 0
recv4_stable_mean = 0

row_size = 3590 #len(recv1_2D[0])-1
for row_i in range(0,row_size):
	col_value = column(recv1_2D, row_i);
	recv1_avg.append(np.mean(col_value))
	recv1_std.append(np.std(col_value))

	if(row_i > max_stime_std):
		recv1_stable_mean = recv1_stable_mean + np.mean(col_value)
		recv1_stable_std = recv1_stable_std + math.pow(np.std(col_value),2)
	
	col_value = column(recv2_2D, row_i);
	recv2_avg.append(np.mean(col_value))
	recv2_std.append(np.std(col_value))

	if(row_i > max_stime_std):
		recv2_stable_mean = recv2_stable_mean + np.mean(col_value)
		recv2_stable_std = recv2_stable_std + math.pow(np.std(col_value),2)
	
	col_value = column(recv3_2D, row_i);
	recv3_avg.append(np.mean(col_value))
	recv3_std.append(np.std(col_value))

	if(row_i > max_stime_std):
		recv3_stable_mean = recv3_stable_mean + np.mean(col_value)
		recv3_stable_std = recv3_stable_std + math.pow(np.std(col_value),2)
		
	col_value = column(recv4_2D, row_i);
	recv4_avg.append(np.mean(col_value))
	recv4_std.append(np.std(col_value))

	if(row_i > max_stime_std):
		recv4_stable_mean = recv4_stable_mean + np.mean(col_value)
		recv4_stable_std = recv4_stable_std + math.pow(np.std(col_value),2)
	
	time_data.append(row_i)
# ----------------------------------------------------------------------
# compute confidence
confidence(recv1_std,dim)
confidence(recv2_std,dim)
confidence(recv3_std,dim)
confidence(recv4_std,dim)

# ----------------------------------------------------------------------
over_ylim_max = 0.03
figure_name = 'figure/std_'+prefix +str(0)+count_str+str(i)+alpha_str+str(alpha)+gamma_str+str(gamma)+ebuse_str+str(ebuse)+'.jpg'
plot_error(figure_name, time_data[0::100], recv1_avg[0::100], recv2, recv3, recv1_std[0::100], 1, 1, over_ylim_max)

figure_name = 'figure/std_'+prefix +str(1)+count_str+str(i)+alpha_str+str(alpha)+gamma_str+str(gamma)+ebuse_str+str(ebuse)+'.jpg'
plot_error(figure_name, time_data[0::100], recv2_avg[0::100], recv2, recv3, recv2_std[0::100], 1, 1, over_ylim_max)


figure_name = 'figure/std_'+prefix +str(2)+count_str+str(i)+alpha_str+str(alpha)+gamma_str+str(gamma)+ebuse_str+str(ebuse)+'.jpg'
plot_error(figure_name, time_data[0::100], recv3_avg[0::100], recv2, recv3, recv3_std[0::100], 1, 1, over_ylim_max)


figure_name = 'figure/std_'+prefix +str(3)+count_str+str(i)+alpha_str+str(alpha)+gamma_str+str(gamma)+ebuse_str+str(ebuse)+'.jpg'
plot_error(figure_name, time_data[0::100], recv4_avg[0::100], recv3, recv4, recv4_std[0::100], 1, 1, over_ylim_max)

# subplot

fig = plt.gcf()
fig.set_size_inches(6.5, 10)

"""ax1 = fig.add_subplot(4, 1, 1)
ax2 = fig.add_subplot(4, 1, 2, sharex=ax1, sharey=ax1)
ax3 = fig.add_subplot(4, 1, 3, sharex=ax1, sharey=ax1)
ax4 = fig.add_subplot(4, 1, 4, sharex=ax1, sharey=ax1)
ax1.axhline(linewidth=4, color = 'black')
ax1.axvline(linewidth=4, color = 'black')

ax2.axhline(linewidth=4, color = 'black')
ax2.axvline(linewidth=4, color = 'black')
ax3.axhline(linewidth=4, color = 'black')
ax3.axvline(linewidth=4, color = 'black')
ax4.axhline(linewidth=4, color = 'black')
ax4.axvline(linewidth=4, color = 'black')"""

figure_name = 'figure/std_avg'+ alpha_str+str(alpha)+gamma_str+str(gamma)+ebuse_str+str(ebuse)+ '.jpg'
plot_error_sub(plt, 411, figure_name, time_data[0::100], recv1_avg[0::100], recv2, recv3, recv1_std[0::100], 1, 1, over_ylim_max)

plot_error_sub(plt, 412, figure_name, time_data[0::100], recv2_avg[0::100], recv2, recv3, recv2_std[0::100], 1, 1, over_ylim_max)


plot_error_sub(plt, 413, figure_name, time_data[0::100], recv3_avg[0::100], recv2, recv3, recv3_std[0::100], 1, 1, over_ylim_max)

plot_error_sub(plt, 414, figure_name, time_data[0::100], recv4_avg[0::100], recv3, recv4, recv4_std[0::100], 1, 1, over_ylim_max)

plt.savefig(figure_name, dpi = 300)
plt.close()
# compute total sample variance
recv1_stable_std_t = math.sqrt(recv1_stable_std/(len(recv1_2D[0])-5-max_stime_std))
recv2_stable_std_t = math.sqrt(recv2_stable_std/(len(recv1_2D[0])-5-max_stime_std))
recv3_stable_std_t = math.sqrt(recv3_stable_std/(len(recv1_2D[0])-5-max_stime_std))
recv4_stable_std_t = math.sqrt(recv4_stable_std/(len(recv1_2D[0])-5-max_stime_std))

recv1_stable_mean_t = (recv1_stable_mean/(len(recv1_2D[0])-5-max_stime_std))
recv2_stable_mean_t = (recv2_stable_mean/(len(recv1_2D[0])-5-max_stime_std))
recv3_stable_mean_t = (recv3_stable_mean/(len(recv1_2D[0])-5-max_stime_std))
recv4_stable_mean_t = (recv4_stable_mean/(len(recv1_2D[0])-5-max_stime_std))

print '* mean stable value: '
print recv1_stable_mean
print recv2_stable_mean
print recv3_stable_mean

print recv1_stable_mean_t
print recv2_stable_mean_t
print recv3_stable_mean_t
# print the last stdev
#print recv1_std[len(recv1_2D[0])-2], recv2_std[len(recv1_2D[0])-2], recv3_std[len(recv1_2D[0])-2]
print '* avg satble value std:'
print recv1_stable_std_t
print recv2_stable_std_t
print recv3_stable_std_t

addToFile(file_name, str(recv1_stable_mean_t)+',')
addToFile(file_name, str(recv2_stable_mean_t)+',')
addToFile(file_name, str(recv3_stable_mean_t)+',')
addToFile(file_name, str(recv4_stable_mean_t)+',')

addToFile(file_name, str(recv1_stable_std_t)+',')
addToFile(file_name, str(recv2_stable_std_t)+',')
addToFile(file_name, str(recv3_stable_std_t)+',')
addToFile(file_name, str(recv4_stable_std_t)+',')

# compute total mean and variance
stable_mean_t = np.mean([recv1_stable_mean_t,recv2_stable_mean_t,recv3_stable_mean_t,recv4_stable_mean_t])
stable_std_t = np.mean([recv1_stable_std_t,recv2_stable_std_t,recv3_stable_std_t,recv4_stable_std_t])

addToFile(file_name, str(stable_mean_t)+',')
addToFile(file_name, str(stable_std_t)+'\n')

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# load reward
inv = 10;
f_name = 'figure/reward_'+ alpha_str+str(alpha)+gamma_str+str(gamma)+ebuse_str+str(ebuse)
start = 35
res_row_size=3590
rewad_ylim_max = 4
prefix = 'outfile0_simple_360_0.1_tstNum_2000000_b1000'
rewad_ylim_max_total = 10
data_type = 3
load_reward(prefix, start, inv, res_row_size, f_name, rewad_ylim_max, rewad_ylim_max_total,  data_type);

# load instant reward
f_name = 'figure/reward_instant_'+ alpha_str+str(alpha)+gamma_str+str(gamma)+ebuse_str+str(ebuse)
start = 43
res_row_size=3590
prefix = 'outfile0_simple_360_0.1_tstNum_2000000_b1000'
rewad_ylim_max_total = 10
data_type = 3
load_reward(prefix, start, inv, res_row_size, f_name, rewad_ylim_max, rewad_ylim_max_total,  data_type);

# load instant traffic
f_name = 'figure/traffic_instant_'+ alpha_str+str(alpha)+gamma_str+str(gamma)+ebuse_str+str(ebuse)
start = 51
res_row_size=3590
rewad_ylim_max = 0.03
inv = 1
rewad_ylim_max_total = 10
data_type = 1
prefix = 'outfile0_simple_360_0.1_tstNum_2000000_b1000'
load_reward(prefix, start, inv, res_row_size, f_name, rewad_ylim_max, rewad_ylim_max_total,  data_type);

# load resource traffic
f_name = 'figure/res_'+ alpha_str+str(alpha)+gamma_str+str(gamma)+ebuse_str+str(ebuse)
start = 0
res_row_size=710
rewad_ylim_max = 400
inv = 1
rewad_ylim_max_total = 1000
data_type = 2
prefix = 'resouce_assign_360_0.1_tstNum_2000000_b1000'
load_reward(prefix, start, inv, res_row_size, f_name, rewad_ylim_max, rewad_ylim_max_total,  data_type);
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------

# resource file load and plot
"""recv1_2D = [];
recv2_2D = [];
recv3_2D = [];
recv4_2D = [];

recv_total_2D = []


for i in range(0,dim):
	prefix = 'resouce_assign_360_0.1_tstNum_2000000_b1000'
	file_name = prefix+alpha_str+str(alpha)+gamma_str+str(gamma)+ebuse_str+str(ebuse)+rep_str+str(i)+postfix #+str(0)+count_str+str(i)+alpha_str+str(alpha)+gamma_str+str(gamma)+ebuse_str+str(ebuse)+postfix
	recv1, recv2 = read_csv_data(file_name, 0,1)
	recv3, recv4 = read_csv_data(file_name, 2,3)
	
	zipped_list = zip(recv1, recv2, recv3, recv4)
	recv_total = [sum(item) for item in zipped_list]


	figure_name = 'figure1/'+prefix +str(0)+count_str+str(i)+str(1)+'.jpg'
	#plot_data(figure_name, recv1, recv2, recv3, 1)
	
	figure_name = 'figure1/'+prefix +str(0)+count_str+str(i)+str(1)+'.jpg'
	#plot_data(figure_name, recv2, recv2, recv3, 1)
	
	figure_name = 'figure1/'+prefix +str(0)+count_str+str(i)+str(1)+'.jpg'
	#plot_data(figure_name, recv3, recv2, recv3, 1)
	
	figure_name = 'figure1/'+prefix +str(0)+count_str+str(i)+str(1)+'.jpg'
	#plot_data(figure_name, recv4, recv2, recv3, 1)
	
	
	recv1_2D.append(recv1);
	recv2_2D.append(recv2);
	recv3_2D.append(recv3);
	recv4_2D.append(recv4);
	
	recv_total_2D.append(recv_total);


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

recv_avg_total = []
recv_std_total = []

time_data = []

print np.shape(recv1_2D)
res_row_size = 710
for row_i in range(0,res_row_size):
	
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
	
	col_value_total = column(recv_total_2D, row_i)
	recv_avg_total.append(np.mean(col_value_total))
	recv_std_total.append(np.std(col_value_total))
	
	time_data.append(5*row_i)
	#print np.std(col_value)
	#print column(recv1_2D, row_i)
	
# ---------------------------------------------------------------------
# compute confidence
confidence(recv1_std,dim)
confidence(recv2_std,dim)
confidence(recv3_std,dim)
confidence(recv4_std,dim)
confidence(recv_std_total,dim)
# ---------------------------------------------------------------------

time_inv = 0;
time_inv_vec = []
recv1_avg_inv = []
recv1_std_inv = []
recv2_avg_inv = []
recv2_std_inv = []
recv3_avg_inv = []
recv3_std_inv = []
recv4_avg_inv = []
recv4_std_inv = []
# sliding average over over inv elts
for row_i in range(0,len(recv1_2D[0])-inv):
	#print recv1_avg[time_inv:time_inv+inv]
	recv1_avg_inv.append(np.mean(recv1_avg[time_inv:time_inv+inv]));
	recv2_avg_inv.append(np.mean(recv2_avg[time_inv:time_inv+inv]));
	recv3_avg_inv.append(np.mean(recv3_avg[time_inv:time_inv+inv]));
	recv4_avg_inv.append(np.mean(recv4_avg[time_inv:time_inv+inv]));

	recv1_std_inv.append(0);
	recv2_std_inv.append(0);
	recv3_std_inv.append(0);
	recv4_std_inv.append(0);
	
	time_inv_vec.append(time_inv*5);
	time_inv = time_inv + 1;
#print recv1_avg_inv

res_ylim_max = 300;
figure_name = 'figure/std_res_'+prefix +str(0)+count_str+str(i)+alpha_str+str(alpha)+gamma_str+str(gamma)+ebuse_str+str(ebuse)+'.jpg'
plot_error(figure_name, time_data[0::inv], recv1_avg[0::inv], recv2, recv3, recv1_std[0::inv], 1, 2, res_ylim_max)

figure_name = 'figure/std_res_'+prefix +str(1)+count_str+str(i)+alpha_str+str(alpha)+gamma_str+str(gamma)+ebuse_str+str(ebuse)+'.jpg'
#plot_error(figure_name, time_data[0::inv], recv2_avg[0::inv], recv2, recv3, recv2_std[0::inv], 1, 2)
plot_error(figure_name, time_data[0::inv], recv2_avg[0::inv], recv2, recv3, recv2_std[0::inv], 1, 2, res_ylim_max)

figure_name = 'figure/std_res_'+prefix +str(2)+count_str+str(i)+alpha_str+str(alpha)+gamma_str+str(gamma)+ebuse_str+str(ebuse)+'.jpg'
plot_error(figure_name, time_data[0::inv], recv3_avg[0::inv], recv2, recv3, recv3_std[0::inv], 1, 2, res_ylim_max)

figure_name = 'figure/std_res_'+prefix +str(3)+count_str+str(i)+alpha_str+str(alpha)+gamma_str+str(gamma)+ebuse_str+str(ebuse)+'.jpg'
plot_error(figure_name, time_data[0::inv], recv4_avg[0::inv], recv2, recv3, recv4_std[0::inv], 1, 2, res_ylim_max)

figure_name = 'figure/std_res_total_'+prefix +str(3)+count_str+str(i)+alpha_str+str(alpha)+gamma_str+str(gamma)+ebuse_str+str(ebuse)+'.jpg'
plot_error(figure_name, time_data[0::inv], recv_avg_total[0::inv], recv2, recv3, recv_std_total[0::inv], 1, 2, 1000)

# subfigure
fig = plt.gcf()
fig.set_size_inches(8, 6.5)

figure_name = 'figure/std_res.jpg'
plot_error_sub(plt, 221, figure_name, time_data[0::inv], recv1_avg[0::inv], recv2, recv3, recv1_std[0::inv], 1, 2, res_ylim_max)

#plot_error(figure_name, time_data[0::inv], recv2_avg[0::inv], recv2, recv3, recv2_std[0::inv], 1, 2)
plot_error_sub(plt, 222, figure_name, time_data[0::inv], recv2_avg[0::inv], recv2, recv3, recv2_std[0::inv], 1, 2, res_ylim_max)

plot_error_sub(plt, 223, figure_name, time_data[0::inv], recv3_avg[0::inv], recv2, recv3, recv3_std[0::inv], 1, 2, res_ylim_max)

plot_error_sub(plt, 224, figure_name, time_data[0::inv], recv4_avg[0::inv], recv2, recv3, recv4_std[0::inv], 1, 2, 1000)

plt.savefig(figure_name, dpi = 300)
plt.close()"""






