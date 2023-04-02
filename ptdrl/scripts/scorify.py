
import numpy as np
import os

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from numpy import inf
from pathlib import Path
import csv
from matplotlib import pyplot as plt
import re
import math
import scipy.stats as stats
from scipy import stats
import itertools

# Open file of scores
dir_path = os.path.dirname(os.path.realpath(__file__))
name_file = 'score_ddqn_mdrnn_e_0_v_best'
#name_file = 'score3'
test_file = dir_path + "/important scores/Bayesian/scores_bayesian_genetic/"+ name_file +".csv"
#test_file = dir_path + "/important scores/Bayesian/scores_appld_vs_bayesian/"+ name_file +".csv"

environments = {0:"Open", 1:"Door", 2:"Curve", 3:"Obstacles"}

print(name_file)


def main():

	num_trajectories = 7
	global test_file

	with open(test_file, newline='') as csvfile:
		reader = csv.DictReader(csvfile)

		# avg_scores = [avg success, avg time, all, min_dist < 0.5 and min_vel < 0, 
		# min_dist < 0.5 and min_vel < -0.2 - -1.4, min_dist < 0.3 and min_vel < -0.2 - -1.4]

		# Lists and variables
		basic_scores = [[] for _ in range(2)]
		basic_scores[0].append(1)
		
		num_par_1 = 14
		num_par_2 = 30
		num_contexts = 4
		avg_time = [] # List of times spent for each episode
		num_rows = 0

		# Meshes, variables and value_func
		mesh_dist_relvel = np.zeros([num_par_1, num_par_2, num_contexts])
		changing_relvel = np.zeros([num_contexts])
		time_relvel = np.zeros([num_contexts])

		mesh_dist_robvel_value = np.zeros([num_par_1, 15, num_contexts])
		changing_robvel_value = np.zeros([num_contexts])
		time_robvel_value = np.zeros([num_contexts])

		mesh_dist_robvel = np.zeros([num_par_1, 15, num_contexts])
		changing_robvel = np.zeros([num_contexts])
		time_robvel = np.zeros([num_contexts])

		value_bayesian = [[] for _ in range(num_contexts)]
		value_rl = [[] for _ in range(num_contexts)]
		value_rl_total = []
		value = None
		
		
		for row in reader: # For each episode
			num_rows += 1
			basic_scores[0].append(float(row['success']))
			basic_scores[1].append(float(row['time']))

			dist_list = []
			vel_list = []
			rob_vel_list = []
			context_list = []

			# Split elements of scores in each episode
			for dist in row['min_dist'].strip('][').split(', '):
				if dist != '':
					dist_list.append(float(dist))
			for vel in row['min_vel'].strip('][').split(', '):
				if vel != '':
					vel_list.append(float(vel))
			for vel_r in row['rob_vel'].strip('][').split(', '):
				if vel_r != '':
					rob_vel_list.append(float(vel_r))
			for context in row['context'].strip('][').split(', '):
				if context != '':
					context_list.append(int(context))

			mesh, changing, time = create_mesh(dist_list, 1.5, num_par_1, 0.1, vel_list, 1.5, num_par_2, 0.1, rob_vel_list, context_list, mesh_func_time)
			mesh_dist_relvel += mesh
			changing_relvel += changing
			time_relvel += time

			mesh, changing, time = create_mesh(dist_list, 1.5, num_par_1, 0.1, rob_vel_list, 1.5, 15, 0.1, rob_vel_list, context_list, mesh_func_time)
			mesh_dist_robvel += mesh
			changing_robvel += changing
			time_robvel += time

			mesh, changing, time = create_mesh(dist_list, 1.5, num_par_1, 0.1, rob_vel_list, 1.5, 15, 0.1, rob_vel_list, context_list, mesh_func_value_bayesian)
			mesh_dist_robvel_value += mesh
			changing_robvel_value += changing
			time_robvel_value += time


			value_bayesian = value_func_bayesian(value_bayesian, dist_list, rob_vel_list, vel_list, context_list)
			value_rl, value_rl_total = value_func_rl(value_rl, value_rl_total, dist_list, rob_vel_list, vel_list, context_list)
			value = value_rl
			value_total = value_rl_total

		# Print basic_scores
		print(f"Success: mean = {np.mean(basic_scores[0])}, std  = {np.std(basic_scores[0])} | Time: mean = {np.mean(basic_scores[1])}, std  = {np.std(basic_scores[1])}")
		
		# Print value and variables by context
		print_by_context(changing_relvel, time_relvel, value, value_total, num_contexts)

		###### PLOT HISTOGRAM OF VALUE FUNCTION ####
		plot_hist(value, 'Value', 'time [seconds]', 'count')

		### MESH FOR DIST VEL VS REL VEL ####
		#plot_mesh(1.5, -1.5, 30, 1.5, 0.1, 14, mesh_dist_relvel, 'Time spent:', name_file, 'relative vel', 'dist ',num_rows)

		### MESH FOR DIST VEL VS ROB VEL ####
		plot_mesh(1.5, -0, 15, 1.5, 0.1, 14, mesh_dist_robvel, 'Time spent:', name_file, 'robot vel', 'dist',num_rows)

		### MESH FOR VALUE FUNCTION ####
		plot_mesh(1.5, 0, 15, 1.5, 0.1, 14, mesh_dist_robvel_value, 'Value', name_file, 'robot vel ', 'dist ',num_rows)

		### HISTOGRAM OF TIMES OF TRAJECTORIES ####
		####plot_hist(avg_time, 'Elapsed time of trajectories ', 'time [seconds]', 'count')



def plot_mesh(start_x, end_x, num_x, start_y, end_y, num_y, mesh_value, title, name_file, x_label, y_label, num_rows):

		y = np.linspace(start_y,end_y,num_y)
		x = np.linspace(start_x,end_x,num_x)

		hf = plt.figure()


		for i in range(4):
			sub_plot_num = "2" + "2"+ str(i+1)
			sub_plot_num = int(sub_plot_num)
			ha = hf.add_subplot(sub_plot_num, projection='3d')

			X, Y = np.meshgrid(x, y)

			ha.plot_surface(X, Y, mesh_value[:,:,i]/num_rows)

			plt.title(f" {title} -{environments[i]}" )
			plt.xlabel(f'{x_label}')
			plt.ylabel(f"{y_label}")

		plt.show()

def plot_hist(value, title, x_label, y_label):

	length = len(value)
	fig, axs = plt.subplots(1, length)


	for i in range(length):
		axs[i].hist(value[i])
		axs[i].set_title(f'{title}-{environments[i]}-{name_file}')
		axs[i].set_xlabel('x_label')
		axs[i].set_ylabel('y_label')

	plt.show()

def create_mesh(list_par_1, start_par_1, num_par_1, d_par_1, list_par_2, start_par_2, num_par_2, d_par_2, list_par_3, context_list, mesh_func):

	if(len(list_par_1) is not len(list_par_2)) or (len(list_par_2) is not len(context_list)):
		assert("Error: lists are not the same size")
	length = len(list_par_1)

	mesh = np.zeros([num_par_1, num_par_2, 4])
	changing_freq = np.zeros([4])
	time_spent = np.zeros([4])

	for i in range(length):

		if i < length-1 and context_list[i] != context_list[i+1]:
			changing_freq[context_list[i]] += 1
		time_spent[context_list[i]] += 1

		p_1 = start_par_1
		for itr in range(num_par_1):
			p_2 = start_par_2
			for jtr in range(num_par_2):
				if ((list_par_1[i] > p_1 - d_par_1 and list_par_1[i] <= p_1 and list_par_2[i] > p_2 - d_par_2 and list_par_2[i] <= p_2)):
					mesh[itr, jtr, context_list[i]] += mesh_func(i, list_par_1, list_par_2, list_par_3)
				p_2 -= d_par_2
			p_1 -= d_par_1
	
	return mesh, changing_freq, time_spent

def mesh_func_time(i, l_1, l2, l_3):
	return 1

def mesh_func_value_bayesian(i, l_1, l_2, l_3):
	min_range = l_1[i]
	rob_vel = l_3[i]

	f = 4*rob_vel*(-1 if min_range < 0.8 else 1) + 0.1*(-1 if min_range < 0.5 else 0)
	return f

def value_func_bayesian(value, l_1, l_2, l_3, context_list):
	if (len(l_1) is not len(l_2)) or (len(l_2) is not len(l_3)) or (len(l_3) is not len(context_list)):
		assert("Error: lists are not the same size")
	length = len(l_1)

	for i in range(length):
		min_range = l_1[i]
		rob_vel = l_2[i]
		rel_vel = l_3[i]
		#value[context_list[i]].append(4*rob_vel*(-1 if min_range < 0.8 else 1)+ 2*(-1 if min_range < 0.5 else 0))
		#value[context_list[i]].append(2*(-1 if min_range < 0.75 else 0))
		#value[context_list[i]].append(10*rel_vel*(1 if min_range < 0.8 else 0))

		value[context_list[i]].append(rob_vel*(-1 if min_range < 0.75 else 1) -1)

		#value[context_list[i]].append(4*rob_vel*np.tanh((min_range-0.75)/0.2))
	
	return value

def value_func_rl(value, value_total, l_1, l_2, l_3, context_list):
	if (len(l_1) is not len(l_2)) or (len(l_2) is not len(l_3)) or (len(l_3) is not len(context_list)):
		assert("Error: lists are not the same size")
	length = len(l_1)
	value_rl = [[] for _ in range(4)]
	value_rl_total = []

	for i in range(length):
		min_range = l_1[i]
		rob_vel = l_2[i]
		rel_vel = l_3[i]

		value_rl[context_list[i]].append(rob_vel*(-1 if min_range < 0.75 else 1) -1)
		value_rl_total.append(rob_vel*(-1 if min_range < 0.75 else 1) -1)

	for i in range(4):
		value[i].append(np.sum(value_rl[i]))
	value_total.append(np.sum(value_rl_total))
	
	return value, value_total


def print_by_context(changing_freq, time_spent, value, value_total, num_contexts):

	for i in range(num_contexts):
		print(f"Context: {environments[i]}")
		print(f"Time spent: {round(100*time_spent[i]/np.sum(time_spent),1)}%, Changing frequency: {round(100*changing_freq[i]/np.sum(time_spent),1)}%")
		print(f"Value: mean  = {round(np.mean(value[i]),2)}, median  = {round(np.median(value[i]),2)}, std  = {round(np.std(value[i]),2)}")
		print()
	print(f"Value total: mean  = {round(np.mean(value_total),2)}, median  = {round(np.median(value_total),2)}, std  = {round(np.std(value_total),2)}")
	
def print_ttest():
	contexts = [0,1,2,3]
	contexts_comb = list(itertools.combinations(contexts, 2))
	contexts_comb = list(set(contexts_comb))

	for i in range(len(contexts_comb)):
		test_file_1 = dir_path + "/important scores/Bayesian/scores_appld_vs_bayesian/"+ base_file + str(contexts_comb[i][0]) +".csv"
		test_file_2 = dir_path + "/important scores/Bayesian/scores_appld_vs_bayesian/"+ base_file + str(contexts_comb[i][1]) +".csv"
		print(f" pvalue for parameters {contexts_comb[i][0]} and {contexts_comb[i][0]} is : {stats.ttest_ind(rvs1,rvs2)}")
	print(contexts_comb)

	



if __name__ == '__main__':

    main()
