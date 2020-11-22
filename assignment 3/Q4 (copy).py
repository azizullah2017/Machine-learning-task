#!/usr/bin/python
import numpy as np
# import pandas as pd
import sys
print(sys.version)
import math

class NeturalNets:
	"""docstring for ClassName"""
	def __init__(self,input_layer,no_hindden_layer,target_layer,baised,wieghts):
		self.input_layer = input_layer
		self.no_hindden_layer = no_hindden_layer
		self.hindden_layer = []
		self.target_layer = target_layer
		self.wieghts = new_list = wieghts[:]
		self.update_wieghts = wieghts[:]
		self.baised = baised
		self.input_nets = []
		self.output_nets = []
		self.output = []
		self.error = []
		self.weight_index = 0
		self.final_wieghts = []


	def forward_propagation(self):
		print ("len1: ",len(self.wieghts))
		print ("len2: ",len(self.hindden_layer))
		self.weight_index = 0
		self.hindden_layer = []
		self.output = []
		# iterate over the input layer and find the values of hidden layers
		for node_h in range(self.no_hindden_layer):
			tmp =  0
			print (len(self.input_layer))
			for node_i in range(len(self.input_layer)):
				print ("wieghts",self.weight_index)
				print (node_i)
				tmp = self.input_layer[node_i]*self.wieghts[self.weight_index] + tmp
				# print (self.wieghts[self.weight_index],self.input_layer[node_i])
				#moving to weight node
				self.weight_index = self.weight_index + 1

			# got all the nets
			self.input_nets.append(tmp + self.baised[0])
			# print (" ",tmp + self.baised[0],self.baised[0])
			# print ("===============================================")
			# applying the activation function
			self.hindden_layer.append(1/(1+(math.exp(-1*(tmp + self.baised[0])))))

		print (len(self.hindden_layer))
		# iterate over the input layer and find the values of hidden layers
		for node_h in range(len(self.target_layer)):
			tmp =  0
			for node_i in range(len(self.hindden_layer)):
				print ("=>",self.weight_index)
				# print (self.wieghts[self.weight_index],self.hindden_layer[node_i])
				tmp = self.wieghts[self.weight_index]*self.hindden_layer[node_i] + tmp
				self.weight_index = 1 + self.weight_index

			self.output.append(1/(1+(math.exp(-1*(tmp + self.baised[1])))))
			self.output_nets.append(tmp + self.baised[1])

		sum_errors = 0
		print (len(self.output))
		for out in range(len(self.output)):
			sum_errors = sum_errors + ((self.target_layer[out] - self.output[out])**2)/2
		
		self.weight_index = self.weight_index-1
		# print (self.hindden_layer)
		return sum_errors

	def back_propagation(self,n):
		# updating the ouput layer weights
		# n = 1
		# print (self.hindden_layer)
		error_out = self.output[0] - self.target_layer[0]	
		out_net = (self.output[0])*(1- self.output[0])
		# print ("value : ",out_net)
		# print ("value : ",error_out)
		# print ("value : ",out_net)
		# print (self.wieghts)
		for node_h in self.hindden_layer:
			error_wieght = error_out * out_net * node_h
			# print (error_wieght)
			# print (self.wieghts[self.weight_index])
			self.update_wieghts[self.weight_index] = self.wieghts[self.weight_index] - n * error_wieght
			# print (self.weight_index,node_h, error_wieght)
			# print (self.wieghts[self.weight_index])
			self.weight_index = self.weight_index -1
		# print (self.input_nets)
		# print ("index",self.weight_index)
		# abc = len(self.hindden_layer)*len(self.input_layer)
		# weights_hindden_layer = self.weight_index
		print ("++++++++++++++++++++++++++++")
		print (self.wieghts)
		print (self.update_wieghts)
		print ("++++++++++++++++++++++++++++")
		index = self.weight_index+1
		netN = 0
		inputs_index = 0
		for weights_index in range(self.weight_index+1):
			# print (weights_index)
			# print(error_out, out_net)
			error_netF = error_out * out_net
			# print ("error_netF : ",error_netF)
			netF_outNode = self.wieghts[index]
			print ("weight :",netF_outNode)

			# check for next node of hidden layer
			if (weights_index+1) % len(self.input_layer)  == 0:
				index = index + 1
				# if weights_index !=1: 
				inputs_index = 0
				outN = self.hindden_layer[int(weights_index / len(self.input_layer))]
				# print ("====================>",int(weights_index / len(self.input_layer)))
			elif weights_index == 0:
				# print ("====================>",int(weights_index / len(self.input_layer)))
				outN = self.hindden_layer[int(weights_index / len(self.input_layer))]
			print ("val=>",error_netF , netF_outNode)
			error_outN = error_netF * netF_outNode
			outN_netN = outN*(1-outN)
			print (weights_index,inputs_index)
			if weights_index ==1:
				netN_W = self.input_layer[inputs_index+1]
			else:
				netN_W = self.input_layer[inputs_index]

			print ("out : ",error_outN)
			print ("outN_netN : ",outN,outN_netN)
			# print ("out : ",weights_index, error_outN, outN_netN, netN_W)
			error_w = error_outN * outN_netN * netN_W
			print (error_outN , outN_netN , netN_W,inputs_index,weights_index)
			print ("cal : ",error_w)
			print ("updated : ",self.wieghts[weights_index] - n * error_w)
			print ("self.weight_index",self.weight_index)
			print ("==========================================")
			# updating wieghts of hidden layer
			self.update_wieghts[self.weight_index] = self.wieghts[weights_index] - n * error_w
			# self.update_wieghts[weights_index] = error_w
			inputs_index = inputs_index + 1
			print ("index",index)
			self.weight_index = self.weight_index -1

		self.wieghts = self.update_wieghts[:]
		print (self.wieghts)
		print (self.update_wieghts)
		print ("++++++++++++++++++++++++++++")
		# print (self.update_wieghts)

	def train(self):
		# error = self.forward_propagation()
		# self.back_propagation(n=1)
		from tqdm import tqdm
		minima = 10
		for i in tqdm(range(0, 30), total = 30, desc ="epoch"): 
		# for x in range(100):
			error = self.forward_propagation()
			self.back_propagation(n=1)
			if error < minima:
				# print (error)
				minima = error 
				self.final_wieghts = self.update_wieghts[:]
		# print (self.forward_propagation())
		# self.back_propagation()
		print ("====================================")
		print ("minima",minima)
		print (self.final_wieghts)
		print ("====================================")

		
		# print (i)
		# print (self.forward_propagation())
		# print (self.back_propagation())

input_layer = [2,3]
no_hindden_layer = 3
target_layer = [0.1]
baised = [-0.2, -0.1]
wieghts = [0.1,-0.2,0.0,0.2,0.3,-0.4,0.3,0.3,-0.4]

NN_obj = NeturalNets(input_layer,no_hindden_layer,target_layer,baised, wieghts)
print (NN_obj.train())

# print (NN_obj.forward_propagation())
# print (NN_obj.back_propagation())
