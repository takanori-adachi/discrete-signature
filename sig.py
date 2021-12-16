#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 12:09:27 2021
Last modifeind on Mon Dec  6 08:51:53 JST 2021

Usage  :  python sig.py  data.dat  maximum_length_of_words  [half-life]*

@author: taka "Takanori Adachi"
"""


import numpy as np
import sys



# suffix for making extended event type
HEAD = '-'
TAIL = '+'
HETA = '*' # any of HEAD or TAIL



# read records like (time, event_type, value) from the file
# 'event_type' must be one-character such as '1', '2', or 'a', 'b', 'c'.
#
class Data(object):
	def __init__(self, fname):
		self.fname = fname
		self.fread()
		self.make()
	
	def fread(self):
		f = open(self.fname, "r")
		self.raw_data = []
		while True:
			line = f.readline()
			if not line:
				break
			if line.startswith(';'): # it's a comment
				continue
			flds = line.rstrip().split('\t' )
			self.raw_data.append([float(flds[0]), flds[1], float(flds[2])])
		# we should sort self.raw_data by its time field, here.
		#print("raw_data=",self.raw_data)

	def make(self):
		self.I = [] # event types
		
		# Collect time and event
		time_l = []
		for r in self.raw_data:
			t = r[0] # time
			if not (t in time_l):
				time_l.append(t)
			e = r[1] # event type
			if not (e in self.I):
				self.I.append(e)
		self.time_a = np.array(time_l) # time array
		self.I_d = len(self.I) # (orignal) data dimension, or number of event types
		self.barI_d = self.I_d * 2 # number of extended event types = size of I * {HEAD, TAIL}
		self.n = self.time_a.size # time-index horizon
		self.T = self.time_a[self.n-1] # time horizon
		
		# Make time diff
		self.delta_t = np.zeros(self.n-1) # time diff array
		for l in range(self.n-1):
			self.delta_t[l] = self.time_a[l+1] - self.time_a[l]
		
		# Make time dic
		self.t2i = {} # time to index dic
		i=0
		for t in time_l:
			self.t2i[t] = i
			i += 1
		#print("t2i=", self.t2i)
		
		# Make event dic
		self.barI = [] # extended event types
		self.ee2i = {} # extended event to index dic
		self.e2j = {} # event to index dic
		self.i2ee = {} # index to extended event dic
		j=0
		for e in self.I:
			self.e2j[e] = j
			i = j * 2
			# register HEAD type
			ee = e + HEAD
			self.barI.append(ee)
			self.ee2i[ee] = i  # head type
			self.i2ee[i] = ee
			# register HETA type; use same i as HEAD
			ee = e + HETA
			self.ee2i[ee] = i  # head-tail type
			i += 1
			# register TAIL type
			ee = e + TAIL
			self.barI.append(ee)
			self.ee2i[ee] = i # tail type
			self.i2ee[i] = ee
			j += 1
		#print("ee2i=", self.ee2i)
		
		# Make data matrix X
		self.X = np.zeros([self.n, self.I_d])
		cur_X = np.zeros(self.I_d)
		cur_t = -1.0
		for r in self.raw_data:
			t = r[0] # time
			if cur_t < 0:
				cur_t = t
			if t > cur_t:
				self.X[self.t2i[cur_t]] = cur_X
				cur_t = t
			j = self.e2j[r[1]] # index of event_type
			cur_X[j] = r[2] # value
		self.X[self.t2i[cur_t]] = cur_X
		
		# Make X diff
		self.delta_X = np.zeros([self.n-1, self.I_d])
		for l in range(self.n-1):
			self.delta_X[l] = self.X[l+1] - self.X[l]
	
	def w2mi(self, w): # word to multi-index
		iss = []
		e = ''
		for c in list(w):
			if e == '': # c is an event type
				e = c
			else: # c is a suffix
				iss.append(self.ee2i[e+c])
				e = ''
		return(iss)
	
	def mi2w(self, iss): # multi-index to word
		w = ""
		for i in iss:
			w += self.i2ee[i]
		return(w)



class Words(object):
	def __init__(self, data, k):
		self.I_d = data.I_d # the cardinality of data.I
		self.barI_d = data.barI_d # the cardinality of data.barI
		self.k = k # maximum length of words; assume k >= 1
		
		# make Istar
		# length == 0
		Iks = []
		Iks.append(['']) # add { lambda }
		
		# 1 <= length <= k
		for j in range(self.k):
			Ij = Iks[j]
			Ij1 = []
			for w in Ij:
				for i in data.barI:
					Ij1.append(w+i)
			Iks.append(Ij1) # add I(j+1)
		self.Istar = sum(Iks, []) # flatten list, or make a big union
	
		# make IstarHalf
		# length == 0
		Iks = []
		Iks.append(['']) # add { lambda }
		
		# length == 1
		I1 = []
		for i in data.I:
			I1.append(i+HETA)
		Iks.append(I1)
		
		# 2 <= length <= k
		for j in range(1,self.k):
			Ij = Iks[j]
			Ij1 = []
			for w in Ij:
				for i in data.barI:
					Ij1.append(w+i)
			Iks.append(Ij1) # add I(j+1)
		self.IstarHalf = sum(Iks, []) # flatten list, or make a big union
	
	def make_container(self): # generate container having [bool, value] pairs for each word as its leafs
		v_list = [] # list of nodes
		for u in range(self.barI_d):
			v_list.append(self.make_v(1))
		return([[False, 0.0], v_list])
	
	def make_v(self, j):
		v_list = [] # list of nodes
		if j < self.k:
			for _ in range(self.barI_d):
				v_list.append(self.make_v(j+1))
		return([[False, 0.0], v_list])
	
	def clear_container(self, v):
		v[0][0] = False
		for vv in v[1]:
			self.clear_container(vv)



class Signature(object): # discrete signature
	def __init__(self, data, k):
		self.data = data
		self.k = k # maximum length of words
		self.words = Words(self.data, self.k)
		self.v = None # container
	
	def set_container(self):
		if self.v == None:
			self.make_container()
		else:
			self.clear_container()
	
	def set_decay_rate(self, mu):
		self.mu = mu # decay_rate
		self.mu_delta_t = np.exp(-mu * data.delta_t)
	
	def set_half_life(self, hl):
		# convert half-life to decay-rate
		mu = 0.0
		if hl > 0:
			mu = np.log(2.0)/hl
		self.set_decay_rate(mu)
	
	def set_interval(self, t1, t2):
		self.t1 = t1
		self.t2 = t2
		self.i1 = data.t2i[t1]
		self.i2 = data.t2i[t2]
	
	def eval(self, w):
		if self.v != None: # use container
			v = self.sig0(self.i1, self.i2, data.w2mi(w))
		else:
			v = self.sig0_simple(self.i1, self.i2, data.w2mi(w))
		return(v)
	
	def print_val(self, v, w):
		self.prS(v, self.t1, self.t2, w)
	
	def sig(self, t1, t2, w):
		if self.v != None: # use container
			v = self.sig0(data.t2i[t1], data.t2i[t2], data.w2mi(w))
		else:
			v = self.sig0_simple(data.t2i[t1], data.t2i[t2], data.w2mi(w))
		#self.prS(v, t1, t2, w)
		return(v)
	
	def sig0(self, m, n, iss): # faster algorithm using container self.v
		c = self.get_c(m, n, iss)
		if c[0]: # if already computed
			return c[1] # return its value
		# otherwise, compute from scratch
		v = 1.0
		if len(iss) > 0:
			w = iss[:-1]
			i = iss[len(iss)-1]
			j, s = self.i2js(i)
			if s == 0: # suffix is '-'
				v = self.mu_delta_t[n-1] * ( self.sig0(m, n-1, iss) + data.delta_X[n-1,j] * self.sig0(m, n-1, w) )
				#v = self.mu_delta_t[n-1] * self.sig0(m, n-1, iss) + data.delta_X[n-1,j] * self.sig0(m, n-1, w)
			else: # suffix is '+'
				v = self.mu_delta_t[n-1] * self.sig0(m, n-1, iss) + data.delta_X[n-1,j] * self.sig0(m, n, w)
		c[0] = True # it is computed
		c[1] = v # and its value is 'v'
		return(v)
	
	def sig0_simple(self, m, n, iss): # algorithm using less memory, but slower than self.sig0
		v = 1.0
		if len(iss) > 0:
			i = iss[len(iss)-1]
			j, s = self.i2js(i)
			w = iss[0:len(iss)-1]
			v = 0.0
			for l in range(m, n):
				v += self.sig0_simple(m, l+s, w) * data.delta_X[l,j]
		#self.prS0(v, m, n, iss)
		return(v)
	
	def i2js(self, i): # extended index to (index, suffix_n) where suffix_n = 0 or 1
		j = int(i / 2)
		s = i - j * 2
		return j, s
	
	def prS0(self, v, m, n, iss):
		self.prS(v, data.time_a[m], data.time_a[n], data.mi2w(iss))
	
	def prS(self, v, t1, t2, w):
		print("S^(%1.2f)(X)_(%1.1f,%1.1f)^(%s)\t= %1.2f"%(self.mu,t1, t2, w, v))
		#if w == "": w = '\lambda'
		##print("S(X)_{%1.1f,%1.1f}^{%s}\t&= %1.2f \\\\"%(t1, t2, w, v))
		#print("S(X)_{%1.0f,%1.0f}^{%s}\t&= %1.0f, \\\\"%(t1, t2, w, v))
	
	def make_container(self):
		self.v = [] # this should be implemented as an array of pointers instead of list in C++
		for m in range(0, self.data.n - 1):
			v_list = []
			for n in range(m+1, self.data.n): # assume m < n
				v_list.append(self.words.make_container())
			self.v.append(v_list)
	
	def get_c(self, m, n, iss): # return [b, value] pair
		if m >= n:
			if len(iss) == 0: # it's a lambda
				return([True, 1.0])
			else:
				return([True, 0.0])
		v = self.v[m][n-(m+1)]
		for i in iss:
			v = v[1][i]
		return(v[0])
	
	def clear_container(self):
		for v_list in self.v:
			for v in v_list:
				self.words.clear_container(v)
	
	def print_all(self):
		if self.mu > 0: # weighted signature
			Istar = self.words.Istar
		else: # flat signature
			Istar = self.words.IstarHalf
		for w in Istar:
			v = self.eval(w)
			self.print_val(v, w)



def get_args():
	fname = ""
	k = 0
	hl_list = [] # list of half-lives
	args = sys.argv
	argn = len(args)
	if argn < 3:
		return "", k, hl_list
	fname = args[1]
	k = int(args[2])  # maximum length of words
	if argn == 3:
		hl_list.append(0.0) # compute only no decay case
	else:
		for n in range(3,argn):
			hl_list.append(float(args[n]))  # half-life
	return fname , k, hl_list



if __name__ == "__main__":
	fname, k, hl_list = get_args()
	if fname == "":
		sys.exit()
	data = Data(fname)
	#print(data.raw_data)
	print("T=",data.time_a)
	print("I=",data.I)
	print("X=",data.X)
	
	sig = Signature(data, k)
	words = sig.words

	sig.set_interval(0.0, data.T)
	for hl in hl_list:
		sig.set_container()
		sig.set_half_life(hl)
		print('half-life =', hl, ' decay-rate =', sig.mu)
		sig.print_all()

