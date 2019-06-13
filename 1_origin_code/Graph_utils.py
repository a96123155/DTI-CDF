import numpy as np
import itertools
from copy import deepcopy
from math import exp

def get_edge_list(sName):
	edge_list = []
	for line in open(sName).readlines():
		edge_list.append(line.strip().split())
	return edge_list




def get_adj_matrix_from_relation(aRelation,dDs,dTs):
	#print 'aRelation',aRelation
	adj = np.zeros((len(dDs.keys()),len(dTs.keys())))
	for element in aRelation:
		i = dDs[element[0]]
		j = dTs[element[1]]
		adj[i][j] = 1
	return adj

def make_sim_matrix(edge_list,dMap):
	sim = np.zeros((len(dMap.keys()),len(dMap.keys())))
	for a,b,c in edge_list:
		if float(c) > 0.0:	
			i = dMap[a]
			j = dMap[b]
			sim[i][j] = float(c)
			sim[j][i] = float(c)

		#fill diagonal with zeros
	np.fill_diagonal(sim,1)
	return sim


def get_All_D_T_thier_Labels_Signatures(R_all_train_test) :
    #print 'R_all_train_test',R_all_train_test
    
    D =[]
    T = []
    DT_signature = {}
    aAllPossiblePairs = []

    #R files
    with open(R_all_train_test, 'r') as f:
        data = f.readlines()
        #print 'data',data
        for line in data:
            #print 'line',line
            (a, b) = line.split()
            #print 'a,b',a,b
            D.append(a)
            T.append(b)


    #give labels for each Drugs and Targets
    D = list(set(D))
    T = list(set(T))


    dDs = dict([(x, j) for j, x in enumerate(D)])
    dTs = dict([(x, j) for j, x in enumerate(T)])
    diDs = dict([(j, x) for j, x in enumerate(D)])
    diTs = dict([(j, x) for j, x in enumerate(T)]) 
    aAllPossiblePairs = list(itertools.product(D,T))
    return D,T,DT_signature,aAllPossiblePairs,dDs,dTs,diDs,diTs


def get_two_hop_similarities(S1,S2):
	row1,col1 = S1.shape
	row2,col2 = S2.shape
	accum = np.zeros((row1,col2))
	maximum = np.zeros((row1,col2))
	#print row1,row2,col1,col2
	for i in range(row1):
		for j in range(col2):
			sim = []
			for k in range(row2):
				#print i,j,k
				if i!=j and j!=k and k!=i:
					#print S1[i][k],S2[k][j]
					mul = S1[i][k]*S2[k][j]
					accum[i][j]+=mul
					sim.append(mul)
			if sim != []:
				maximum[i][j] = max(sim)

	#for i in range(row1):
	#	accum[i,:]/=sum(accum[i,:])
	#	maximum[i,:]/=sum(maximum[i,:])
	return (accum,maximum)


def get_drug_relation(D_sim,DT,norm=True):
	row1,col1 = D_sim.shape
	row2,col2 = DT.shape

	accum = np.zeros((row1,col2))
	maximum = np.zeros((row1,col2))

	for i in range(row1):
		for j in range(col2):
			sim = []
			for k in range(row2):
				if k!=i: #no loops between drugs DDT
					mul = D_sim[i][k]*DT[k][j]
					accum[i][j]+= mul
					sim.append(mul)
			if sim!=[]:
				maximum[i][j] = max(sim)

	if norm:
		for i in range(row1):
			if (sum(accum[i,:])):
				accum[i,:]/=sum(accum[i,:])
			if (sum(maximum[i,:])):
				maximum[i,:]/=sum(maximum[i,:])
	return (accum,maximum)

def get_relation_target(T_sim,DT,norm=True):
	row1,col1 = DT.shape
	row2,col2 = T_sim.shape

	accum = np.zeros((row1,col2))
	maximum = np.zeros((row1,col2))

	for i in range(row1):
		for j in range(col2):
			sim = []
			for k in range(row2):
				if k!=j: #no loops between targets DTT
					mul = DT[i][k]*T_sim[k][j]
					accum[i][j]+= mul
					sim.append(mul)
			if sim!=[]:
				maximum[i][j] = max(sim)
	if norm:
		for i in range(row1):
			if (sum(accum[i,:])>0):
				accum[i,:]/=sum(accum[i,:])
			if (sum(maximum[i,:])>0):
				maximum[i,:]/=sum(maximum[i,:])

	return (accum,maximum)


def get_DTDT(DT):
	
	TD = np.transpose(DT)
	row1,col1 = DT.shape
	row2,col2 = TD.shape

	DD = np.zeros((row1,col2))
	for i in range(row1):
		for j in range(col2):
			for k in range(row2):
				if i!=j: #no loops between targets DTT
					mul = DT[i][k]*TD[k][j]
					DD[i][j]+= mul

	accum,maximum = get_drug_relation(DD,DT)
	return accum,maximum


def get_DDTT(DT,D_sim,T_sim):
	DDT_accum,DDT_max = get_drug_relation(D_sim,DT,False)
	DDTT_accum,_ = get_relation_target(T_sim,DDT_accum)
	_,DDTT_max = get_relation_target(T_sim,DDT_max)
	return DDTT_accum,DDTT_max





def get_feature_vector(all_pairs,mat,diDs,diTs):
	feature_vector = []

	for D,T in all_pairs:
		i=diDs[D]
		j=diTs[T]
		feature_vector.append(mat[i][j])
	return feature_vector


def neigborhood_prob(mat,sim,k=5):
	(row,col) = mat.shape
	
	indexZero = np.where(~mat.any(axis=1))[0]
	numIndexZeros = len(indexZero)

	np.fill_diagonal(sim,0)
	if numIndexZeros > 0:
		sim[:,indexZero] = 0


	for i in range(row):
		currSimForZeros = sim[i,:]
		indexRank = np.argsort(currSimForZeros)#<- rank(currSimForZeros)

		indexNeig = indexRank[-k:]
		#indexNeig = np.where(indexRank < (row - k -1))
		#indexNeig = indexRank[-1*(k+1):-1]
		
		simCurr = currSimForZeros[indexNeig]

		mat_known = mat[indexNeig, :]
		
		#print sum(simCurr),np.dot(simCurr ,mat_known)
		#print  
		if sum(simCurr)>0:
			mat[i,: ] = np.dot(simCurr ,mat_known) / sum(simCurr)
		#print mat[i,:]
	#print mat
	return mat

def mask_matrix(mat,test_pairs,transpose=False):
	
	new_mat = deepcopy(mat)
	for i,j in test_pairs:
		if transpose:
			new_mat[j,i] = 0
		else:
			new_mat[i,j] = 0
	return new_mat


def mat2vec(mat):
	return list(mat.reshape((mat.shape[0]*mat.shape[1])))




def impute_zeros(inMat,inSim,k=5):
	
	mat = deepcopy(inMat)
	sim = deepcopy(inSim)
	(row,col) = mat.shape
	np.fill_diagonal(mat,0)


	indexZero = np.where(~mat.any(axis=1))[0]
	numIndexZeros = len(indexZero)

	np.fill_diagonal(sim,0)
	if numIndexZeros > 0:
		sim[:,indexZero] = 0
	for i in indexZero:
		currSimForZeros = sim[i,:]
		indexRank = np.argsort(currSimForZeros)

		indexNeig = indexRank[-k:]
		simCurr = currSimForZeros[indexNeig]

		mat_known = mat[indexNeig, :]
		
		
		if sum(simCurr) >0:  
			mat[i,: ] = np.dot(simCurr ,mat_known) / sum(simCurr)
		
	
	return mat


def func(x):
	return exp(-1*x)


def Get_GIP_profile(adj,t):
	'''It assumes target drug matrix'''

	bw = 1
	if t == "d": #profile for drugs similarity
		ga = np.dot(np.transpose(adj),adj)
	elif t=="t":
		ga = np.dot(adj,np.transpose(adj))
	else:
		sys.exit("The type is not supported: %s"%t)

	ga = bw*ga/np.mean(np.diag(ga))
	di = np.diag(ga)
	x =  np.tile(di,(1,di.shape[0])).reshape(di.shape[0],di.shape[0])
	#z = np.tile(np.transpose(di),(di.shape[0],1)).reshape(di.shape[0],di.shape[0])



	d =x+np.transpose(x)-2*ga
	
	f = np.vectorize(func)
	return f(d)
