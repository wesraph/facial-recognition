#!/usr/bin/python

import numpy as np

norms = { "L1":lambda x: np.sum(np.abs(x)),
	  "L2":lambda x: np.sum(x**2),
	  "inf":lambda x: np.max(np.abs(x))
	}

def compute_distances(data, query, norm="L2"):
	""" Compute distances.

	Computes the distances between the vectors (rows) of a dataset and a
	single query). Three distances are supported:
	  * Manhattan distance ("L1");
	  * squared Euclidean distance ("L2");
	  * Chebyshev distance ("inf").

	:param data: Dataset matrix with samples as rows.
	:param query: Query vector
	:type data: (n,d)-sized Numpy array of floats
	:type query: (d)-sized Numpy array of floats

	:result: The distances of the data vectors to the query.
	:rtype: (n)-sized Numpy array of floats
	"""

	norm_function = norms[norm]
	distances = np.zeros((len(data),), dtype=np.float32)
	for i, d in enumerate(data):
		distances[i] = norm_function(d-query)
	return distances

def knn_search(data, query, k=1, norm='L2'):
	""" Brute-force k-NN search

	Performs a brute-force k-NN search for the given query in data.
	Three distance are supported:
	  * Manhattan distance ("L1");
	  * squared Euclidean distance ("L2");
	  * Chebyshev distance ("inf").

	:param data: Dataset matrix with samples as rows.
	:param query: Query vector
	:param k: Number of nearest neighbors to return
	:param norm: Distance to use ("L1", "L2" (default) or "inf")
	:type data: (n,d)-sized Numpy array of floats
	:type query: (d)-sized Numpy array of floats
	:type k: int
	:type norm: str

	:return: k nearest neighbors (as their indices in the input matrix),
		distances to the query
	:rtype: (k)-sized Numpy array of ints, (k)-sized Numpy array of floats
	"""
	distances = compute_distances(data, query, norm)
	if k == 1:
		min_idx = np.argmin(distances)
		return [min_idx], [distances[min_idx]]
	else:
		min_idx = np.argpartition(distances, k)[:k]
		return min_idx, distances[min_idx]

def radius_search(data, query, r=1., norm='L2'):
	""" Brute-force radius search

	Performs a brute-force radius search for the given query in data.
	Three distance are supported:
	  * Manhattan distance ("L1");
	  * squared Euclidean distance ("L2");
	  * Chebyshev distance ("inf").

	:param data: Dataset matrix with samples as rows.
	:param query: Query vector
	:param r: The search radius
	:param norm: Distance to use ("L1", "L2" (default) or "inf")
	:type data: (n,d)-sized Numpy array of floats
	:type query: (d)-sized Numpy array of floats
	:type r: float
	:type norm: str

	:return: data vectors (as their indices in the input matrix) within a
		radius of r to the query, and their distances to the query
	:rtype: (m)-sized Numpy array of ints, (m)-sized Numpy array of floats
	"""
	distances = compute_distances(data, query, norm)
	indices = np.where(distances <= r)[0]
	return indices, distances[indices]
