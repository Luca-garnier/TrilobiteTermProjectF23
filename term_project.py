import numpy as np
import math

#---------------------------------------------SCRIPT VARIABLES---------------------------------------------#
EUCLIDEAN_DISTANCE_ARRAY = []
COSINE_SIMILARITY_ARRAY = []
JACCARD_SIMILARITY_ARRAY = []

trilobiteMatrix = np.array([
    [1,0,0,0,1,1,0,0,0,0,0,1,0,0,1,0,0,1,0,1,0,0],
    [1,0,0,0,1,1,0,0,0,0,0,1,0,0,1,0,0,1,0,1,0,0],
    [1,0,0,0,1,1,0,0,0,0,0,1,0,0,1,0,0,1,0,1,0,0],
    [0,0,1,0,0,1,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0],
    [0,0,1,0,0,1,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0],
    [1,0,0,0,1,1,0,0,0,0,0,1,0,0,1,0,0,1,0,1,0,0],
    [0,0,0,0,1,1,0,0,1,1,0,1,1,1,1,0,0,0,0,0,0,0],
    [1,0,0,0,1,1,0,0,0,0,0,1,1,0,1,0,0,0,0,0,0,0],
    [1,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0],
    [1,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0],
    [1,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0],
    [0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0],
    [0,1,0,0,1,0,0,1,1,1,0,1,0,0,1,0,0,1,0,1,0,0],
    [0,1,0,0,1,1,0,0,1,1,0,1,0,0,1,0,0,0,0,0,0,0],
    [0,1,0,0,1,1,0,0,0,1,1,0,0,0,1,0,0,0,0,0,0,0]
    ])


#---------------------------------------------COMPARISON TECHNIQUES FUNCTIONS---------------------------------------------#
#Euclidean distance:
#This technique measures the straight-line distance between two points in n-dimensional space.
#It is a simple and widely used distance metric and can be used to compare vectors that represent different characteristics found in different orders of trilobites.
def euclidean_distance(vector1, vector2):
    if len(vector1) != len(vector2):
        raise ValueError("Vectors must be of the same length")
    distance = 0
    for i in range(len(vector1)):
        distance += (vector1[i] - vector2[i])**2
    distance = math.sqrt(distance)
    return distance


#Cosine similarity:
#This technique measures the cosine of the angle between two vectors in n-dimensional space.
#It is useful when the magnitude of the vectors is not important and can be used to compare vectors that represent different characteristics found in different orders of trilobites.
def cosine_similarity(vector1, vector2):
    if len(vector1) != len(vector2):
        raise ValueError("Vectors must be of the same length")
    dot_product = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    similarity = dot_product / (norm1 * norm2)
    return similarity


#Jaccard similarity:
#This technique measures the similarity between two sets of data by comparing the number of common elements between them.
#It is useful when the data is binary or categorical and can be used to compare vectors that represent different characteristics found in different orders of trilobites.
def jaccard_similarity(vector1, vector2):
    if len(vector1) != len(vector2):
        raise ValueError("Vectors must be of the same length")
    intersection = sum(a & b for a, b in zip(vector1, vector2))
    union = sum(a | b for a, b in zip(vector1, vector2))
    similarity = intersection / union if union != 0 else 0
    return similarity


#---------------------------------------------HELPER FUNCTIONS---------------------------------------------#
#Helper function to return trilobiteMatrix excluding reference vector (targetted by index)
def trilobiteMatrixWithoutRowX(excluded_row_index):
        return np.concatenate((trilobiteMatrix[:excluded_row_index], trilobiteMatrix[excluded_row_index+1:]), axis=0)

#Helper function to return result array for given comparison technique (euclidean distance, cosine similarity or jaccard similarity)
#In the returned result array, the {euclideanDistanceVector | cosSimVector | jaccSimVector} of each vector from the trilobiteMatrix is found @ his respective index in the trilobiteMatrix.
#For instance, if calculating enclidean distance, the euclideanDistanceVector of trilobiteMatrix[i] can be found @ finalEuclideanDistanceArray[i]
def resultingComparisonArray(func):
    numberOfArtifacts = len(trilobiteMatrix)
    finalResultArray = list(range(numberOfArtifacts))
    for i in range(numberOfArtifacts):
       finalResultArray[i] = func(trilobiteMatrix[i], trilobiteMatrixWithoutRowX(i))
    return finalResultArray

#Given a reference vector and a matrix, calculate the euclidean distance between reference vector and all comparative vectors included in matrix.
#Return resulting distances in distance vector. 
def euclideanDistanceVector(referenceVector, matrix):
    euclideanDistances = []
    for row in matrix:
        dst = euclidean_distance(referenceVector, row)
        euclideanDistances.append(dst)
    return euclideanDistances

#Given a reference vector and a matrix, calculate the cosine similarity between reference vector and all comparative vectors included in matrix.
#Return resulting similarity scores in cosSimScores vector. 
def cosSimVector(referenceVector, matrix):
    cosSimScores = []
    for row in matrix:
        cosSim = cosine_similarity(referenceVector, row)
        cosSimScores.append(cosSim)
    return cosSimScores

#Given a reference vector and a matrix, calculate the jaccard similarity between reference vector and all comparative vectors included in matrix.
#Return resulting similarity scores in jaccSimScores vector. 
def jaccSimVector(referenceVector, matrix):
    jaccSimScores = []
    for row in matrix:
        jacSim = jaccard_similarity(referenceVector, row)
        jaccSimScores.append(jacSim)
    return jaccSimScores

def main():
    global EUCLIDEAN_DISTANCE_ARRAY
    global COSINE_SIMILARITY_ARRAY
    global JACCARD_SIMILARITY_ARRAY

    EUCLIDEAN_DISTANCE_ARRAY = resultingComparisonArray(euclideanDistanceVector)
    COSINE_SIMILARITY_ARRAY = resultingComparisonArray(cosSimVector)
    JACCARD_SIMILARITY_ARRAY = resultingComparisonArray(jaccSimVector)
    
    ARRAY_MAP = {
    'EucDst': EUCLIDEAN_DISTANCE_ARRAY,
    'CosSim': COSINE_SIMILARITY_ARRAY,
    'JacSim': JACCARD_SIMILARITY_ARRAY
    }

    for key, value in ARRAY_MAP.items():
        print(f"COMPARISON TECHNIQUE: {key}\n {value}\n\n\n")






if __name__ == "__main__":
    main()