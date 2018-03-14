package org.apache.spark.mllib.feature.MR_DIS

import java.io.PrintWriter

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression.LabeledPoint

import scala.collection.mutable.ListBuffer

/**
 *
 * Set of serializable methods that are used by the [[KNN]] algorithm.
 *
 * This methods have been implemented in a different class in order to reduce the number 
 * of methods that need to be shared between all the working nodes.
 *
 * This implementation is based on the work made by Maillo et al., that was adapted to
 * fulfil al the requirements needed to be run by this library.
 * 
 * Jesús Maillo, Isaac Triguero and Francisco Herrera. "Un enfoque MapReduce del algoritmo
 * k-vecinos más cercanos para Big Data" In Actas de la XVI Edición Conferencia de la
 * Asociación Española para la Inteligencia Artificial CAEPIA 2015
 * Repository: https://github.com/JMailloH/kNN_IS
 *
 * @author Alejandro González Rogel
 * @version 1.0.0
 */
@SerialVersionUID(1L)
private class KNNInNodes extends Serializable {


  var k = 0
  var distType = 0
  var subdel = 0

  /**
   * Given a training and dataset and a test set, it returns the classes of the K
   * nearest neighbours of each test point.
   *
   * @param iter Training set iterator. It reads instances one at a time.
   * @param testSet Test set.
   *
   * @return Iterator for the returned data. This new data is composed by pairs of the following values:
   *   ID of the instance that was classified.
   *   Ordered list of the K nearest neighbours.
   */
  def knn(iter: Iterator[LabeledPoint], testSet: Broadcast[Array[Vector]]): Iterator[(Long, ListBuffer[(Double, Double)])] = { 

    // Initialization
    var results: ListBuffer[(Long, ListBuffer[(Double, Double)])] = ListBuffer.empty

    // Create object KNN with the necesary information
    val knn = new KNNSeq()
    val knnParameters: Array[String] = Array.ofDim(2)
    knnParameters(0) = "-k"
    knnParameters(1) = k.toString()
    knn.setParameters(knnParameters)
    knn.train(iter.toList)

    var auxSubDel: Long = subdel
    for { instancia <- testSet.value } {
      results += ((auxSubDel, knn.knearestClasses(instancia)))
      auxSubDel = auxSubDel + 1
    }

    results.iterator

  }

  /**
   * Given two ordered distance lists, it merges them and returns the K nearest
   * points.
   *
   * Input lists must be ordered: first element must be the closes one.
   *
   * @param mapOut1 An ordered list of pairs class-distance.
   * @param mapOut2 An ordered list of pairs class-distance.
   * //@result An ordered list of pairs class-distance with the closest elements of
   * both inputs.
   */

  def combine(mapOut1: ListBuffer[(Double, Double)], mapOut2: ListBuffer[(Double, Double)]): ListBuffer[(Double, Double)] = {

    val numNeighbors = mapOut1.length
    var itOut1 = 0

    for (j <- 0 to numNeighbors - 1) { // Loop for the k neighbors
      if (mapOut1(itOut1)._2 <= mapOut2(j)._2) {
        mapOut2(j) = (mapOut1(itOut1)._1, mapOut1(itOut1)._2)
        itOut1 = itOut1 + 1
      }
    }

    mapOut2
  }

  /**
   * Given a set of close instances, it returns the most common class.
   *
   * //@param Tuple with an instance ID and a list of class-distance of the
   *    closest neighbours
   * @return Tuple with the instance ID and its predicted class.
   */
  def calcPredictedClass(tupla: (Long, ListBuffer[(Double, Double)])): (Long, Double) = {

    val classification = tupla._2.groupBy(t => t._1).maxBy(t => t._2.length)

    (tupla._1, classification._1)

  }

}
