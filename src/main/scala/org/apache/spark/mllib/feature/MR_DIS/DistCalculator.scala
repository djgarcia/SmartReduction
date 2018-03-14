package org.apache.spark.mllib.feature.MR_DIS

/**
 * Distance meassurement between points.
 *
 * @constructor Creates an instance with different algorithms to calculate the distanc
 *    between two given points.
 *
 * @author Alejandro González Rogel
 * @version 1.0.0
 */
class DistCalculator extends Serializable {

  /**
   * Calculates the euclidean distance between two points.
   *
   * This algorithm does not apply the exact euclidean formula in
   * order to save computational resources.
   *
   * @param point1  First point.
   * @param point2  Second point.
   */
  def euclideanDistance(point1: Array[Double],
                        point2: Array[Double]): Double = {

    var dist = 0.0
    for { i <- 0 until point1.size } {
      dist += Math.pow((point1(i) - point2(i)), 2)
    }

    dist
  }

}
