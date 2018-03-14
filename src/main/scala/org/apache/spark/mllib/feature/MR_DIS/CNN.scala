package org.apache.spark.mllib.feature.MR_DIS

import org.apache.spark.mllib.regression.LabeledPoint

import scala.collection.mutable.MutableList

/**
 * Condensed Nearest Neighbor (CNN) instance selection algorithm.
 *
 * Builds a set S from an original set T such that every datapoint of T is closer to a
 * point of S of the same class than to a point of S of a different class.
 *
 * @constructor Create a new CNN instance.
 *
 * @author Alejandro González Rogel
 * @version 1.0.0
 */
class CNN extends TraitSeqIS {

  /**
   * Distance calculator.
   */
  val distCalc = new DistCalculator

  def instSelection(data: Iterable[LabeledPoint]): Iterable[LabeledPoint] = {

    var s = new MutableList[LabeledPoint]

    // Start subset S with one instance
    s += data.head

    // Apply CNN
    var keepGoing = true
    while (keepGoing) {
      doTheLogic(s, data) match {
        case Left(inst) => s += inst
        case Right(_)   => keepGoing = false
      }

    }

    s

  }

  /**
   * Given two datasets, checks that, for every instance in the second dataset, there
   * exists an instance of the same class in the first dataset that is the closest to the
   * first datapoint.
   *
   * If the above is not true, returns the instance of the second dataset which closest 
   * neighbour is not of the same class.
   * @param  s  Dataset 1
   * @param  data Dataset 2
   * @return Instance of the second dataset which closest neighbour in the first one is not of the same class or “False”.
   */
  private def doTheLogic(
    s: Iterable[LabeledPoint],
    data: Iterable[LabeledPoint]): Either[LabeledPoint, Boolean] = {

    var iter = data.iterator
    var nextInstanceFound = false
    var nextInstance: LabeledPoint = null

    while (iter.hasNext && !nextInstanceFound) {
      val actualInst = iter.next()
      // Get the closest instance(s)
      val closestInst = closestInstances(s, actualInst)

      // Check if one of the closest instances belongs to the same class.
      var addToS = true
      for { instance <- closestInst } {
        if (instance.label == actualInst.label) {
          addToS = false
        }
      }

      // Return instance if all the closest instances belong to a different
      // class.
      if (addToS) {
        nextInstanceFound = true
        nextInstance = actualInst
      }
    }

    if (nextInstanceFound) {
      Left(nextInstance)
    } else {
      // End of the algorithm.
      Right(false)

    }

  }

  /**
   * Gets the closest instance(s) to a given element.
   *
   * @param  s  Total dataset.
   * @param  inst  Datapoint.
   * @return  Closest datapoint.
   */
  private def closestInstances(s: Iterable[LabeledPoint],
                               inst: LabeledPoint): Iterable[LabeledPoint] = {

    var minDist = Double.MaxValue
    var closest: MutableList[LabeledPoint] = MutableList.empty[LabeledPoint]
    var iterador = s.iterator
    while (iterador.hasNext) {
      var actualInstance = iterador.next()
      val actualDist = distCalc.euclideanDistance(
        inst.features.toArray, actualInstance.features.toArray)
      if (actualDist == minDist) {
        closest += actualInstance
      } else if (actualDist < minDist) {
        minDist = actualDist
        closest = MutableList.empty[LabeledPoint]
        closest += actualInstance
      }
    }

    closest
  }

  /**
   *
   * CNN do not require any parameter.
   *
   * Unused method.
   *
   */
  override def setParameters(args: Array[String]): Unit = {
  }

}
