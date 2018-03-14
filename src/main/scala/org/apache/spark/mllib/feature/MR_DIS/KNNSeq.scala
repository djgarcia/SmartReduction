package org.apache.spark.mllib.feature.MR_DIS

import java.io.PrintWriter
import java.util.logging.{Level, Logger}

import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression.LabeledPoint

import scala.collection.mutable.ListBuffer

/**
 * Sequential kNN classifier.
 *
 * KNN is a classification algorithm where the instance we wish to classify is
 * compared with all the data points of the training set and given the most common
 * class among the k nearest points.
 *
 * @constructor Creates a new classifier.
 *
 * @author Alejandro González Rogel
 * @version 1.0.0
 */
class KNNSeq extends TraitSeqClassifier {

  /**
   * Path to the logger strings.
   */
  private val bundleName = "resources.loggerStrings.stringsKNNSeq";
  /**
   * Logger.
   */
//  private val logger = Logger.getLogger(this.getClass.getName(), bundleName);
  /**
   * Distance calculator (between data points).
   */
  val distCalc = new DistCalculator

  /**
   * Number of nearest neighbours.
   */
  var k = 1

  /**
   * Training dataset.
   */
  var trainingData: Iterable[LabeledPoint] = Iterable.empty[LabeledPoint]

  override def setParameters(args: Array[String]): Unit = {

    // Check if we have the correct number of parameters.
    if (args.size % 2 != 0) {
      //logger.log(Level.SEVERE, "KNNPairNumberParamError",
        //this.getClass.getName)
      throw new IllegalArgumentException()
    }
    for { i <- 0 until args.size by 2 } {
      try {
        args(i) match {
          case "-k" => k = args(i + 1).toInt
          case somethingElse: Any =>
            //logger.log(Level.SEVERE, "KNNWrongArgsError", somethingElse.toString())
            //logger.log(Level.SEVERE, "KNNISPossibleArgs")
            throw new IllegalArgumentException()
        }
      } catch {
        case ex: NumberFormatException =>
          //logger.log(Level.SEVERE, "KNNNoNumberError", args(i + 1))
          throw new IllegalArgumentException()
      }
    }

    // If ‘k’ has an incorrect value.
    if (k <= 0) {
      //logger.log(Level.SEVERE, "KNNWrongArgsValuesError")
      //logger.log(Level.SEVERE, "KNNPossibleArgs")
      throw new IllegalArgumentException()
    }
  }

  override def train(trainingSet: Iterable[LabeledPoint]): Unit = {
    trainingData = trainingSet
  }

  override def classify(inst: Vector): Double = {

    val closest = knearestClasses(inst)

    // Compute the most common class among the closest neighbours.
    val classification = closest.groupBy(t => t._1).maxBy(t => t._2.length)

    classification._1
  }

  override def classify(instances: Iterable[Vector]): Array[Double] = {

    var result = Array.ofDim[Double](instances.size)
    val iter = instances.iterator
    var count = 0
    while (iter.hasNext) {
      val inst = iter.next
      result(count) = classify(inst)
      count += 1
    }
    result

  }

  /**
   * Gets the closer instances to a given data point.
   *
   * //@param  distances  List with tuples (class, distance)
   * @return Vector with the k tuples (class, distance) of the closest instances.
   *     They are sorted in ascending order according to their distance to the point we
   *     we are evaluating.
   */
  def knearestClasses(inst: Vector): ListBuffer[(Double, Double)] = {

    // Compute distance to each datapoint.
    val distances = for { actualInst <- trainingData }
      yield (actualInst.label, distCalc.euclideanDistance(
      inst.toArray, actualInst.features.toArray))

    var closest: ListBuffer[(Double, Double)] = ListBuffer.empty
    var iter = distances.iterator

    closest += iter.next()

    // For each data point
    while (iter.hasNext) {
      var actualInst = iter.next
      // If we have not selected more than k instances.
      if (closest.size < k) {
        closest += actualInst
      } else {
        var maxDist = closest.maxBy((t) => t._2)._2
        // If the distance is shorter than the maximum distance found so far.
        if (actualInst._2 < maxDist) {
          closest(closest.indexOf(closest.maxBy((t) => t._2))) = actualInst
        }
      }
    }

    closest.sortBy(tupla => tupla._2)
  }

}
