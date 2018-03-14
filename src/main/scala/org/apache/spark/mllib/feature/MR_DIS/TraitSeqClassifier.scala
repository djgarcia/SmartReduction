package org.apache.spark.mllib.feature.MR_DIS

import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression.LabeledPoint

/**
 * Defines all the basic methods that a sequential classifier needs to include
 * in order to be executed by this library.
 *
 * @author Alejandro González Rogel
 * @version 1.0.0
 */
trait TraitSeqClassifier {

  /**
   * Train the classifier.
   *
   * @param  trainingSet  Training set.
   */
  def train(trainingSet: Iterable[LabeledPoint]): Unit

  /**
   * Classify one instance.
   *
   * @param inst  Instance
   * @return Class predicted by the classifier.
   */
  def classify(inst: Vector): Double

  /**
   * Classify a set of instances.
   *
   * @param instances  Instances to classify.
   * @return Class predicted for each instance.
   */
  def classify(instances: Iterable[Vector]): Array[Double]

  /**
   * Given a list of parameters, update the attributes of the classifier.
   *
   * @param  args  List of parameters.
   */
  def setParameters(args: Array[String]): Unit

}
