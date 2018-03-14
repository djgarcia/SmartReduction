package org.apache.spark.mllib.feature.MR_DIS

import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

/**
 * Defines the basic methods that every classifier must have to be part of this library.
 *
 * @author Alejandro González Rogel
 * @version 1.0.0
 */
trait TraitClassifier {

  /**
   * Trains the classifier in order to tune its parameters.
   *
   * @param  trainingSet  Data set.
   */
  def train(trainingSet: RDD[LabeledPoint]): Unit


  /**
   * Classifies any given instance 
   *
   * @param instances  Unique ID of the instances we want to classify.
   *
   * @return Predicted classes.
   */
  def classify(instances: RDD[(Long,Vector)]): RDD[(Long,Double)]

  /**
   * Given a list of parameters, this method analyses it and use it to change the
   * parameters of the classifier.
   *
   * @param  args  List of strings, each one representing a value or a class attribute
   *   identifier.
   */
  def setParameters(args: Array[String]): Unit

}
