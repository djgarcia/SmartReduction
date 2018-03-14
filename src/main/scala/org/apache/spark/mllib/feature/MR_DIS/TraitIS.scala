package org.apache.spark.mllib.feature.MR_DIS

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

/**
 * Defines all the methods that an instance selection algorithm must implement.
 *
 * @author Alejandro González Rogel
 * @version 1.0.0
 */
trait TraitIS {

  /**
   *
   * Given an initial dataset, return a similar (in the worst case scenario) or smaller
   * dataset obtained by applying an instance selection algorithm.
   *
   * //@param  sc  Spark context of the application.
   * @param  data  Initial dataset.
   * @return  Remaining instances after the algorithm has been executed.
   */
  def instSelection(data: RDD[LabeledPoint]): RDD[LabeledPoint]

  /**
   * Given a list of parameters, this method reads them and uses them to modify the
   * default values of the algorithm.
   *
   * @param  args  Arguments of the algorithm.
   * @throws IllegalArgumentException If any of the parameters is not correct.
   */
  @throws(classOf[IllegalArgumentException])
  def setParameters(args: Array[String]): Unit

  /**
   * Given a parameter-value pair, assigns the mentioned value to the class attribute
   * the parameter refers to.
   *
   * @param  identifier Attribute id.
   * @param  value  Value
   *
   * @throws IllegalArgumentException If any of the input values is not correct:
   * The id does not exist or the type of the value is not the expected one.
   */
  @throws(classOf[IllegalArgumentException])
  protected def assignValToParam(identifier: String, value: String): Unit

}
