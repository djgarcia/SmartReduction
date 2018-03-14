package org.apache.spark.mllib.feature.MR_DIS

import org.apache.spark.mllib.regression.LabeledPoint

/**
 * Trait that defines all the necessary methods for this library to run the algorith.
 *
 * @author Alejandro González Rogel
 * @version 1.0.0
 */

@SerialVersionUID(1L)
trait TraitSeqIS extends Serializable {

  /**
   *
   * Given a dataset, it return an smaller or equal dataset resulting of the application
   * of an instance selection algorithm.
   * 
   * @param  data  Original dataset.
   * @return  Final dataset.
   *
   * @throws IllegalArgumentException If any of the parameters given to the
   * algorithm is not correct.
   */
  @throws(classOf[IllegalArgumentException])
  def instSelection(data: Iterable[LabeledPoint]): Iterable[LabeledPoint]

  /**
   * Given a list of parameters, it analyses them and updates the class
   * attribute values.
   *
   * @param  args  New arguments for the algorithm.
   */
  def setParameters(args: Array[String]): Unit

}
