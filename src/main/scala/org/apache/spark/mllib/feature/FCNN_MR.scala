package org.apache.spark.mllib.feature

import org.apache.spark.mllib.feature.Keel.FCNN
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

class FCNN_MR(val data: RDD[LabeledPoint], val k: Int = 1) extends Serializable {

  def runPR(): RDD[LabeledPoint] = {

    data.mapPartitions { partition =>

      val data = partition.toArray

      val dataAsArray = data.map(_.features.toArray)
      val classes = data.map(_.label.toInt)
      val fcnn = new FCNN(dataAsArray, classes, k)

      val fcnnData: Array[LabeledPoint] = fcnn.ejecutar()

      fcnnData.toIterator
    }
  }
}
