package org.apache.spark.mllib.feature

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

class SSMASFLSDE_MR(val data: RDD[LabeledPoint], val header: String) extends Serializable {

  def runPR(): RDD[LabeledPoint] = {

    data.mapPartitions { partition =>

      if (partition.isEmpty) {
        partition
      } else {
        val data = partition.toArray

        val dataAsArray = data.map(_.features.toArray)
        val classes = data.map(_.label.toInt)
        val ssma = new Keel.SSMASFLSDE(dataAsArray, classes)

        val ssmaData: Array[LabeledPoint] = ssma.ejecutar()

        ssmaData.toIterator
      }
    }
  }
}
