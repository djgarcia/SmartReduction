package org.apache.spark.mllib.feature

import org.apache.spark.mllib.classification.kNN_IS.kNN_IS
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

class RMHC_MR(val data: RDD[LabeledPoint], val m: Double, val p: Int, val k: Int, val seed: Int) extends Serializable {

  def runFilter(): RDD[LabeledPoint] = {

    val size = data.count()
    var S = data.sample(withReplacement = false, m, seed)
    var A, A_sk = 0.0
    val numClass = data.map(_.label).distinct().collect().length
    val numFeatures = data.first().features.size

    for (i <- 0 until p) {
      val S_sk = S
      val mutation = mutate(S, S_sk)

      val knn = kNN_IS.setup(S_sk, mutation, k, 2, numClass, numFeatures, S_sk.partitions.length, 64, -1, 1)
      val predictions = knn.predict(data.context)
      val metrics = new MulticlassMetrics(predictions)
      A_sk = metrics.precision

      if (A_sk > A) {
        S = mutation
        A = A_sk
      }
    }
    S
  }

  private def mutate(S: RDD[LabeledPoint], S_sk: RDD[LabeledPoint]): RDD[LabeledPoint] = {
    val r = scala.util.Random
    val instance = data.subtract(S).takeSample(withReplacement = false, 1, seed)(0)
    val pos = r.nextInt(S_sk.count().toInt)
    S_sk.zipWithIndex.map { case (v, k) =>
      if (k == pos) {
        instance
      } else {
        v
      }
    }
  }
}
