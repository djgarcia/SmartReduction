package org.apache.spark.mllib.feature.MR_DIS

import java.util.Random

import org.apache.spark.Partitioner

/**
 * Random partitioner for the objects that form a [[org.apache.spark.rdd.RDD]].
 *
 * The generated partitions might not have the same number of objects, but the
 * size of all of them should be similar.
 
 * @constructor Creates a new random partitioner.
 * @param  numPartitions  Number of partitions in which the dataset will be divided.
 * @param  totalInst Total number of instances of our dataset.
 *
 * @author Alejandro González Rogel
 * @version 1.0.0
 */
class RandomPartitioner(val numPartitions: Int, val totalInst:Long, val seed:Long) extends Partitioner {

  /**
   * Current repetition.
   */
  var rep = 0

  /**
   * Gives a random partition to a given instance.
   *
   * @param key Key of the instance to distribute.
   * @return Id of the selected partition.
   */
  def getPartition(key: Any): Int = {
    val k = key.asInstanceOf[Long]
    val finalSeed = seed + k + (rep*totalInst)
    val r = new Random(finalSeed)
    r.nextInt(numPartitions)
  }
}

