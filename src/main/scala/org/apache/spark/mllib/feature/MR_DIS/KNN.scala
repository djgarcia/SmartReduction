package org.apache.spark.mllib.feature.MR_DIS

import java.io.PrintWriter
import java.util.Random
import java.util.logging.{Level, Logger}

import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.rdd.RDD.rddToPairRDDFunctions

/**
 * Parallel kNN classifier.
 *
 * This classifier assigns a class to an instance by checking the most common class
 * between the K closest points of the training dataset.
 * 
 * This implementation is based on the work made by:
 *
 * Jesús Maillo, Isaac Triguero and Francisco Herrera. "Un enfoque MapReduce del algoritmo
 * k-vecinos más cercanos para Big Data" In Actas de la XVI Edición Conferencia de la
 * Asociación Española para la Inteligencia Artificial CAEPIA 2015
 * Online repository: https://github.com/JMailloH/kNN_IS
 *
 */

class KNN extends TraitClassifier {

  /**
   * Path to the logger strings.
   */
  private val bundleName = "resources.loggerStrings.stringsKNN";
  /**
   * Logger.
   */
  //private val logger = Logger.getLogger(this.getClass.getName(), bundleName);

  /**
   * Number of nearest neighbours.
   */
  var k = 1

  /**
   * Training set.
   */
  var trainingData: RDD[LabeledPoint] = null

  /**
   * Number of partitions on which the training set will be divided.
   */
  var numPartitions = 2

  /**
   * Number of partition on which we will perform the reduce operation.
   */
  var numReduces = 1

  /**
   * Number of partitions on which we will divide the test set.
   *
   * -1 for auto-setting of this parameter according to the maximum
   * allowed weight of each partition.
   * 
   * TODO "auto-setting" mode is not implemented.
   */
  var numReducePartitions = 1

  /**
   * Maximum allowed weight of each partition the test set.
   *
   * It is only used if ‘numReducePartitions’ is set to ‘auto-setting’.
   *
   * TODO Not implemented.
   */
  var maxWeight = 0.0

  /**
   * Algorithm to compute distance between instances.
   * 
   * MANHATTAN = 1 ; EUCLIDEAN = 2 ; HVDM = 3
   *
   * TODO Not implemented. Distance is always euclidean.
   */
  var distType = 2

  /**
   * Seed.
   */
  var seed:Long = 1

  /**
   * Spark context.
   */
  private var sc: SparkContext = null

  override def train(trainingSet: RDD[LabeledPoint]): Unit = {
  // TODO Check how good this method is when distributing the instances.
    
    if (numPartitions != trainingSet.getNumPartitions) {

      var tmpRDD = trainingSet.mapPartitionsWithIndex((i, iter) => {
        var tmp = iter.toArray;
        var tmp2 = new Array[(Long, LabeledPoint)](tmp.size)
        var r = new Random(i)
        for (j <- 0 until tmp.size) {
          tmp2.update(j, (r.nextInt, tmp(j)))
        }
        tmp2.toIterator
      })

      val partitioner = new RandomPartitioner(numPartitions, tmpRDD.count(), 1)
      trainingData = tmpRDD.partitionBy(partitioner).map(tupla => tupla._2)
    } else {
      trainingData = trainingSet
    }
    trainingData.persist
    sc = trainingSet.sparkContext
  }

  override def classify(instances: RDD[(Long, Vector)]): RDD[(Long, Double)] = {

    // Count the samples of each data set and the number of classes
    val numSamplesTrain = trainingData.count()
    instances.persist
    val numSamplesTest = instances.count()

    // numSamplesTest per iteration
    var numIterations = numReducePartitions
    // Número de instancias de test por cada iteración.
    var inc = (numSamplesTest / numIterations).toInt

    // top and sub limits.
    var subdel = 0
    var topdel = inc - 1
    if (numIterations == 1) { // If only one partition
      topdel = numSamplesTest.toInt + 1
    }

    // TODO Get rid of all these null values.
    var test: Broadcast[Array[Vector]] = null
    var result: RDD[(Long, Double)] = null
    var rightPredictedClasses: Array[Array[Array[Int]]] =
      new Array[Array[Array[Int]]](numIterations)

    val knnInNodes = new KNNInNodes();
    knnInNodes.k = k
    knnInNodes.distType = distType

    for (i <- 0 to numIterations - 1) {

      if (i == numIterations - 1) {
       // This ‘*2’ multiplication allows us to select everything that is left during
       // the final iteration.
        test = broadcastTest(instances.filterByRange(subdel, topdel * 2).map(line => line._2).collect)

      } else {
        test = broadcastTest(instances.filterByRange(subdel, topdel).map(line => line._2).collect)
      }

      // Calling KNN (Map Phase)  

      knnInNodes.subdel = subdel
      var resultKNNPartitioned =
        trainingData.mapPartitions(part => knnInNodes.knn(part, test))

      // Reduce phase
      var partResult =
        resultKNNPartitioned.reduceByKey(knnInNodes.combine(_, _), numReduces)

      if (result == null) {
        result = partResult.map(tupla => knnInNodes.calcPredictedClass(tupla))
      } else {
        val tmp = partResult.map(tupla => knnInNodes.calcPredictedClass(tupla))
        result = result.union(tmp)

      }

      // TODO This count here forces the execution up until this point.
      // Otherwise, the task manager gets confuse during the ‘knnInNodes.subdel = subdel’
      // Look for a solution.
      result.count

      subdel = subdel + inc
      topdel = topdel + inc
      // TODO
      // This was a ‘destroy’
      test.unpersist

    }
    result.repartition(numReduces)
  }

  override def setParameters(args: Array[String]): Unit = {

    // Check that we received the correct number of attributes.
    if (args.size % 2 != 0) {
      //logger.log(Level.SEVERE, "KNNPairNumberParamError",
        //this.getClass.getName)
      throw new IllegalArgumentException()
    }

    for { i <- 0 until args.size by 2 } {
      try {
        args(i) match {
          case "-k"    => k = args(i + 1).toInt
          case "-np"   => numPartitions = args(i + 1).toInt
          case "-nr"   => numReduces = args(i + 1).toInt
          case "-nrp"  => numReducePartitions = args(i + 1).toInt
          case "-maxW" => maxWeight = args(i + 1).toDouble
          case "-dt"   => distType = args(i + 1).toInt
          case "-s"    => seed = args(i + 1).toLong
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

    // If any of the variables has an incorrect value.
    if (k <= 0 || numPartitions <= 0 || numReduces <= 0 || numReducePartitions <= -1
      || maxWeight < 0 || distType < 1 || distType > 3) {
      //logger.log(Level.SEVERE, "KNNWrongArgsValuesError")
      //logger.log(Level.SEVERE, "KNNPossibleArgs")
      throw new IllegalArgumentException()
    }

  }

  /**
   * Creates a broadcast variable so it can distribute a dataset among all the
   * working ndoes.
   *
   * @param data    Dataset
   * @param context Spark context
   *
   */
  private def broadcastTest(data: Array[Vector]) = sc.broadcast(data)

}