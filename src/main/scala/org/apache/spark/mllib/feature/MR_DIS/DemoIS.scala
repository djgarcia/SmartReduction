package org.apache.spark.mllib.feature.MR_DIS

import java.io.{File, FileWriter, IOException, PrintWriter}
import java.text.SimpleDateFormat
import java.util.Calendar
import java.util.logging.{Level, Logger}

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.rdd.RDD.rddToPairRDDFunctions

import scala.collection.mutable.{ListBuffer, MutableList}

/**
 * Democratic Instance Selection algorithm.
 *
 * Democratic Instance Selection is an instance selection algorithm that 
 * applies, for several rounds and over disjoint subsets of the original dataset,
 * simpler instance selection algorithms. Once this phase is over, is uses the obtained
 * results to decide which instances should be eliminated.
 *
 * García-Osorio, César, Aida de Haro-García, and Nicolás García-Pedrajas.
 * "Democratic instance selection: a linear complexity instance selection
 * algorithm based on classifier ensemble concepts." Artificial Intelligence
 * 174.5 (2010): 410-441.
 *
 * @constructor Create a new instance selector with by default attribute values.
 *
 * @author  Alejandro González Rogel
 * @version 1.0.0
 */
class DemoIS(val knn: Int = 1, val partitions: Int = 10, val perc: Int = 10) extends TraitIS {

  /**
   * Path to the log strings.
   */
  private val bundleName = "resources.loggerStrings.stringsDemoIS"
  /**
   * Logger.
   */
  //private val logger = Logger.getLogger(this.getClass.getName(), bundleName)

  /**
   * Number of voting phases.
   */
  var numRepeticiones = 10

  /**
   *  Weight for the accuracy when computing the fitness function.
   */
  var alpha = 0.75

  /**
   *  Number of partitions on which the dataset will be divided after we performed
   * the voting.
   */
  var numPartitions = partitions

  /**
   * Number of nearest neighbours to use in the kNN.
   */
  var k = knn

  /**
   * Number of partitions in which the training set will be divided for the kNN.
   */
  var numPartitionsKNN = partitions

  /**
   * Number of times we apply a reduction in the kNN.
   */
  var numReducesKNN = 16

  /**
   * Number of partitions in which the test set will be divided during the kNN.
   */
  var numReducePartKNN = 16 // -1 auto-setting

  /**
   * Maximum weight of each partition in the test set.
   */
  var maxWeightKNN = 0.0

  /**
   * Seed for random number generation.
   */
  var seed: Long = 1

  /**
   * Subset of the original dataset that will be used to calculate the fitness value
   *
   * It is a percentage.
   *
   */
  var datasetPerc = perc.toDouble

  /**
   * Distance measurement formula.
   *
   * MANHATTAN = 1 ; EUCLIDEAN = 2 ; HVDM = 3
   * TODO Actualmente no cuenta con utilidad
   */
  var distType = 2

  /**
   * Seed for the kNN.
   */
  var sknn: Long = 1

  /**
   * If extra information about the execution should be stored.
   */
  var xtraInfo = false

  override def instSelection(
    originalData: RDD[LabeledPoint]): RDD[LabeledPoint] = {

    // Add a key to every instance.
    val dataAndVotes = originalData.map(inst => (0, inst))

    // Map operation - voting
    val ratedData = doVoting(dataAndVotes)

    // Reduce operation - Get fitness value and select instances.
    val (criterions, extraData) = lookForBestCriterion(ratedData)
    val indBestCrit = criterions.indexOf(criterions.min) + 1

    // Store information
    if (xtraInfo) {
      writeExtraInfo(criterions, extraData)
    }
    ratedData.filter(
      tuple => tuple._1 < indBestCrit).map(tuple => tuple._2)

  }

  /**
   * It will, for a predefined number of times, distribute the original dataset
   * between the nodes and apply an instance selection algorithm individually for
   * each partition, updating the number of votes of the non-selected instances.
   *
   * @param RDDconContador Original dataset where entry is a tuple with the instance
   *   and the number of votes that such instance has received so far.
   *
   * @return Tuples with an instance and the total number of votes that it has received.
   *
   * @todo Allow the user to choose which algorithm he wants to apply in the nodes.
   */
  private def doVoting(
    RDDconContador: RDD[(Int, LabeledPoint)]): RDD[(Int, LabeledPoint)] = {

    // Instance selector that will be used for computing the votes.
    val seqAlgorithm: TraitSeqIS = new CNN()

    // Instructions to apply individually on each partition.
    val votingInNodes = new VotingInNodes()
    
    var actualRDD = RDDconContador.zipWithIndex().map(line => (line._2, line._1))

    // Partitioner that will distribute the instances.
    //val partitioner = new RandomPartitioner(numPartitions, actualRDD.count(), seed)

    for { i <- 0 until numRepeticiones } {

      // Redistribute the instances between the nodes.
      //actualRDD = actualRDD.partitionBy(partitioner)
      // Apply the voting.
      actualRDD = actualRDD.mapPartitions(instancesIterator =>
        votingInNodes
          .applyIterationPerPartition(instancesIterator, seqAlgorithm)).persist

      //partitioner.rep += 1

      // TODO It was necessary to force the execution up until this point.
      actualRDD.foreachPartition(x => None)

    }
    actualRDD.persist()
    actualRDD.name = "InstancesAndVotes"
    actualRDD.map(tupla => tupla._2)

  }

  /**
   * Gets the fitness value for each one of the possible reduction values and chooses
   * the one with the lowest value.
   *
   * @param  RDDconContador Dataset with tuples instance-votes
   *
   * @return Tuple with the number of votes that produces the lowest fitness and the
   * value of such fitness.
   *
   */
  private def lookForBestCriterion(
    RDDconContador: RDD[(Int, LabeledPoint)]): (MutableList[Double], MutableList[Array[Double]]) = {

    // Donde almacenar los valores del criterion de cada número de votos.
    var criterions = new MutableList[Double]
    // Donde almacenar los valores de parámetros en cada iteración en busca
    // del mejor criterio.
    var extraData = new MutableList[Array[Double]]
    // Creamos un conjunto de test.
    val testRDD =
      RDDconContador.sample(false, datasetPerc / 100, seed).map(tupla => tupla._2)

    val originalDatasetSize = RDDconContador.count()

    for { i <- 1 to numRepeticiones } {

      // Select all the instances which votes are above the partial threshold.
      // TODO I’m not sure if this persist() is really necessary.
      val selectedInst = RDDconContador.filter(
        tupla => tupla._1 < i).map(tupla => tupla._2).persist()

      if (selectedInst.isEmpty) {
        criterions += Double.MaxValue
      } else {
        var tmp = calcCriterion(selectedInst, testRDD,
          originalDatasetSize)
        criterions += tmp._1
        extraData += tmp._2
      }
    }

    (criterions, extraData)
  }

  /**
   * Calculates the fitness value.
   * 
   * It uses the formula: alpha * testError + (1 - alpha) * subDatasize
   *
   * @param  selectedInst  Subset of instances above the threshold value.
   * @param  testRDD  Test set for the classifier.
   * @param  dataSetSize  Size of the original dataset.
   *
   * @return Fitness value and information about the other variables values.
   *
   */
  private def calcCriterion(selectedInst: RDD[LabeledPoint],
                            testRDD: RDD[LabeledPoint],
                            dataSetSize: Long): (Double, Array[Double]) = {

    // Get the reduction percentage of the subset with respect to the original dataset.
    val subSetSize = selectedInst.count.toDouble
    val subSetPerc = subSetSize / dataSetSize


    val knn = new KNN()
    // TODO Hardcoded
    val knnParameters: ListBuffer[String] = new ListBuffer[String]
    knnParameters += "-k"
    knnParameters += k.toString()
    knnParameters += "-np"
    knnParameters += numPartitionsKNN.toString()
    knnParameters += "-nr"
    knnParameters += numReducesKNN.toString()
    knnParameters += "-nrp"
    knnParameters += numReducePartKNN.toString()
    knnParameters += "-maxW"
    knnParameters += maxWeightKNN.toString()
    knnParameters += "-dt"
    knnParameters += distType.toString()
    knnParameters += "-s"
    knnParameters += sknn.toString()
    knn.setParameters(knnParameters.toArray)
    knn.train(selectedInst)
    val tmp = testRDD.zipWithIndex().map(line => (line._2, line._1)).persist
    val testFeatures = tmp.map(tuple => (tuple._1, tuple._2.features))
    val testClasses = tmp.map(tuple => (tuple._1, tuple._2.label))
    val classResults = knn.classify(testFeatures)

    val failures = classResults.join(testClasses).filter(tuple => tuple._2._1 != tuple._2._2).count()

    val testError = failures.toDouble / testRDD.count

    // Extra information about the execution.
    // It will be stored if the user asked for it.
    var info = Array(dataSetSize, subSetSize, subSetPerc, failures, testError)

    // Calculate fitness function.
    (alpha * testError + (1 - alpha) * subSetPerc, info)

  }

  /**
   * Stores in a .csv file all the information about the execution.
   *
   * This information contains the different values of all the important
   * parameters that have been used during the computation of the fitness values.
   *
   *
   * @param  criterion  List with all the computed fitness values.
   * @param  extraData  Values for the different attributes that contributed to this
   *    calculation. These attributes are:
   *             totalDataSize, subDataSize, Reduction, failures, testError,
   *             finalCriterion
   */
  private def writeExtraInfo(criterion: MutableList[Double],
                             extraData: MutableList[Array[Double]]): Unit = {

    val fileSeparator = System.getProperty("file.separator")

    val resultPath = "results" + fileSeparator + "extra_info"

    val myDateFormat = new SimpleDateFormat("dd-MM-yyyy_HH-mm-ss");

    val now = Calendar.getInstance().getTime()

    val fileName = resultPath + fileSeparator + "info_criterion_demoIS" +
      "_" + myDateFormat.format(now) + ".csv"

    val resultDir = new File(resultPath)
    if (!resultDir.exists()) {
      resultDir.mkdir()
    }

    val file = new File(fileName)

    val writer = new FileWriter(file)
    try {
      writer.write("n (#),totalDataSize (#),subDataSize (#),Reduction (%)," +
        "failures (#),testError (%), finalCriterion\n")
      for { i <- 0 until extraData.size } {
        writer.write(i.toString() + ',' + extraData.get(i).get(0) + "," +
          extraData.get(i).get(1) + "," + (100 - extraData.get(i).get(2) * 100) +
          "," + extraData.get(i).get(3) + "," + extraData.get(i).get(4) + "," +
          criterion(i) + "\n")
      }
      writer.write("Best criterion:" + "," + criterion.indexOf(criterion.min) +
        "," + criterion.min + "\n")
    } catch {
      case e: IOException => None
      // TODO Add error message.
    } finally {
      writer.close()
    }

  }

  override def setParameters(args: Array[String]): Unit = {

    // Check for the correct number of arguments.
    checkIfCorrectNumber(args.size)

    for { i <- 0 until args.size by 2 } {
      try {
        val identifier = args(i)
        val value = args(i + 1)
        assignValToParam(identifier, value)
      } catch {
        case ex: NumberFormatException =>
          //logger.log(Level.SEVERE, "DemoISNoNumberError", args(i + 1))
          throw new IllegalArgumentException()
      }
    }

    if (numRepeticiones <= 0 || alpha < 0 || alpha > 1 || k <= 0 ||
      numPartitions <= 0 || datasetPerc <= 0 || datasetPerc >= 100
      || distType < 1 || distType > 3) {
      //logger.log(Level.SEVERE, "DemoISWrongArgsValuesError")
      //logger.log(Level.SEVERE, "DemoISPossibleArgs")
      throw new IllegalArgumentException()
    }

  }

  protected override def assignValToParam(identifier: String,
                                          value: String): Unit = {
    identifier match {
      case "-rep"     => numRepeticiones = value.toInt
      case "-alpha"   => alpha = value.toDouble
      case "-s"       => seed = value.toInt
      case "-k"       => k = value.toInt
      case "-np"      => numPartitions = value.toInt
      case "-dsperc"  => datasetPerc = value.toDouble
      case "-npknn"   => numPartitionsKNN = value.toInt
      case "-nrknn"   => numReducesKNN = value.toInt
      case "-nrpknn"  => numReducePartKNN = value.toInt
      case "-maxWknn" => maxWeightKNN = value.toDouble
      case "-d"       => distType = value.toInt
      case "-sknn"    => sknn = value.toLong
      case "-ei"      => if (value.toLong > 0) xtraInfo = true
      case somethingElse: Any =>
        //logger.log(Level.SEVERE, "DemoISWrongArgsError",
          //somethingElse.toString())
       // logger.log(Level.SEVERE, "DemoISPossibleArgs")
        throw new IllegalArgumentException()
    }
  }

  /**
   * Check if the number of parameters that has been introduced is correct.
   *
   * @param argsNumber Number of parameters.
   *
   * @throws IllegalArgumentException If the number of parameters is not the
   * expected one.
   */
  @throws(classOf[IllegalArgumentException])
  def checkIfCorrectNumber(argsNumber: Int): Unit = {

    if (argsNumber % 2 != 0) {
      //logger.log(Level.SEVERE, "DemoISPairNumberParamError",
        //this.getClass.getName)
      throw new IllegalArgumentException()
    }

  }

}
