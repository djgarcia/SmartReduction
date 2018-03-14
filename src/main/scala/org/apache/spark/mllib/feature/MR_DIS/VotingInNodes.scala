package org.apache.spark.mllib.feature.MR_DIS

import org.apache.spark.mllib.regression.LabeledPoint

import scala.collection.mutable.MutableList

/**
 * Intructions involved in the voting process in each individual node.
 *
 * Instances will receive a vote if they have not been selected by the
 * instance selector algorithm.
 *
 */
@SerialVersionUID(1L)
private class VotingInNodes extends Serializable {

  /**
   * Voting algorithm.
   *
   * It applies an instance selection algorithm to the given dataset. Then, all
   * those instances that were not selected by the instance selector will receive a vote.
   *
   * @param  instancesIterator  Iterator over the initial dataset. Each element is a
   *   tuple instance - number of votes.
   *   (número de votos).
   * @param  linearIS  Sequential instance selection algorithm.
   * @return Iterator over the dataset after the votes has been updated. Each element is a
   *   tuple instance - number of votes.
   *
   */
  def applyIterationPerPartition(
    instancesIterator: Iterator[(Long,(Int, LabeledPoint))],
    linearIS: TraitSeqIS): Iterator[(Long,(Int, LabeledPoint))] = {

    // Store all the values in lists.
    var instancias = new MutableList[LabeledPoint]
    var myIterableCopy = new MutableList[(Long,(Int, LabeledPoint))]
    while (instancesIterator.hasNext) {
      var tmp = instancesIterator.next
      myIterableCopy += tmp
      instancias += tmp._2._2
    }

    // Run instance selection algorithm.
    var selected = linearIS.instSelection(instancias)

    // Update the voting count.
    var iter = myIterableCopy.iterator
    var actIndex = -1
    while (iter.hasNext) {
      actIndex += 1
      var actualInst = iter.next._2._2
      if (!selected.exists { inst => inst.eq(actualInst) }) {
        myIterableCopy.update(actIndex,
          (myIterableCopy.get(actIndex).get._1,(myIterableCopy.get(actIndex).get._2._1 + 1,
            myIterableCopy.get(actIndex).get._2._2)))
      }
    }
    myIterableCopy.iterator
  }

}
