# SmartReduction

This framework implements four distance based Big Data preprocessing algorithms for prototype selection and generation: FCNN_MR, SSMASFLSDE_MR, RMHC_MR, MR_DIS, with special emphasis in their scalability and performance traits.

## Example (FCNN_MR)


```scala
import org.apache.spark.mllib.feature._

// Data must be cached in order to improve the performance

val fcnn_mr_model = new FCNN_MR(trainingData, // RDD[LabeledPoint]
                              k) // number of neighbors

val fcnn_mr = fcnn_mr_model.runFilter()
```
## Example (MR_DIS)


```scala
import org.apache.spark.mllib.feature._

// Data must be cached in order to improve the performance

val mr_dis_model = new DemoIS(k,  // number of neighbors
                              partitions) // number of partitions
                              .instSelection(trainingData) // RDD[LabeledPoint]

val mr_dis = mr_dis_model.runFilter()
```

## Example (SSMASFLSDE_MR)


```scala
import org.apache.spark.mllib.feature._

// Data must be cached in order to improve the performance

val ssmasflsde_mr_model = new SSMASFLSDE_MR(trainingData) // RDD[LabeledPoint]

val ssmasflsde_mr = ssmasflsde_mr_model.runFilter()
```

## Example (RMHC_MR)


```scala
import org.apache.spark.mllib.feature._

// Data must be cached in order to improve the performance

val rmhc_mr_model = new RMHC_MR(trainingData, // RDD[LabeledPoint]
                              p, // Percentage of instances (max 1.0)
                              iterations, // Number of iterations
                              seed)

val rmhc_mr = rmhc_mr_model.runFilter()
```
