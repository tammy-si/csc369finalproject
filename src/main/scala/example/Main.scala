package example

import org.apache.spark.sql.SparkSession
import org.apache.spark.SparkContext._
import scala.io._
import org.apache.spark.{ SparkConf, SparkContext }
import org.apache.spark.rdd._
import org.apache.log4j.Logger
import org.apache.log4j.Level
import scala.collection._
import org.apache.spark.rdd.RDD

object Main {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("FinalProject")
      .master("local[*]")
      .getOrCreate()
    val sc = spark.sparkContext
    val inputPath = args(0)
    val outputPath = args(1)
    val rawdiabetesDF = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(inputPath)
    rawdiabetesDF.printSchema()
    val allFeatures =  Seq(
      "GenHlth", "HighBP", "BMI", "HighChol", "Age",
      "HeartDiseaseorAttack", "Educatoin", "MentHlth", "Smoker", "Sex", "Veggies"
    )
    // Get feature indices
    val featureCols = rawdiabetesDF.columns.filter(_ != "Diabetes")
    val selectedFeatures = allFeatures.map(name => featureCols.indexOf(name))
    val filteredFeatures = selectedFeatures.filter(_ >= 0)


    // Map RDD
    val fullData = rawdiabetesDF.rdd.map { row =>
      val features = featureCols.map(col => row.getAs[Any](col).toString.toDouble)
      val label = row.getAs[Any]("Diabetes").toString.toDouble.toInt
      (features, label)
    }.cache()

    val (train, test) = trainTestSplit(fullData, 0.8)
    val model = trainNaiveBayes(train, filteredFeatures)
    val acc = accuracy(test, filteredFeatures , model)
    sc.parallelize(Seq(f"Model accuracy: ${acc * 100}%.2f%%"))
      .coalesce(1)
      .saveAsTextFile(outputPath)


    /* performance stuff */
    val classes1 = test.map(_._2).distinct().collect().sorted

    val classMetrics = classes1.map { c =>
      val counts = test.map { case (features, label) =>
          val pred = predict(features, filteredFeatures, model)
          if (label == c && pred == c) ("TP", 1)
          else if (label != c && pred == c) ("FP", 1)
          else if (label == c && pred != c) ("FN", 1)
          else ("TN", 1)
        }.reduceByKey((a, b) => a + b)
        .collectAsMap()

      val TP = counts.getOrElse("TP", 0).toDouble
      val FP = counts.getOrElse("FP", 0).toDouble
      val FN = counts.getOrElse("FN", 0).toDouble

      val precision = if (TP + FP > 0) TP / (TP + FP) else 0.0
      val recall = if (TP + FN > 0) TP / (TP + FN) else 0.0
      val f1 = if (precision + recall > 0) 2 * (precision * recall) / (precision + recall) else 0.0

      (c, precision, recall, f1)
    }

    val metricsOutput = collection.mutable.ArrayBuffer[String]()

    //println(f"\nPer-Class Metrics:")
    metricsOutput += "\nPer-Class Metrics:"

    classMetrics.foreach { case (c, prec, rec, f1) =>
      metricsOutput += f"Class $c: Precision = $prec%.4f, Recall = $rec%.4f, F1 = $f1%.4f"
    }
    //precision, recall, f1
    //and then confusion matrix (3x3)

    val confusionPairs = test.map { case (features, label) =>
      val prediction = predict(features, filteredFeatures, model)
      ((label, prediction), 1)
    }

    val confusionMatrix = confusionPairs
      .reduceByKey((a, b) => a + b)
      .collect()
      .toMap

    val classes = test.map(_._2).distinct().collect().sorted

    metricsOutput += "\nConfusion Matrix:"
    metricsOutput += f"             ${classes.map(c => f"Pred $c%8s").mkString}"

    //println("\nConfusion Matrix:")
    metricsOutput += f"             ${classes.map(c => f"Pred $c%8s").mkString}"
    for (actual <- classes) {
      val row = new StringBuilder(f"\nActual $actual%5d")

      //print(f"\nActual $actual%5d")
      for (predicted <- classes) {
        val count = confusionMatrix.getOrElse((actual, predicted), 0)
        row.append(f"$count%12d")
      }
      metricsOutput += row.toString()
    }
    //println()
    sc.parallelize(metricsOutput)
      .coalesce(1)
      .saveAsTextFile(outputPath + "/metrics")
  }

  def trainTestSplit[T](data: RDD[T], trainFraction: Double = 0.8): (RDD[T], RDD[T]) = {
    val splits = data.randomSplit(Array(trainFraction, 1 - trainFraction), seed = 42)
    (splits(0), splits(1))
  }

  def gaussianLogProb(x: Double, mean: Double, variance: Double): Double = {
    val eps = 1e-6
    val varAdj = if (variance < eps) eps else variance
    -0.5 * math.log(2 * math.Pi * varAdj) - math.pow(x - mean, 2) / (2 * varAdj)
  }

  def trainNaiveBayes(
                       train: RDD[(Array[Double], Int)],
                       featureIndices: Seq[Int]
                     ): (Map[Int, Map[Int, (Double, Double)]], Map[Int, Double]) = {
    val classGrouped = train.groupBy(_._2)
    val featureStats = classGrouped.mapValues { samples =>
      val points = samples.map(_._1)
      featureIndices.map { i =>
        val values = points.map(_(i)).toArray
        val mean = values.sum / values.length
        val variance = values.map(v => math.pow(v - mean, 2)).sum / values.length
        (mean, variance)
      }.zipWithIndex.map { case ((mean, variance), i) => (i, (mean, variance)) }.toMap
    }.collect().toMap
    val classCounts = train.map { case (_, label) => (label, 1) }.reduceByKey(_ + _).collectAsMap()
    val total = classCounts.values.sum.toDouble
    val classPriors = classCounts.map { case (label, count) => (label, math.log(count / total)) }.toMap
    (featureStats, classPriors)
  }

  def predict(
               features: Array[Double],
               featureIndices: Seq[Int],
               model: (Map[Int, Map[Int, (Double, Double)]], Map[Int, Double])
             ): Int = {
    val (featureStats, classPriors) = model
    classPriors.map { case (label, priorLogProb) =>
      val likelihood = featureIndices.indices.map { i =>
        val (mean, variance) = featureStats(label)(i)
        gaussianLogProb(features(featureIndices(i)), mean, variance)
      }.sum
      (label, priorLogProb + likelihood)
    }.maxBy(_._2)._1
  }

  def accuracy(
                data: RDD[(Array[Double], Int)],
                featureIndices: Seq[Int],
                model: (Map[Int, Map[Int, (Double, Double)]], Map[Int, Double])
              ): Double = {
    val correct = data.map { case (features, label) =>
      val prediction = predict(features, featureIndices, model)
      if (prediction == label) 1 else 0
    }.sum()
    correct / data.count().toDouble
  }
}
