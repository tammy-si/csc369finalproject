package example

import org.apache.spark.sql.SparkSession
import org.apache.spark.SparkContext._

import scala.io._
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd._
import org.apache.log4j.Logger
import org.apache.log4j.Level

import scala.collection._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{avg, col, round, desc, first}

object demographics {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)
    Logger.getRootLogger.setLevel(Level.ERROR)

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

    val healthRates = rawdiabetesDF
      .withColumn("DiabetesClass", col("Diabetes").cast("int"))
      .groupBy("DiabetesClass")
      .agg(
        round(avg(col("Smoker")) * 100, 1).alias("Smoker_Percent"),
        round(avg(col("Fruits")) * 100, 1).alias("Fruits_Percent"),
        round(avg(col("PhysActivity")) * 100, 1).alias("PhysActivity_Percent"),
        round(avg(col("HighChol")) * 100, 1).alias("HighChol_Percent"),
        round(avg(col("HighBP")) * 100, 1).alias("HighBP_Percent")
      )
      .orderBy("DiabetesClass")
    healthRates.show()
    // mode code import org.apache.spark.sql.functions._

    val ageModes = rawdiabetesDF
      .withColumn("DiabetesClass", col("Diabetes").cast("int"))
      .groupBy("DiabetesClass", "Age")
      .count()
      .orderBy(col("DiabetesClass"), desc("count"))
      .groupBy("DiabetesClass")
      .agg(first("Age").alias("Most_Common_Age_Group"))

    ageModes.show()

    val incomeModes = rawdiabetesDF
      .withColumn("DiabetesClass", col("Diabetes").cast("int"))
      .groupBy("DiabetesClass", "Income")
      .count()
      .orderBy(col("DiabetesClass"), desc("count"))
      .groupBy("DiabetesClass")
      .agg(first("Income").alias("Most_Common_Income_Group"))

    incomeModes.show()

    val educationModes = rawdiabetesDF
      .withColumn("DiabetesClass", col("Diabetes").cast("int"))
      .groupBy("DiabetesClass", "Education")
      .count()
      .orderBy(col("DiabetesClass"), desc("count"))
      .groupBy("DiabetesClass")
      .agg(first("Education").alias("Most_Common_Education_Level"))

    educationModes.show()

    val sexAndHealthcareRates = rawdiabetesDF
      .withColumn("DiabetesClass", col("Diabetes").cast("int"))
      .groupBy("DiabetesClass")
      .agg(
        round(avg(col("Sex")) * 100, 1).alias("Male_Percent"),
        round(avg(col("AnyHealthcare")) * 100, 1).alias("Healthcare_Access_Percent")
      )
      .orderBy("DiabetesClass")

    sexAndHealthcareRates.show()
    spark.stop()
  }
}
