

#################################################################
#---------         Load libraries            -------------------#
#################################################################


#-- Import the spark session
from pyspark.sql import SparkSession

#################################################################
#---------         Initialize context        -------------------#
#################################################################


#-- Create spark context
spark = SparkSession.builder.appName("Operations").getOrCreate()

#################################################################
#---------         Read data                 -------------------#
#################################################################


#- Read csv
df = spark.read.csv('data.csv',inferSchema=True,header=True)

#- Read parquet
df = spark.read.parquet('data.parquet')

#- Read json
df = spark.read.json('data.json')


#-- Define a table schema
from pyspark.sql.types import StructField,StringType,IntegerType,StructType

#-- StructField(Column_name, data type, nullable flag)
data_schema = [StructField("age", IntegerType(), True),StructField("name", StringType(), True)]
final_struc = StructType(fields=data_schema)

#-- Read with predefined schema
df = spark.read.json('data.json', schema = final_struc)




#################################################################
#---------         Data exploration        ---------------------#
#################################################################


#-- Show first rows
df.head(2)

#-- Print the schema
df.printSchema()

#-- Column names
df.columns

#-- Number of rows
df.count()

#-- Check column type
type(df.select('age'))


#-- Select 1
df.select("Close").show()
#-- Select multiple
df.select(['Open','Close']).show()

#-- Filter 1 condition
df.filter("Close<500").show()
#-- Filter multiple conditions
df.filter( (df["Close"] < 200) & (df['Open'] > 200) ).show()
#-- Filter inequality
df.filter(df["Low"] == 197.16).show()

#-- To save a result use .collect()
result = df.filter(df["Low"] == 197.16).collect()



#################################################################
#---------         Data munging            ---------------------#
#################################################################


#-- Create new column 
df.withColumn('newage',df['age']/2).show()


#-- Rename column
df.withColumnRenamed('age','supernewage').show()


#-- Group by with operation
df.groupBy("Company").mean().show()
df.groupBy("Company").count().show()
df.groupBy("Company").max().show()
df.groupBy("Company").min().show()
df.groupBy("Company").sum().show()

#-- Group by single columns
df.agg({'Sales':'max'}).show()

grouped = df.groupBy("Company")
grouped.agg({"Sales":'max'}).show()


#-- Functions
from pyspark.sql.functions import countDistinct, avg,stddev, format_number,dayofmonth,hour,dayofyear,month,year,weekofyear,date_format

#-- Apply functions directly
df.select(countDistinct("Sales")).show()
df.select(avg('Sales')).show()
df.select(stddev("Sales")).show()
df.select(dayofmonth(df['Date'])).show()
df.select(hour(df['Date'])).show()
df.withColumn("Year",year(df['Date'])).show()


#-- Rename the result
df.select(countDistinct("Sales").alias("Distinct Sales")).show()



#-- Order by 

#-- (default is ascending)
df.orderBy(df["Sales"].desc()).show()




#################################################################
#---------         Use SQL                 ---------------------#
#################################################################


# Register the DataFrame as a SQL temporary view
df.createOrReplaceTempView("people")

#-- SQL operations using spark.sql
spark.sql("SELECT * FROM people WHERE age=30").show()



#################################################################
#---------         Missing data            ---------------------#
#################################################################


#-- Drop any row with missing data
df.na.drop().show()

# Has to have at least 2 NON-null values
df.na.drop(thresh=2).show()

#-- Drops row only if Sales is NA
df.na.drop(subset=["Sales"]).show()

#-- Drops if any col is NA
df.na.drop(how='any').show()

#-- Drops if all cols are NA
df.na.drop(how='all').show()




#-- Fille NA
from pyspark.sql.functions import mean

#-- Get mean of the column
mean_val = df.select(mean(df['Sales'])).collect()
mean_sales = mean_val[0][0]

#-- Fill with NA
df.na.fill(mean_sales,["Sales"]).show()




#################################################################
#---------         Modeling                ---------------------#
#################################################################


from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Import VectorAssembler and Vectors
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

#-- Defining the feature cols
assembler = VectorAssembler(
  inputCols=['Apps',
             'Accept',
             'Grad_Rate'],
              outputCol="features")

#-- Assemble
output = assembler.transform(data)


from pyspark.ml.feature import StringIndexer
indexer = StringIndexer(inputCol="Private", outputCol="target")
output_fixed = indexer.fit(output).transform(output)
final_data = output_fixed.select("features",'target')

#-- Split data
(trainingData, testData) = final_data.randomSplit([0.7, 0.3])

#-- Setup the models
dtc = DecisionTreeClassifier(labelCol='target',featuresCol='features')
rfc = RandomForestClassifier(labelCol='target',featuresCol='features')
gbt = GBTClassifier(labelCol='target',featuresCol='features')


#-- Train
dtc_model = dtc.fit(train_data)
rfc_model = rfc.fit(train_data)
gbt_model = gbt.fit(train_data)


#-- Predict
dtc_predictions = dtc_model.transform(test_data)
rfc_predictions = rfc_model.transform(test_data)
gbt_predictions = gbt_model.transform(test_data)


#-- Metrics
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


#-- Initialize an accuracy measure
acc_evaluator = MulticlassClassificationEvaluator(labelCol="target", predictionCol="prediction", metricName="accuracy")


#-- Evaluate accuracy
dtc_acc = acc_evaluator.evaluate(dtc_predictions)
rfc_acc = acc_evaluator.evaluate(rfc_predictions)
gbt_acc = acc_evaluator.evaluate(gbt_predictions)











