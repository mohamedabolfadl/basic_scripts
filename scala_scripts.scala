

///////////////////////////////////////////////////////////////////
//---------         Load libraries            -------------------//
///////////////////////////////////////////////////////////////////


//-- Import the spark session
import org.apache.spark.sql.SparkSession

///////////////////////////////////////////////////////////////////
//---------         Initialize context        -------------------//
///////////////////////////////////////////////////////////////////


//-- Create spark context
val spark = SparkSession.builder().getOrCreate()

///////////////////////////////////////////////////////////////////
//---------         Read data                 -------------------//
///////////////////////////////////////////////////////////////////


//- Read csv
val df = spark.read.option("header","true").option("inferSchema","true").csv("CitiGroup2006_2008")



///////////////////////////////////////////////////////////////////
//---------         Data exploration        ---------------------//
///////////////////////////////////////////////////////////////////


//-- Show first rows
df.head(2)

//-- Print the schema
df.printSchema()

//-- Print the column names
df.columns



// This import is needed to use the $-notation
import spark.implicits._


// Select
df.select($"Date",$"Close").show(2)


//-- Filter 
df.filter($"Close" > 480).show()
df.filter("Close > 480").count()
df.filter("Close<480 AND High < 484.40").show()

// Count how many results
df.filter($"Close">480).count()


//-- To save a result use .collect()
result = df.filter("Low == 197.16").collect()


