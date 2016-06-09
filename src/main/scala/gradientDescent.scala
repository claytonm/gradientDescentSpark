import scala.collection.mutable.ArrayBuffer
import scala.io.Source
import breeze.linalg._
import breeze.numerics._
import java.io._

/**
  * Created by clay on 6/1/16.
  */

object GradientDescent {

  val trainingDataPath = "/Users/clay/education/mmds/data/ratings.train.txt"
  val testDataPath = "/Users/clay/education/mmds/data/ratings.val.txt"

  // class to store movie/user/rating tuples
  case class MovieAndUser(movie: Int, user: Int, rating: Int)
  // class to store total number of movies and total number of users
  case class Dimensions(movie: Int, user: Int)

  def extractMovieAndUser(line: String): MovieAndUser = {
    // input: String in format movie tab user tab rating
    // output: MovieAndUser object
    val movieUserPair = line.split("\t")
    val movie = movieUserPair(0).toInt
    val user = movieUserPair(1).toInt
    val rating = if (movieUserPair(2).isEmpty) 0 else movieUserPair(2).toInt
    MovieAndUser(movie, user, rating)
  }

  def getDimensions(filePath: String): Dimensions = {
    // input: filepath to training data
    // each line in training data is in format movie tab user tab rating
    var maxUser = 0
    var maxMovie = 0
    // for each line in file, if movieID or userID is greater than current max ID, then replace
    // maxUser or maxMovie with current user or movie
    for (line <- Source.fromFile(trainingDataPath).getLines()) {
      val movieUserPair = extractMovieAndUser(line)
      maxUser = math.max(maxUser, movieUserPair.user)
      maxMovie = math.max(maxMovie, movieUserPair.movie)
    }
    return (Dimensions(maxMovie, maxUser))
  }

  def getMaxRating(filePath: String): Tuple2[Int, Int] = {
    // input: file path to training data
    // output: maximum rating in dataset
    // this function is just for data QA: ratings should be between 0 and 5, inclusive
    var maxRating = 0
    var minRating = 0
    var numNAs = 0
    // for each line in file, if movieID or userID is greater than current max ID, then replace
    for (line <- Source.fromFile(trainingDataPath).getLines()) {
      val movieUserPair = extractMovieAndUser(line)
      maxRating = math.max(maxRating, movieUserPair.rating)
      minRating = math.min(minRating, movieUserPair.rating)
    }
    (maxRating, minRating)
  }

  def createMatrixQ(dimensions: Dimensions, k: Int): DenseMatrix[Double] = {
    // create matrix Q to hold movie estimates
    val maxValue = math.sqrt(5 / k.toDouble)
    val Q = DenseMatrix.rand(dimensions.movie, k) * maxValue
    Q
  }

  def createMatrixP(dimensions: Dimensions, k: Int): DenseMatrix[Double] = {
    // create matrix P to hold user estimates
    val maxValue = math.sqrt(5 / k.toDouble)
    val P = DenseMatrix.rand(dimensions.user, k) * maxValue
    P
  }

  def error(rating: Int, q: DenseVector[Double], p: DenseVector[Double]): Double = {
    // input: user rating, row of q matrix, row of p matrix
    // output: error (difference between user rating and estimate (q dot p)
    rating - q.dot(p)
  }

  def errorTerm(rating: Int, q: DenseVector[Double], p: DenseVector[Double]): Double = {
    // input: same as error function
    // output: error term to use to calculate error formula
    math.pow(rating - q.dot(p), 2)
  }

  def L2NormSquared(m: DenseMatrix[Double]): Double = {
    // input: matrix of type Double
    // output: L2 Norm of matrix
    sum(pow(m,2))
  }


  def updateQ(q: DenseVector[Double],
              p: DenseVector[Double],
              learningRate: Double,
              error: Double,
              lambda: Double): DenseVector[Double] = {
    // input: row of matrix Q, row of matrix P, learning rate, error, lambda
    // output: updated value of row q
    q + learningRate * (error * p - lambda * q)
  }

  def updateP(q: DenseVector[Double],
              p: DenseVector[Double],
              learningRate: Double,
              error: Double,
              lambda: Double): DenseVector[Double] = {
    // input: row of matrix Q, row of matrix P, learning rate, error, lambda
    // output: updated value of row q
    p + learningRate * (error * q - lambda * p)
  }

  def updateMatrices(Q: DenseMatrix[Double],
                     P: DenseMatrix[Double],
                     source: String,
                     learningRate: Double,
                     lambda: Double): Unit = {
    // calculates on full iteration of gradient descent
    // ouput: None
    // side effects: update matrices Q and P
    for (line <- Source.fromFile(source).getLines()) {
      val movieAndUser = extractMovieAndUser(line)

      val p = P(movieAndUser.user - 1, ::).t
      val q = Q(movieAndUser.movie - 1, ::).t
      val e = error(movieAndUser.rating, q, p)

      val qUpdate = updateQ(q, p, learningRate, e, lambda)
      val pUpdate = updateP(q, p, learningRate, e, lambda)

      P(movieAndUser.user - 1, ::) := pUpdate.t
      Q(movieAndUser.movie - 1, ::) := qUpdate.t
    }
  }

  def calculateError(Q: DenseMatrix[Double],
                     P: DenseMatrix[Double],
                     source: String,
                     learningRate: Double,
                     lambda: Double): Double = {
    // calculate error after an iteration of gradient descent
    // calculateError and updateMatrices comprise one full
    // iteration of gradient descent
    var errorsIterK = 0.0
    for (line <- Source.fromFile(source).getLines()) {
      val movieAndUser = extractMovieAndUser(line)
      val p = P(movieAndUser.user - 1, ::).t
      val q = Q(movieAndUser.movie - 1, ::).t
      val eTerm = errorTerm(movieAndUser.rating, q, p)
      errorsIterK = errorsIterK + eTerm
    }
    errorsIterK + lambda*(L2NormSquared(P) + L2NormSquared(Q))
  }


  // search for the best learning rate by starting at 0.01 and increasing by 0.01
  // number of iterations
  val iterations = 40
  // get number of users and movies
  val dimensions = getDimensions(trainingDataPath)
  // lambda determines how much we penalize for over-fitting
  val lambda = 0.2
  // k is the number of columns of P and Q
  val k = 20
  val trials = 15 // number of learning rates to try
  var learningRate = 0.01 // starting learning rate
  var bestLearningRate = learningRate // current best learning rate
  // class to represent a  (learning rate, error vector) pair
  case class errorPair(learningRate: Double, errors: DenseVector[Double])
  var learningRateTable = new ArrayBuffer[errorPair](trials)

  var P = createMatrixP(dimensions, k) // initial value of P
  var Q = createMatrixQ(dimensions, k)// initial value of Q
  var currentErrors = DenseVector.zeros[Double](iterations) // error vector for each trial
  var currentError = Inf

  // get errors for starting learning rate
  for (i <- 1 to iterations) {
    updateMatrices(Q, P, trainingDataPath, learningRate, lambda)
    currentErrors(i) = calculateError(Q, P, trainingDataPath, learningRate, lambda)
  }

  var bestError = currentErrors(iterations - 1)
  learningRateTable += errorPair(learningRate, currentErrors)

  // loop through different learning rates to find optimal
  for (i <- 1 to trials) {
    learningRate = 0.01 + learningRate
    currentErrors = DenseVector.zeros[Double](iterations)
    P = createMatrixP(dimensions, k)
    Q = createMatrixQ(dimensions, k)
    for (i <- 1 to iterations) {
      updateMatrices(Q, P, trainingDataPath, learningRate, lambda)
      currentErrors(i) = calculateError(Q, P, trainingDataPath, learningRate, lambda)
    }
    currentError = currentErrors(iterations - 1)
    if (!(currentError.isNaN) & currentError < bestError) {
      bestError = currentError
      bestLearningRate = learningRate
    }
    learningRateTable += errorPair(learningRate, currentErrors)
  }

  // print out results for graphing in R
  val pw = new PrintWriter(new File("errors.txt" ))
  learningRateTable.foreach(line => pw.write(line.learningRate + "," + line.errors.toArray.mkString(",") + "\n"))
  pw.close
}
