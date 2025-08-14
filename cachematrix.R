## This script defines two functions that work together to cache the inverse of a matrix.
## Caching avoids repeating the costly computation of a matrix inverse when it has already been calculated.





## makeCacheMatrix creates a special "matrix" object that can store a matrix and its cached inverse.
## It returns a list of four functions to set/get the matrix and set/get its inverse.
makeCacheMatrix <- function(x = matrix()) {
  inv <- NULL
  set <- function(y) {
    x <<- y
    inv <<- NULL
  }
  get <- function() x
  setinverse <- function(inverse_matrix) inv <<- inverse_matrix
  getinverse <- function() inv
  list(set = set, get = get,
       setinverse = setinverse,
       getinverse = getinverse)


## cacheSolve computes the inverse of the matrix from a "matrix" object created by makeCacheMatrix.
## If the inverse has already been calculated and the matrix has not changed, it retrieves the cached inverse.

cacheSolve <- function(x, ...) {
  inverse_matrix <- x$getinverse()
  if(!is.null(inverse_matrix)) {
    message("getting cached inverse")
    return(inverse_matrix)
  }
  data <- x$get()
  inverse_matrix <- solve(data, ...)
  x$setinverse(inverse_matrix)
  inverse_matrix
}
