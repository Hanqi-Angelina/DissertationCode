# simple cleaning code provided together with the dataset
correct_coding <- function (x) {
  if (!is.factor(x)) stop("Must be a factor")
  
  levs <- levels(x)
  ct <- table(x)
  
  if (any(levs == "0") && any(ct == 0)) {
    levs0 <- setdiff(levs, "0") 
    levels(x) <- c(levs0, "x0")
    x <- droplevels(x)
  }
  else if (any(!is.na(suppressWarnings(as.numeric(levs)))) && any(ct == 0)) {
    levs0 <- levs[is.na(suppressWarnings(as.numeric(levs)))]
    levels(x) <- c(levs0, "x0")
    x <- droplevels(x)
  }
  
  return(x)
}