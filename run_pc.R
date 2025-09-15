library(pcalg)
library(readr)

args = commandArgs(trailingOnly=TRUE)

input_csv_path = args[1]
output_csv_path = args[2]
alpha = as.numeric(args[3])

# --- Load the data and run the PC algorithm ---
data = read_csv(input_csv_path, show_col_types = FALSE)
var_names = colnames(data)
suffStat = list(C = cor(data), n = nrow(data))

# Run the PC algorithm
pc_fit = pc(
  suffStat,
  indepTest = gaussCItest,
  labels = var_names,
  alpha = alpha, 
  solve.confl = T
)

# Extract the adjacency matrix
adj_matrix = as(pc_fit, "matrix")

# --- Save the result to a CSV file ---
write.csv(adj_matrix, output_csv_path, row.names = T)
