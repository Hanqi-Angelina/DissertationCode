library(readr)
library(pcalg)

args = commandArgs(trailingOnly=TRUE)

input_csv_path = args[1]
output_csv_path = args[2]
alpha = as.numeric(args[3])

# --- Load the data and run the GES algorithm ---
data = read_csv(input_csv_path, show_col_types = FALSE)
var_names = colnames(data)

# Create a score object for the GES algorithm, use the BIC score, no alpha needed
score = new("GaussL0penObsScore", data = data)

# Run the GES algorithm
ges_fit = ges(
  score = score
)

# Extract the adjacency matrix from the fitted object

# 1. Convert the graph object to the default logical matrix
logical_adj_matrix <- as(ges_fit$essgraph, "matrix")

# 2. Convert this logical matrix to an integer matrix (0s and 1s)
integer_adj_matrix <- logical_adj_matrix * 1

# 3. Get the canonical node order from the graph object
node_order <- ges_fit$essgraph$.nodes

# 4. Assign the node names to the rows and columns of your integer matrix
colnames(integer_adj_matrix) <- node_order
rownames(integer_adj_matrix) <- node_order

# --- Save the result to a CSV file ---
write.csv(integer_adj_matrix, output_csv_path, row.names = T)