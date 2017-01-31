set term eps
set output "tanh_single_hidden_layer_loss.eps"
set title "Act fn: tanh, network: [784, h, 10], optimizer: sgd"

set xlabel "Number of epochs"
set ylabel "Validation Loss"
set key outside

plot "h1_100_tanh_sgd.txt" using 1:3 title "h=100" with linespoints, "h1_25_tanh_sgd.txt" using 1:3 title "h=25" with linespoints,"h1_5_tanh_sgd.txt" using 1:3 title "h=5" with linespoints

set term eps
set output "relu_single_hidden_layer_loss.eps"
set title "Act fn: relu, network: [784, h, 10], optimizer: sgd"

set xlabel "Number of epochs"
set ylabel "Validation Loss"
set key outside

plot "h1_100_sgd.txt" using 1:3 title "h=100" with linespoints, "h1_25_sgd.txt" using 1:3 title "h=25" with linespoints,"h1_5_sgd.txt" using 1:3 title "h=5" with linespoints

set term eps
set output "relu_tanh_single_hidden_layer_loss.eps"
set title "Act fn: a, network: [784, h, 10], optimizer: sgd"

set xlabel "Number of epochs"
set ylabel "Validation Loss"
set key outside

plot "h1_100_sgd.txt" using 1:3 title "h=100, a=relu" with linespoints, "h1_25_sgd.txt" using 1:3 title "h=25, a=relu" with linespoints, "h1_100_tanh_sgd.txt" using 1:3 title "h=100, a=tanh" with linespoints, "h1_25_tanh_sgd.txt" using 1:3 title "h=25, a=tanh" with linespoints


set term eps
set output "relu_multi_hidden_layer_loss.eps"
set title "Act fn: relu, network: [784, h, 10], optimizer: adam"

set xlabel "Number of epochs"
set ylabel "Validation Loss"
set key outside

plot "h0_adam.txt" using 1:3 title "h=[]" with linespoints, "h1_100_adam.txt" using 1:3 title "h=[100]" with linespoints,"h2_100_25_adam.txt" using 1:3 title "h=[100, 25]" with linespoints, "h2_100_50_25_adam.txt" using 1:3 title "h=[100, 50, 25]" with linespoints, "h4_100_50_25_25_adam.txt" using 1:3 title "h=[100, 50, 25, 25]" with linespoints pointsize 0.5

set term eps
set output "relu_25_opts_loss.eps"
set title "Act fn: relu, network: [784, 25, 10], optimizer: o"

set xlabel "Number of epochs"
set ylabel "Validation Loss"
set key outside

plot "h1_25_sgd.txt" using 1:3 title "o=sgd" with linespoints, "h1_25_mom.txt" using 1:3 title "o=momentum" with linespoints,"h1_25_adam.txt" using 1:3 title "o=adam" with linespoints

set term eps
set output "relu_100_opts_loss.eps"
set title "Act fn: relu, network: [784, 100, 10], optimizer: o"

set xlabel "Number of epochs"
set ylabel "Validation Loss"
set key outside

plot "h1_100_sgd.txt" using 1:3 title "o=sgd" with linespoints, "h1_100_mom.txt" using 1:3 title "o=momentum" with linespoints,"h1_100_adam.txt" using 1:3 title "o=adam" with linespoints

set term eps
set output "relu_100_25_opts_loss.eps"
set title "Act fn: relu, network: [784, 100, 25, 10], optimizer: o"

set xlabel "Number of epochs"
set ylabel "Validation Loss"
set key outside

plot "h2_100_25_adam.txt" using 1:3 title "o=adam" with linespoints, "h2_100_25_mom.txt" using 1:3 title "o=momentum" with linespoints

set term eps
set output "num_back_grads_relu.eps"
set title "Actfn: relu, network: [784, 100, 25, 10]"

set xlabel "Layers"
set ylabel "Mean difference"
set key outside

plot "num_grads.txt" using 1:2 title "diff" with linespoints

set term eps
set output "num_back_grads_relu_var.eps"
set title "Actfn: relu, network: [784, 100, 25, 10]"

set xlabel "Layers"
set ylabel "Mean squared difference"
set key outside

plot "var_num_grads.txt" using 1:2 title "diff" with linespoints