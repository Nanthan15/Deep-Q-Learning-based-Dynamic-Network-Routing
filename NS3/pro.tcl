set ns [new Simulator]

set nf [open pro.nam w]
$ns namtrace-all $nf

proc finish {} {
    global ns nf
    $ns flush-trace
    close $nf
    exec nam pro.nam &
    exit 0
}

# Create nodes
set n0 [$ns node]
set n1 [$ns node]
set n2 [$ns node]
set n3 [$ns node]
set n4 [$ns node]
set n5 [$ns node]

# Create links between nodes with specified weights
$ns duplex-link $n0 $n1 10Mb 10ms DropTail
$ns duplex-link $n1 $n5 10Mb 10ms DropTail
$ns duplex-link $n5 $n4 10Mb 10ms DropTail
$ns duplex-link $n4 $n3 10Mb 10ms DropTail
$ns duplex-link $n3 $n2 10Mb 10ms DropTail
$ns duplex-link $n2 $n0 10Mb 10ms DropTail
$ns duplex-link $n0 $n4 10Mb 10ms DropTail
$ns duplex-link $n2 $n4 10Mb 10ms DropTail

# Set link orientations for NAM visualization
$ns duplex-link-op $n0 $n1 orient right
$ns duplex-link-op $n1 $n5 orient down
$ns duplex-link-op $n5 $n4 orient left
$ns duplex-link-op $n4 $n3 orient left
$ns duplex-link-op $n3 $n2 orient up
$ns duplex-link-op $n2 $n0 orient right
$ns duplex-link-op $n0 $n4 orient down
$ns duplex-link-op $n2 $n4 orient right-up

# Adding weights to the links (only for reference, not used in NS2 simulation)
$ns at 0.0 "$n0 label 4"
$ns at 0.0 "$n1 label 2"
$ns at 0.0 "$n5 label 3"
$ns at 0.0 "$n4 label 7"
$ns at 0.0 "$n3 label 8"
$ns at 0.0 "$n2 label 12"
$ns at 0.0 "$n0 label 15"
$ns at 0.0 "$n4 label 1"

# Create TCP agent and attach it to node 3 (start node)
set tcp [new Agent/TCP]
$tcp set class_ 2
$ns attach-agent $n3 $tcp

# Create TCP sink and attach it to node 0 (destination node)
set sink [new Agent/TCPSink]
$ns attach-agent $n0 $sink
$ns connect $tcp $sink

# Configure the route along the optimal path [3, 4, 5, 1, 0]
$ns rtproto Static
$ns at 0.0 "$ns route-to $n3 $n4"
$ns at 0.0 "$ns route-to $n4 $n5"
$ns at 0.0 "$ns route-to $n5 $n1"
$ns at 0.0 "$ns route-to $n1 $n0"

# Create FTP application and start transmission
set ftp [new Application/FTP]
$ftp attach-agent $tcp
$ftp set type_ FTP
$ftp set packet_size_ 1000
$ftp set rate_ 1Mb

$ns at 1.0 "$ftp start"
$ns at 4.0 "$ftp stop"

$ns at 5.0 "finish"

$ns run
