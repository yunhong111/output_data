#!/bin/bash
# Trie script

keyFile=~/Research/PCAP/mapped_caida_trace/BT_hao_Originalkey_prefix_key

ipfileFolder=~/Research/PCAP/mapped_caida_trace/

memSize=(200 250 300 320 340 360 380 410 450)

feedbackPortion=(0.01 0.05 0.1 0.2 0.4 0.6)

interval=(2000000 5000000 10000000 20000000 50000000 100000000)

blackKeySize=(500 1000 2000 4000 8000)

key_name_vec=(key_out_0_1_t1 key_out_0_1_t0.9 key_out_0_3_t1 key_out_0_3_t0.9 key_out_0_5_t1 key_out_0_5_t0.9 key_out_0_3_t0.7 key_out_0_5_t0.7 key_out_0_3_t0.8 key_out_0_4_t1 key_out_0_4_t0.9 key_out_0_4_t0.8 key_out_0_2_t1 key_out_0_2_t0.9 key_out_0_2_t0.8)

other_key_name_vec=(other_key_out_0_1_t1 other_key_out_0_1_t0.9 other_key_out_0_3_t1 other_key_out_0_3_t0.9 other_key_out_0_5_t1 other_key_out_0_5_t0.9 other_key_out_0_3_t0.7 other_key_out_0_5_t0.7 other_key_out_0_3_t0.8 other_key_out_0_4_t1 other_key_out_0_4_t0.9 other_key_out_0_4_t0.8 other_key_out_0_2_t1 other_key_out_0_2_t0.9 other_key_out_0_2_t0.8)

aggr_key_name_vec=(aggr_key_out_0_1_t1 aggr_key_out_0_1_t0.9 aggr_key_out_0_3_t1 aggr_key_out_0_3_t0.9 aggr_key_out_0_5_t1 aggr_key_out_0_5_t0.9 aggr_key_out_0_3_t0.7 aggr_key_out_0_5_t0.7 aggr_key_out_0_3_t0.8 aggr_key_out_0_4_t1 aggr_key_out_0_4_t0.9 aggr_key_out_0_4_t0.8 aggr_key_out_0_2_t1 aggr_key_out_0_2_t0.9 aggr_key_out_0_2_t0.8)

for i in $(seq 12 14)

do
	export OMP_NUM_THREADS=2
	
	key_name=../key/${key_name_vec[i]} 
	other_key_name=../key/${other_key_name_vec[i]}
	aggr_key_name=../key/${aggr_key_name_vec[i]}
	
		
	./trieNoiseMain ${keyFile} ${memSize[5]} ${ipfileFolder} ${feedbackPortion[2]} ${interval[0]} ${blackKeySize[2]} ${key_name} ${other_key_name} ${aggr_key_name} $i &

done






